import json
import logging
import numbers
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from docket.dependencies import Perpetual
from redis.asyncio import Redis
from ulid import ULID

from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.extraction import (
    extract_memories_with_strategy,
    handle_extraction,
)
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    EventDate,
    LastAccessed,
    MemoryHash,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.llms import (
    AnthropicClientWrapper,
    OpenAIClientWrapper,
    get_model_client,
    optimize_query_for_vector_search,
)
from agent_memory_server.models import (
    ExtractedMemoryRecord,
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.recency import (
    _days_between,
    generate_memory_hash,
    rerank_with_recency,
    update_memory_hash_if_text_changed,
)
from agent_memory_server.utils.redis import get_redis_conn
from agent_memory_server.vectorstore_factory import get_vectorstore_adapter


def _parse_extraction_response_with_fallback(content: str, logger) -> dict:
    """
    Parse JSON response with fallback mechanisms for malformed responses.

    Args:
        content: The JSON content to parse
        logger: Logger instance for error reporting

    Returns:
        Parsed JSON dictionary with 'memories' key

    Raises:
        json.JSONDecodeError: If all parsing attempts fail
    """
    # Try standard JSON parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Attempt to repair common JSON issues
        logger.warning(
            f"Initial JSON parsing failed, attempting repair on content: {content[:500]}..."
        )

        # Try to extract just the memories array if it exists
        memories_match = re.search(r'"memories"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if memories_match:
            try:
                # Try to reconstruct a valid JSON object
                memories_json = '{"memories": [' + memories_match.group(1) + "]}"
                extraction_result = json.loads(memories_json)
                logger.info("Successfully repaired malformed JSON response")
                return extraction_result
            except json.JSONDecodeError:
                logger.error("JSON repair attempt failed")
                raise
        else:
            logger.error("Could not find memories array in malformed response")
            raise


# Prompt for extracting memories from messages in working memory context
WORKING_MEMORY_EXTRACTION_PROMPT = """
You are a memory extraction assistant. Your job is to analyze conversation
messages and extract information that might be useful in future conversations.

Extract two types of memories from the following message:
1. EPISODIC: Experiences or events that have a time dimension.
   (They MUST have a time dimension to be "episodic.")
   Example: "User mentioned they visited Paris in August of 2025" or "User had trouble with the login process on 2025-01-15"

2. SEMANTIC: User preferences, facts, or general knowledge about the agent's
   environment that might be useful long-term.
   Example: "User prefers dark mode UI" or "User works as a data scientist"

For each memory, return a JSON object with the following fields:
- type: str -- The memory type, either "episodic" or "semantic"
- text: str -- The actual information to store
- topics: list[str] -- Relevant topics for this memory
- entities: list[str] -- Named entities mentioned
- event_date: str | null -- For episodic memories, the date/time when the event occurred (ISO 8601 format), null for semantic memories

IMPORTANT RULES:
1. Only extract information that might be genuinely useful for future interactions.
2. Do not extract procedural knowledge or instructions.
3. If given `user_id`, focus on user-specific information, preferences, and facts.
4. Return an empty list if no useful memories can be extracted.

Message: {message}

Return format:
{{
    "memories": [
        {{
            "type": "episodic",
            "text": "...",
            "topics": ["..."],
            "entities": ["..."],
            "event_date": "2024-01-15T14:30:00Z"
        }},
        {{
            "type": "semantic",
            "text": "...",
            "topics": ["..."],
            "entities": ["..."],
            "event_date": null
        }}
    ]
}}

Extracted memories:
"""


logger = logging.getLogger(__name__)

# Debounce configuration for thread-aware extraction
EXTRACTION_DEBOUNCE_TTL = 300  # 5 minutes
EXTRACTION_DEBOUNCE_KEY_PREFIX = "extraction_debounce"


async def should_extract_session_thread(session_id: str, redis: Redis) -> bool:
    """
    Check if enough time has passed since last thread-aware extraction for this session.

    This implements a debounce mechanism to avoid constantly re-extracting memories
    from the same conversation thread as new messages arrive.

    Args:
        session_id: The session ID to check
        redis: Redis client

    Returns:
        True if extraction should proceed, False if debounced
    """

    debounce_key = f"{EXTRACTION_DEBOUNCE_KEY_PREFIX}:{session_id}"

    # Check if debounce key exists
    exists = await redis.exists(debounce_key)
    if not exists:
        # Set debounce key with TTL to prevent extraction for the next period
        await redis.setex(debounce_key, EXTRACTION_DEBOUNCE_TTL, "extracting")
        logger.info(
            f"Starting thread-aware extraction for session {session_id} (debounce set for {EXTRACTION_DEBOUNCE_TTL}s)"
        )
        return True

    remaining_ttl = await redis.ttl(debounce_key)
    logger.info(
        f"Skipping thread-aware extraction for session {session_id} (debounced, {remaining_ttl}s remaining)"
    )
    return False


async def extract_memories_from_session_thread(
    session_id: str,
    namespace: str | None = None,
    user_id: str | None = None,
    llm_client: OpenAIClientWrapper | AnthropicClientWrapper | None = None,
) -> list[MemoryRecord]:
    """
    Extract memories from the entire conversation thread in working memory.

    This provides full conversational context for proper contextual grounding,
    allowing pronouns and references to be resolved across the entire thread.

    Args:
        session_id: The session ID to extract memories from
        namespace: Optional namespace for the memories
        user_id: Optional user ID for the memories
        llm_client: Optional LLM client for extraction

    Returns:
        List of extracted memory records with proper contextual grounding
    """
    from agent_memory_server.working_memory import get_working_memory

    # Get the complete working memory thread
    working_memory = await get_working_memory(
        session_id=session_id, namespace=namespace, user_id=user_id
    )

    if not working_memory or not working_memory.messages:
        logger.info(f"No working memory messages found for session {session_id}")
        return []

    # Build full conversation context from all messages
    conversation_messages = []
    for msg in working_memory.messages:
        # Include role and content for better context
        role_prefix = (
            f"[{msg.role.upper()}]: " if hasattr(msg, "role") and msg.role else ""
        )
        conversation_messages.append(f"{role_prefix}{msg.content}")

    full_conversation = "\n".join(conversation_messages)

    logger.info(
        f"Extracting memories from {len(working_memory.messages)} messages in session {session_id}"
    )
    logger.debug(
        f"Full conversation context length: {len(full_conversation)} characters"
    )

    # Use the new memory strategy system for extraction
    from agent_memory_server.memory_strategies import get_memory_strategy

    try:
        # Get the discrete memory strategy for contextual grounding
        strategy = get_memory_strategy("discrete")

        # Extract memories using the strategy
        memories_data = await strategy.extract_memories(full_conversation)

        logger.info(
            f"Extracted {len(memories_data)} memories from session thread {session_id}"
        )

        # Convert to MemoryRecord objects
        extracted_memories = []
        for memory_data in memories_data:
            memory = MemoryRecord(
                id=str(ULID()),
                text=memory_data["text"],
                memory_type=memory_data.get("type", "semantic"),
                topics=memory_data.get("topics", []),
                entities=memory_data.get("entities", []),
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
                discrete_memory_extracted="t",  # Mark as extracted
            )
            extracted_memories.append(memory)

        return extracted_memories

    except Exception as e:
        logger.error(f"Error extracting memories from session thread {session_id}: {e}")
        return []


async def extract_memory_structure(memory: MemoryRecord):
    redis = await get_redis_conn()

    # Process messages for topic/entity extraction
    topics, entities = await handle_extraction(memory.text)

    merged_topics = memory.topics + topics if memory.topics else topics
    merged_entities = memory.entities + entities if memory.entities else entities

    # Convert lists to comma-separated strings for TAG fields
    topics_joined = ",".join(merged_topics) if merged_topics else ""
    entities_joined = ",".join(merged_entities) if merged_entities else ""

    await redis.hset(
        Keys.memory_key(memory.id),
        mapping={"topics": topics_joined, "entities": entities_joined},
    )  # type: ignore


async def merge_memories_with_llm(
    memories: list[MemoryRecord], llm_client: Any = None
) -> MemoryRecord:
    """
    Use an LLM to merge similar or duplicate memories.

    Args:
        memories: List of MemoryRecord objects to merge
        llm_client: Optional LLM client to use for merging

    Returns:
        A merged memory
    """
    # If there's only one memory, just return it
    if len(memories) == 1:
        return memories[0]

    user_ids = {memory.user_id for memory in memories if memory.user_id}

    if len(user_ids) > 1:
        raise ValueError("Cannot merge memories with different user IDs")

        # Create a unified set of topics and entities
    all_topics = set()
    all_entities = set()

    for memory in memories:
        if memory.topics:
            all_topics.update(memory.topics)

        if memory.entities:
            all_entities.update(memory.entities)

    # Get the memory texts for LLM prompt
    memory_texts = [m.text for m in memories]

    # Construct the LLM prompt
    instruction = """
    You are a memory merging assistant. Your job is to merge similar or
    duplicate memories.

    You will be given a list of memories. You will need to merge them into a
    single, coherent memory.
    """
    memory_list = "\n".join([f"{i}: {text}" for i, text in enumerate(memory_texts, 1)])

    prompt = f"""
    {instruction}

    The memories:
    {memory_list}

    The merged memory:
    """

    model_name = "gpt-4o-mini"

    if not llm_client:
        model_client: (
            OpenAIClientWrapper | AnthropicClientWrapper
        ) = await get_model_client(model_name)
    else:
        model_client = llm_client

    response = await model_client.create_chat_completion(
        model=model_name,
        prompt=prompt,  # type: ignore
    )

    # Extract the merged content
    merged_text = ""
    if response.choices and len(response.choices) > 0:
        # Handle different response formats
        if hasattr(response.choices[0], "message"):
            merged_text = response.choices[0].message.content
        elif hasattr(response.choices[0], "text"):
            merged_text = response.choices[0].text
        else:
            # Fallback if the structure is different
            merged_text = str(response.choices[0])

    def coerce_to_float(m: MemoryRecord, key: str) -> float:
        try:
            val = getattr(m, key)
        except AttributeError:
            val = time.time()
        if val is None:
            return time.time()
        if isinstance(val, datetime):
            return float(val.timestamp())
        return float(val)

    # Use the oldest creation timestamp
    created_at = min(coerce_to_float(m, "created_at") for m in memories)

    # Use the most recent last_accessed timestamp
    last_accessed = max(coerce_to_float(m, "last_accessed") for m in memories)

    # Prefer non-empty namespace, user_id, session_id from memories
    namespace = next((m.namespace for m in memories if m.namespace), None)
    user_id = next((m.user_id for m in memories if m.user_id), None)
    session_id = next((m.session_id for m in memories if m.session_id), None)

    # Get the memory type from the first memory
    memory_type = next((m.memory_type for m in memories if m.memory_type), "semantic")

    # Create the merged memory
    merged_memory = MemoryRecord(
        text=merged_text.strip(),
        id=str(ULID()),
        user_id=user_id,
        session_id=session_id,
        namespace=namespace,
        created_at=datetime.fromtimestamp(created_at, UTC),
        last_accessed=datetime.fromtimestamp(last_accessed, UTC),
        updated_at=datetime.now(UTC),
        topics=list(all_topics) if all_topics else None,
        entities=list(all_entities) if all_entities else None,
        memory_type=MemoryTypeEnum(memory_type),
        discrete_memory_extracted="t",
    )

    # Generate a new hash for the merged memory
    merged_memory.memory_hash = generate_memory_hash(merged_memory)

    return merged_memory


async def compact_long_term_memories(
    limit: int = 1000,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    llm_client: OpenAIClientWrapper | AnthropicClientWrapper | None = None,
    redis_client: Redis | None = None,
    vector_distance_threshold: float = 0.2,
    compact_hash_duplicates: bool = True,
    compact_semantic_duplicates: bool = True,
    perpetual: Perpetual = Perpetual(
        every=timedelta(minutes=settings.compaction_every_minutes), automatic=True
    ),
) -> int:
    """
    Compact long-term memories by merging duplicates and semantically similar memories.

    This function can identify and merge two types of duplicate memories:
    1. Hash-based duplicates: Memories with identical content (using memory_hash)
    2. Semantic duplicates: Memories with similar meaning but different text

    Returns the count of remaining memories after compaction.
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    if not llm_client:
        llm_client = await get_model_client(model_name="gpt-4o-mini")

    logger.info(
        f"Starting memory compaction: namespace={namespace}, "
        f"user_id={user_id}, session_id={session_id}, "
        f"hash_duplicates={compact_hash_duplicates}, "
        f"semantic_duplicates={compact_semantic_duplicates}"
    )

    # Build filters for memory queries
    filters = []
    if namespace:
        filters.append(f"@namespace:{{{namespace}}}")
    if user_id:
        filters.append(f"@user_id:{{{user_id}}}")
    if session_id:
        filters.append(f"@session_id:{{{session_id}}}")

    filter_str = " ".join(filters) if filters else "*"

    # Track metrics
    memories_merged = 0
    start_time = time.time()

    # Step 1: Compact hash-based duplicates using Redis aggregation
    if compact_hash_duplicates:
        logger.info("Starting hash-based duplicate compaction")
        try:
            index_name = Keys.search_index_name()

            # Create aggregation query to group by memory_hash and find duplicates
            agg_query = [
                "FT.AGGREGATE",
                index_name,
                filter_str,
                "GROUPBY",
                str(1),
                "@memory_hash",
                "REDUCE",
                "COUNT",
                str(0),
                "AS",
                "count",
                "FILTER",
                "@count>1",  # Only groups with more than 1 memory
                "SORTBY",
                str(2),
                "@count",
                "DESC",
                "LIMIT",
                str(0),
                str(limit),
            ]

            # Execute aggregation to find duplicate groups
            duplicate_groups = await redis_client.execute_command(*agg_query)

            if duplicate_groups and duplicate_groups[0] > 0:
                num_groups = duplicate_groups[0]
                logger.info(
                    f"Found {num_groups} groups with hash-based duplicates to process"
                )

                # Process each group of duplicates
                for i in range(1, len(duplicate_groups), 2):
                    try:
                        # Get the hash and count from aggregation results
                        group_data = duplicate_groups[i]
                        memory_hash = None
                        count = 0

                        for j in range(0, len(group_data), 2):
                            if group_data[j] == b"memory_hash":
                                if group_data[j + 1] is not None:
                                    memory_hash = group_data[j + 1].decode()
                            elif group_data[j] == b"count":
                                count = int(group_data[j + 1])

                        if not memory_hash or count <= 1:
                            continue

                        # Find all memories with this hash
                        # Use FT.SEARCH to find the actual memories with this hash
                        # TODO: Use RedisVL index
                        if filters:
                            # Combine hash query with filters using boolean AND
                            query_expr = f"(@memory_hash:{{{memory_hash}}}) ({' '.join(filters)})"
                        else:
                            query_expr = f"@memory_hash:{{{memory_hash}}}"

                        search_results = await redis_client.execute_command(
                            "FT.SEARCH",
                            index_name,
                            f"'{query_expr}'",
                            "RETURN",
                            "6",
                            "id_",
                            "text",
                            "last_accessed",
                            "created_at",
                            "user_id",
                            "session_id",
                            "SORTBY",
                            "last_accessed",
                            "ASC",
                        )

                        if search_results and search_results[0] > 1:
                            num_duplicates = search_results[0]

                            # Keep the newest memory (last in sorted results)
                            # and delete the rest
                            memories_to_delete = []

                            # Each memory result has: key + 6 field-value pairs = 13 elements
                            # Keys are at positions: 1, 14, 27, ... (1 + n * 13)
                            elements_per_memory = 1 + 6 * 2  # key + 6 field-value pairs
                            for n in range(num_duplicates):
                                key_index = 1 + n * elements_per_memory
                                # Skip the last item (newest) which we'll keep
                                if n < num_duplicates - 1 and key_index < len(
                                    search_results
                                ):
                                    key = search_results[key_index]
                                    if key is not None:
                                        key_str = (
                                            key.decode()
                                            if isinstance(key, bytes)
                                            else key
                                        )
                                        memories_to_delete.append(key_str)

                            # Delete older duplicates
                            if memories_to_delete:
                                pipeline = redis_client.pipeline()
                                for key in memories_to_delete:
                                    pipeline.delete(key)

                                await pipeline.execute()
                                memories_merged += len(memories_to_delete)
                                logger.info(
                                    f"Deleted {len(memories_to_delete)} hash-based duplicates "
                                    f"with hash {memory_hash}"
                                )
                    except Exception as e:
                        logger.error(f"Error processing duplicate group: {e}")
            else:
                logger.info("No hash-based duplicates found")

            logger.info(
                f"Completed hash-based deduplication. Removed {memories_merged} duplicate memories."
            )
        except Exception as e:
            logger.error(f"Error during hash-based duplicate compaction: {e}")

    # Step 2: Compact semantic duplicates using vector search
    semantic_memories_merged = 0
    if compact_semantic_duplicates:
        logger.info("Starting semantic duplicate compaction")
        # Get the correct index name
        index_name = Keys.search_index_name()
        logger.info(f"Using index '{index_name}' for semantic duplicate compaction.")

        # Index will be created automatically when we add memories if it doesn't exist
        # No need to check or create it explicitly

        # Get all memories using the vector store adapter
        try:
            # Convert filters to adapter format
            namespace_filter = None
            user_id_filter = None
            session_id_filter = None

            if namespace:
                from agent_memory_server.filters import Namespace

                namespace_filter = Namespace(eq=namespace)
            if user_id:
                from agent_memory_server.filters import UserId

                user_id_filter = UserId(eq=user_id)
            if session_id:
                from agent_memory_server.filters import SessionId

                session_id_filter = SessionId(eq=session_id)

            # Use vectorstore adapter to get all memories
            adapter = await get_vectorstore_adapter()
            search_result = await adapter.search_memories(
                query="",  # Empty query to get all matching filter criteria
                namespace=namespace_filter,
                user_id=user_id_filter,
                session_id=session_id_filter,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error searching for memories: {e}")
            search_result = None

        if search_result and search_result.memories:
            logger.info(
                f"Found {search_result.total} memories to check for semantic duplicates"
            )

            # Process memories in batches to avoid overloading
            batch_size = 50
            processed_ids = set()  # Track which memories have been processed

            memories_list = search_result.memories
            for i in range(0, len(memories_list), batch_size):
                batch = memories_list[i : i + batch_size]

                for memory_result in batch:
                    memory_id = memory_result.id

                    # Skip if already processed
                    if memory_id in processed_ids:
                        continue

                    # Convert MemoryRecordResult to MemoryRecord for deduplication
                    memory_obj = MemoryRecord(
                        id=memory_result.id,
                        text=memory_result.text,
                        user_id=memory_result.user_id,
                        session_id=memory_result.session_id,
                        namespace=memory_result.namespace,
                        created_at=memory_result.created_at,
                        last_accessed=memory_result.last_accessed,
                        topics=memory_result.topics or [],
                        entities=memory_result.entities or [],
                        memory_type=memory_result.memory_type,  # type: ignore
                        discrete_memory_extracted=memory_result.discrete_memory_extracted,  # type: ignore
                    )

                    # Add this memory to processed list BEFORE processing to prevent cycles
                    processed_ids.add(memory_id)

                    # Check for semantic duplicates
                    (
                        merged_memory,
                        was_merged,
                    ) = await deduplicate_by_semantic_search(
                        memory=memory_obj,
                        redis_client=redis_client,
                        llm_client=llm_client,
                        namespace=namespace,
                        user_id=user_id,
                        session_id=session_id,
                        vector_distance_threshold=vector_distance_threshold,
                    )

                    if was_merged:
                        semantic_memories_merged += 1
                        # Delete the original memory using the adapter
                        await adapter.delete_memories([memory_id])

                        # Re-index the merged memory
                        if merged_memory:
                            await index_long_term_memories(
                                [merged_memory],
                                redis_client=redis_client,
                                deduplicate=False,  # Already deduplicated
                            )
                            # Mark the merged memory as processed to prevent cycles
                            processed_ids.add(merged_memory.id)
        logger.info(
            f"Completed semantic deduplication. Merged {semantic_memories_merged} memories."
        )

    # Get the count of remaining memories
    total_memories = await count_long_term_memories(
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
        redis_client=redis_client,
    )

    end_time = time.time()
    total_merged = memories_merged + semantic_memories_merged

    logger.info(
        f"Memory compaction completed in {end_time - start_time:.2f}s. "
        f"Merged {total_merged} memories. "
        f"{total_memories} memories remain."
    )

    return total_memories


async def index_long_term_memories(
    memories: list[MemoryRecord | ExtractedMemoryRecord],
    redis_client: Redis | None = None,
    deduplicate: bool = False,
    vector_distance_threshold: float = 0.12,
    llm_client: Any = None,
) -> None:
    """
    Index long-term memories using the pluggable VectorStore adapter.

    Args:
        memories: List of long-term memories to index
        redis_client: Optional Redis client (kept for compatibility, may be unused depending on backend)
        deduplicate: Whether to deduplicate memories before indexing
        vector_distance_threshold: Threshold for semantic similarity
        llm_client: Optional LLM client for semantic merging
    """
    background_tasks = get_background_tasks()

    # Process memories for deduplication if requested
    processed_memories = []
    if deduplicate:
        # Get Redis client for deduplication operations (still needed for existing dedup logic)
        redis = redis_client or await get_redis_conn()
        model_client = llm_client or await get_model_client(
            model_name=settings.generation_model
        )

        for memory in memories:
            current_memory = memory
            was_deduplicated = False

            # Check for id-based duplicates
            if not was_deduplicated:
                deduped_memory, was_overwrite = await deduplicate_by_id(
                    memory=current_memory,
                    redis_client=redis,
                )
                if was_overwrite:
                    # This overwrote an existing memory with the same ID
                    current_memory = deduped_memory or current_memory
                    logger.info(f"Overwrote memory with ID {memory.id}")
                else:
                    current_memory = deduped_memory or current_memory

            # Check for hash-based duplicates
            if not was_deduplicated:
                deduped_memory, was_dup = await deduplicate_by_hash(
                    memory=current_memory,
                    redis_client=redis,
                )
                if was_dup:
                    # This is a duplicate, skip it
                    was_deduplicated = True
                else:
                    current_memory = deduped_memory or current_memory

            # Check for semantic duplicates
            if not was_deduplicated:
                deduped_memory, was_merged = await deduplicate_by_semantic_search(
                    memory=current_memory,
                    redis_client=redis,
                    llm_client=model_client,
                    vector_distance_threshold=vector_distance_threshold,
                )
                if was_merged:
                    current_memory = deduped_memory or current_memory

            # Add the memory to be indexed if not a pure duplicate
            if not was_deduplicated:
                processed_memories.append(current_memory)
    else:
        processed_memories = memories

    # If all memories were duplicates, we're done
    if not processed_memories:
        logger.info("All memories were duplicates, nothing to index")
        return

    # Get the VectorStore adapter and add memories
    adapter = await get_vectorstore_adapter()

    # Add memories to the vector store
    try:
        ids = await adapter.add_memories(processed_memories)
        logger.info(f"Indexed {len(processed_memories)} memories with IDs: {ids}")
    except Exception as e:
        logger.error(f"Error indexing memories: {e}")
        raise

    # Schedule background tasks for topic/entity extraction
    for memory in processed_memories:
        background_tasks.add_task(extract_memory_structure, memory)

    if settings.enable_discrete_memory_extraction:
        needs_extraction = [
            memory
            for memory in processed_memories
            if memory.discrete_memory_extracted == "f"
        ]
        # Extract discrete memories from the indexed messages and persist
        # them as separate long-term memory records. This process also
        # runs deduplication if requested.
        background_tasks.add_task(
            extract_memories_with_strategy,
            memories=needs_extraction,
            deduplicate=deduplicate,
        )


async def search_long_term_memories(
    text: str,
    session_id: SessionId | None = None,
    user_id: UserId | None = None,
    namespace: Namespace | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    distance_threshold: float | None = None,
    memory_type: MemoryType | None = None,
    event_date: EventDate | None = None,
    memory_hash: MemoryHash | None = None,
    server_side_recency: bool | None = None,
    recency_params: dict | None = None,
    limit: int = 10,
    offset: int = 0,
    optimize_query: bool = False,
) -> MemoryRecordResults:
    """
    Search for long-term memories using the pluggable VectorStore adapter.

    Args:
        text: Query for vector search - will be used for semantic similarity matching
        session_id: Optional session ID filter
        user_id: Optional user ID filter
        namespace: Optional namespace filter
        created_at: Optional created at filter
        last_accessed: Optional last accessed filter
        topics: Optional topics filter
        entities: Optional entities filter
        distance_threshold: Optional similarity threshold
        memory_type: Optional memory type filter
        event_date: Optional event date filter
        memory_hash: Optional memory hash filter
        limit: Maximum number of results
        offset: Offset for pagination
        optimize_query: Whether to optimize the query for vector search using a fast model (default: False)

    Returns:
        MemoryRecordResults containing matching memories
    """
    # Optimize query for vector search if requested.
    search_query = text
    optimized_applied = False
    if optimize_query and text:
        search_query = await optimize_query_for_vector_search(text)
        optimized_applied = True

    # Get the VectorStore adapter
    adapter = await get_vectorstore_adapter()

    # Delegate search to the adapter
    results = await adapter.search_memories(
        query=search_query,
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        created_at=created_at,
        last_accessed=last_accessed,
        topics=topics,
        entities=entities,
        memory_type=memory_type,
        event_date=event_date,
        memory_hash=memory_hash,
        distance_threshold=distance_threshold,
        server_side_recency=server_side_recency,
        recency_params=recency_params,
        limit=limit,
        offset=offset,
    )

    # If an optimized query with a strict distance threshold returns no results,
    # retry once with the original query to preserve recall.
    try:
        if (
            optimized_applied
            and distance_threshold is not None
            and results.total == 0
            and search_query != text
        ):
            results = await adapter.search_memories(
                query=text,
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                created_at=created_at,
                last_accessed=last_accessed,
                topics=topics,
                entities=entities,
                memory_type=memory_type,
                event_date=event_date,
                memory_hash=memory_hash,
                distance_threshold=distance_threshold,
                server_side_recency=server_side_recency,
                recency_params=recency_params,
                limit=limit,
                offset=offset,
            )
    except Exception:
        # Best-effort fallback; return the original results on any error
        pass

    return results


async def count_long_term_memories(
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Count the total number of long-term memories matching the given filters.

    Uses the pluggable VectorStore adapter instead of direct Redis calls.

    Args:
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        redis_client: Optional Redis client (for compatibility - not used by adapter)

    Returns:
        Total count of memories matching filters
    """
    # Get the VectorStore adapter
    adapter = await get_vectorstore_adapter()

    # Delegate to the adapter
    return await adapter.count_memories(
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
    )


async def deduplicate_by_hash(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory has hash-based duplicates and handle accordingly.

    Memories have a hash generated from their text and metadata. If we
    see the exact-same memory again, we ignore it.

    Args:
        memory: The memory to check for duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter

    Returns:
        Tuple of (memory to save (if any), was_duplicate)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # Generate hash for the memory
    memory_hash = generate_memory_hash(memory)

    # Use vectorstore adapter to search for memories with the same hash
    # Build filter objects
    namespace_filter = None
    if namespace or memory.namespace:
        namespace_filter = Namespace(eq=namespace or memory.namespace)

    user_id_filter = None
    if user_id or memory.user_id:
        user_id_filter = UserId(eq=user_id or memory.user_id)

    session_id_filter = None
    if session_id or memory.session_id:
        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Create memory hash filter
    memory_hash_filter = MemoryHash(eq=memory_hash)

    # Use vectorstore adapter to search for memories with the same hash
    adapter = await get_vectorstore_adapter()

    # Search for existing memories with the same hash
    # Use a dummy query since we're filtering by hash, not doing semantic search
    results = await adapter.search_memories(
        query="",  # Empty query since we're filtering by hash
        session_id=session_id_filter,
        user_id=user_id_filter,
        namespace=namespace_filter,
        memory_hash=memory_hash_filter,
        limit=1,  # We only need to know if one exists
    )

    if results.memories and len(results.memories) > 0:
        # Found existing memory with the same hash
        logger.info(f"Found existing memory with hash {memory_hash}")

        # Update the last_accessed timestamp of the existing memory
        existing_memory = results.memories[0]
        if existing_memory.id:
            # Use the memory key format to update last_accessed
            existing_key = Keys.memory_key(existing_memory.id)
            await redis_client.hset(
                existing_key,
                mapping={"last_accessed": str(int(datetime.now(UTC).timestamp()))},
            )  # type: ignore

            # Don't save this memory, it's a duplicate
            return None, True
    # No duplicates found, return the original memory
    return memory, False


async def deduplicate_by_id(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory with the same ID exists and deduplicate if found.

    When two memories have the same ID, the most recent memory replaces the
    oldest memory. (They are not merged.)

    Args:
        memory: The memory to check for ID duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter

    Returns:
        Tuple of (memory to save (potentially updated), was_overwrite)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # If no id, can't deduplicate by id
    if not memory.id:
        return memory, False

    # Use vectorstore adapter to search for memories with the same id
    # Build filter objects
    namespace_filter = None
    if namespace or memory.namespace:
        from agent_memory_server.filters import Namespace

        namespace_filter = Namespace(eq=namespace or memory.namespace)

    user_id_filter = None
    if user_id or memory.user_id:
        from agent_memory_server.filters import UserId

        user_id_filter = UserId(eq=user_id or memory.user_id)

    session_id_filter = None
    if session_id or memory.session_id:
        from agent_memory_server.filters import SessionId

        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Create id filter
    from agent_memory_server.filters import Id

    id_filter = Id(eq=memory.id)

    # Use vectorstore adapter to search for memories with the same id
    adapter = await get_vectorstore_adapter()

    # Search for existing memories with the same id
    # Use a dummy query since we're filtering by id, not doing semantic search
    results = await adapter.search_memories(
        query="",  # Empty query since we're filtering by id
        session_id=session_id_filter,
        user_id=user_id_filter,
        namespace=namespace_filter,
        id=id_filter,
        limit=1,  # We only need to know if one exists
    )

    if results.memories and len(results.memories) > 0:
        # Found existing memory with the same id
        existing_memory = results.memories[0]
        logger.info(f"Found existing memory with id {memory.id}, will overwrite")

        # If the existing memory was already persisted, preserve that timestamp
        if existing_memory.persisted_at:
            memory.persisted_at = existing_memory.persisted_at

        # Delete the existing memory using the adapter
        if existing_memory.id:
            await adapter.delete_memories([existing_memory.id])

        # Return the memory to be saved (overwriting the existing one)
        return memory, True

    # No existing memory with this id found
    return memory, False


async def deduplicate_by_semantic_search(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    llm_client: Any = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    vector_distance_threshold: float = 0.2,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory has semantic duplicates and merge if found.

    Unlike deduplicate_by_id, this function does not overwrite any existing
    memories. Instead, all semantically similar duplicates are merged.

    Args:
        memory: The memory to check for semantic duplicates
        redis_client: Optional Redis client
        llm_client: Optional LLM client for merging
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        vector_distance_threshold: Distance threshold for semantic similarity

    Returns:
        Tuple of (memory to save (potentially merged), was_merged)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    if not llm_client:
        llm_client = await get_model_client(model_name="gpt-4o-mini")

    # Use vector store adapter to find semantically similar memories
    adapter = await get_vectorstore_adapter()

    # Convert filters to adapter format
    namespace_filter = None
    user_id_filter = None
    session_id_filter = None

    # TODO: Refactor to avoid inline imports (fix circular imports)
    if namespace or memory.namespace:
        from agent_memory_server.filters import Namespace

        namespace_filter = Namespace(eq=namespace or memory.namespace)
    if user_id or memory.user_id:
        from agent_memory_server.filters import UserId

        user_id_filter = UserId(eq=user_id or memory.user_id)
    if session_id or memory.session_id:
        from agent_memory_server.filters import SessionId

        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Use the vectorstore adapter for semantic search
    # TODO: Paginate through results?
    search_result = await adapter.search_memories(
        query=memory.text,  # Use memory text for semantic search
        namespace=namespace_filter,
        user_id=user_id_filter,
        session_id=session_id_filter,
        distance_threshold=vector_distance_threshold,
        limit=10,
    )

    vector_search_result = search_result.memories if search_result else []

    # Filter out the memory itself from the search results (avoid self-duplication)
    vector_search_result = [m for m in vector_search_result if m.id != memory.id]

    if vector_search_result and len(vector_search_result) > 0:
        # Found semantically similar memories
        similar_memory_ids = [memory.id for memory in vector_search_result]

        # Merge the memories
        merged_memory = await merge_memories_with_llm(
            [memory] + vector_search_result,
            llm_client=llm_client,
        )

        # Delete the similar memories using the adapter
        if similar_memory_ids:
            await adapter.delete_memories(similar_memory_ids)

        logger.info(
            f"Merged new memory with {len(similar_memory_ids)} semantic duplicates"
        )
        return merged_memory, True

    # No similar memories found or error occurred
    return memory, False


async def promote_working_memory_to_long_term(
    session_id: str,
    namespace: str | None = None,
    user_id: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Promote eligible working memory records to long-term storage.

    This function:
    1. Identifies memory records with no persisted_at from working memory
    2. For message records, runs extraction to generate semantic/episodic memories
    3. Uses id to detect and replace duplicates in long-term memory
    4. Persists the record and stamps it with persisted_at = now()
    5. Updates the working memory session store to reflect new timestamps

    Args:
        session_id: The session ID to promote memories from
        namespace: Optional namespace for the session
        user_id: Optional user ID for the session
        redis_client: Optional Redis client to use

    Returns:
        Number of memories promoted to long-term storage
    """

    from agent_memory_server import working_memory
    from agent_memory_server.utils.redis import get_redis_conn

    redis = redis_client or await get_redis_conn()

    # Get current working memory
    current_working_memory = await working_memory.get_working_memory(
        session_id=session_id,
        namespace=namespace,
        user_id=user_id,
        redis_client=redis,
    )

    if not current_working_memory:
        logger.debug(f"No working memory found for session {session_id}")
        return 0

    logger.info("Promoting memories to long-term storage...")

    promoted_count = 0
    updated_memories = []
    extracted_memories = []

    # Thread-aware discrete memory extraction with debouncing
    unextracted_messages = [
        message
        for message in current_working_memory.messages
        if message.discrete_memory_extracted == "f"
    ]

    extracted_memories = []
    if settings.enable_discrete_memory_extraction and unextracted_messages:
        # Check if we should run thread-aware extraction (debounced)
        if await should_extract_session_thread(session_id, redis):
            logger.info(
                f"Running thread-aware extraction from {len(current_working_memory.messages)} total messages in session {session_id}"
            )
            extracted_memories = await extract_memories_from_session_thread(
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )

            # Mark ALL messages in the session as extracted since we processed the full thread
            for message in current_working_memory.messages:
                message.discrete_memory_extracted = "t"

        else:
            logger.info(f"Skipping extraction for session {session_id} - debounced")

    # Combine existing memories with newly extracted memories for processing
    all_memories_to_process = list(current_working_memory.memories)
    if extracted_memories:
        logger.info(
            f"Adding {len(extracted_memories)} extracted memories for promotion"
        )
        all_memories_to_process.extend(extracted_memories)

    for memory in all_memories_to_process:
        if memory.persisted_at is None:
            # This memory needs to be promoted

            # Check for id-based duplicates and handle accordingly
            deduped_memory, was_overwrite = await deduplicate_by_id(
                memory=memory,
                redis_client=redis,
            )

            # Set persisted_at timestamp
            current_memory = deduped_memory or memory
            current_memory.persisted_at = datetime.now(UTC)

            # Set extraction strategy configuration from working memory
            current_memory.extraction_strategy = (
                current_working_memory.long_term_memory_strategy.strategy
            )
            current_memory.extraction_strategy_config = (
                current_working_memory.long_term_memory_strategy.config
            )

            # Index the memory in long-term storage
            await index_long_term_memories(
                [current_memory],
                redis_client=redis,
                deduplicate=False,  # Already deduplicated by id
            )

            promoted_count += 1
            updated_memories.append(current_memory)

            if was_overwrite:
                logger.info(f"Overwrote existing memory with id {memory.id}")
            else:
                logger.info(f"Promoted new memory with id {memory.id}")
        else:
            # This memory is already persisted, keep as-is
            updated_memories.append(memory)

    count_persisted_messages = 0
    message_records_to_index = []

    # Process unpersisted messages if configured to do so
    if settings.index_all_messages_in_long_term_memory:
        updated_messages = []
        for msg in current_working_memory.messages:
            if msg.persisted_at is None:
                # Skip messages with empty or None content
                if not msg.content or not msg.content.strip():
                    logger.warning(f"Skipping message with empty content: {msg.id}")
                    updated_messages.append(msg)
                    continue

                # Generate ID if not present (backward compatibility)
                if not msg.id:
                    msg.id = str(ULID())

                memory_record = MemoryRecord(
                    id=msg.id,
                    session_id=session_id,
                    text=f"{msg.role}: {msg.content}",
                    namespace=namespace,
                    user_id=current_working_memory.user_id,
                    persisted_at=None,
                    created_at=msg.created_at,
                    memory_type=MemoryTypeEnum.MESSAGE,
                )

                # Apply same deduplication logic as structured memories
                deduped_memory, was_overwrite = await deduplicate_by_id(
                    memory=memory_record,
                    redis_client=redis,
                )

                # Set persisted_at timestamp
                current_memory = deduped_memory or memory_record
                current_memory.persisted_at = datetime.now(UTC)

                # Set extraction strategy configuration from working memory
                current_memory.extraction_strategy = "message"

                # Collect memory record for batch indexing
                message_records_to_index.append(current_memory)

                # Update message with persisted_at timestamp
                msg.persisted_at = current_memory.persisted_at
                promoted_count += 1

                if was_overwrite:
                    logger.info(
                        f"Overwrote existing long-term message memory with ID {msg.id}"
                    )
                else:
                    logger.info(
                        f"Promoted new long-term message memory with ID {msg.id}"
                    )

            updated_messages.append(msg)

        # Batch index all new memory records for messages
        if message_records_to_index:
            count_persisted_messages = len(message_records_to_index)
            await index_long_term_memories(
                message_records_to_index,
                redis_client=redis,
                deduplicate=False,  # Already deduplicated by ID
            )
    else:
        count_persisted_messages = 0
        updated_messages = current_working_memory.messages

    # Check if any messages were marked as extracted
    messages_marked_extracted = (
        settings.enable_discrete_memory_extraction
        and unextracted_messages
        and await should_extract_session_thread(session_id, redis)
    )

    # Update working memory with the new persisted_at timestamps and extracted memories
    if (
        promoted_count > 0
        or extracted_memories
        or count_persisted_messages > 0
        or messages_marked_extracted
    ):
        updated_working_memory = current_working_memory.model_copy()
        updated_working_memory.memories = updated_memories
        updated_working_memory.messages = updated_messages
        updated_working_memory.updated_at = datetime.now(UTC)

        await working_memory.set_working_memory(
            working_memory=updated_working_memory,
            redis_client=redis,
        )

        logger.info(
            f"Successfully promoted {promoted_count} memories and {len(message_records_to_index)} messages to long-term storage"
            + (
                f" and extracted {len(extracted_memories)} new memories"
                if extracted_memories
                else ""
            )
        )

    return promoted_count


async def delete_long_term_memories(
    ids: list[str],
) -> int:
    """
    Delete long-term memories by ID.
    """
    adapter = await get_vectorstore_adapter()
    return await adapter.delete_memories(ids)


async def get_long_term_memory_by_id(memory_id: str) -> MemoryRecord | None:
    """
    Get a single long-term memory by its ID.

    Args:
        memory_id: The ID of the memory to retrieve

    Returns:
        MemoryRecord if found, None if not found
    """
    from agent_memory_server.filters import Id

    adapter = await get_vectorstore_adapter()

    # Search for the memory by ID using the existing search function
    results = await adapter.search_memories(
        query="",  # Empty search text to get all results
        limit=1,
        id=Id(eq=memory_id),
    )

    if results.memories:
        return results.memories[0]
    return None


async def update_long_term_memory(
    memory_id: str,
    updates: dict[str, Any],
) -> MemoryRecord | None:
    """
    Update a long-term memory by ID.

    Args:
        memory_id: The ID of the memory to update
        updates: Dictionary of fields to update

    Returns:
        Updated MemoryRecord if found and updated, None if not found

    Raises:
        ValueError: If the update contains invalid fields
    """
    # First, get the existing memory
    existing_memory = await get_long_term_memory_by_id(memory_id)
    if not existing_memory:
        return None

    # Valid fields that can be updated
    updatable_fields = {
        "text",
        "topics",
        "entities",
        "memory_type",
        "namespace",
        "user_id",
        "session_id",
        "event_date",
    }

    # Validate update fields
    invalid_fields = set(updates.keys()) - updatable_fields
    if invalid_fields:
        raise ValueError(
            f"Cannot update fields: {invalid_fields}. Valid fields: {updatable_fields}"
        )

    # Create updated memory record using efficient model_copy and hash helper
    base_updates = {**updates, "updated_at": datetime.now(UTC)}
    update_dict = update_memory_hash_if_text_changed(existing_memory, base_updates)
    updated_memory = existing_memory.model_copy(update=update_dict)

    # Update in the vectorstore
    adapter = await get_vectorstore_adapter()
    await adapter.update_memories([updated_memory])

    return updated_memory


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int, float, or other number type)."""
    return isinstance(value, numbers.Number)


def select_ids_for_forgetting(
    results: Iterable[MemoryRecordResult],
    *,
    policy: dict,
    now: datetime,
    pinned_ids: set[str] | None = None,
) -> list[str]:
    """Select IDs for deletion based on TTL, inactivity and budget policies.

    Policy keys:
      - max_age_days: float | None
      - max_inactive_days: float | None
      - budget: int | None (keep top N by recency score)
      - memory_type_allowlist: set[str] | list[str] | None (only consider these types for deletion)
      - hard_age_multiplier: float (default 12.0) - multiplier for max_age_days to determine extremely old items
    """
    pinned_ids = pinned_ids or set()
    max_age_days = policy.get("max_age_days")
    max_inactive_days = policy.get("max_inactive_days")
    hard_age_multiplier = float(policy.get("hard_age_multiplier", 12.0))
    budget = policy.get("budget")
    allowlist = policy.get("memory_type_allowlist")
    if allowlist is not None and not isinstance(allowlist, set):
        allowlist = set(allowlist)

    to_delete: set[str] = set()
    eligible_for_budget: list[MemoryRecordResult] = []

    for mem in results:
        if not mem.id or mem.id in pinned_ids or getattr(mem, "pinned", False):
            continue

        # If allowlist provided, only consider those types for deletion
        mem_type_value = (
            mem.memory_type.value
            if isinstance(mem.memory_type, MemoryTypeEnum)
            else mem.memory_type
        )
        if allowlist is not None and mem_type_value not in allowlist:
            # Not eligible for deletion under current policy
            continue

        age_days = _days_between(now, mem.created_at)
        inactive_days = _days_between(now, mem.last_accessed)

        # Combined TTL/inactivity policy:
        # - If both thresholds are set, prefer not to delete recently accessed
        #   items unless they are extremely old.
        # - Extremely old: age > max_age_days * hard_age_multiplier (default 12x)
        if _is_numeric(max_age_days) and _is_numeric(max_inactive_days):
            if age_days > float(max_age_days) * hard_age_multiplier:
                to_delete.add(mem.id)
                continue
            if age_days > float(max_age_days) and inactive_days > float(
                max_inactive_days
            ):
                to_delete.add(mem.id)
                continue
        else:
            ttl_hit = _is_numeric(max_age_days) and age_days > float(max_age_days)
            inactivity_hit = _is_numeric(max_inactive_days) and (
                inactive_days > float(max_inactive_days)
            )
            if ttl_hit or inactivity_hit:
                to_delete.add(mem.id)
                continue

        # Eligible for budget consideration
        eligible_for_budget.append(mem)

    # Budget-based pruning (keep top N by recency among eligible)
    if isinstance(budget, int) and budget >= 0 and budget < len(eligible_for_budget):
        params = {
            "semantic_weight": 0.0,  # budget considers only recency
            "recency_weight": 1.0,
            "freshness_weight": 0.6,
            "novelty_weight": 0.4,
            "half_life_last_access_days": 7.0,
            "half_life_created_days": 30.0,
        }
        ranked = rerank_with_recency(eligible_for_budget, now=now, params=params)
        keep_ids = {mem.id for mem in ranked[:budget]}
        for mem in eligible_for_budget:
            if mem.id not in keep_ids:
                to_delete.add(mem.id)

    return list(to_delete)


async def update_last_accessed(
    ids: list[str],
    *,
    redis_client: Redis | None = None,
    min_interval_seconds: int = 900,
) -> int:
    """Rate-limited update of last_accessed for a list of memory IDs.

    Returns the number of records updated.
    """
    if not ids:
        return 0

    redis = redis_client or await get_redis_conn()
    now_ts = int(datetime.now(UTC).timestamp())

    # Batch read existing last_accessed
    keys = [Keys.memory_key(mid) for mid in ids]
    pipeline = redis.pipeline()
    for key in keys:
        pipeline.hget(key, "last_accessed")
    current_vals = await pipeline.execute()

    # Decide which to update and whether to increment access_count
    to_update: list[tuple[str, int]] = []
    incr_keys: list[str] = []
    for key, val in zip(keys, current_vals, strict=False):
        try:
            last_ts = int(val) if val is not None else 0
        except (TypeError, ValueError):
            last_ts = 0
        if now_ts - last_ts >= min_interval_seconds:
            to_update.append((key, now_ts))
            incr_keys.append(key)

    if not to_update:
        return 0

    pipeline2 = redis.pipeline()
    for key, ts in to_update:
        pipeline2.hset(key, mapping={"last_accessed": str(ts)})
        pipeline2.hincrby(key, "access_count", 1)
    await pipeline2.execute()
    return len(to_update)


async def forget_long_term_memories(
    policy: dict,
    *,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 1000,
    dry_run: bool = True,
    pinned_ids: list[str] | None = None,
) -> dict:
    """Select and delete long-term memories according to policy.

    Uses the vectorstore adapter to fetch candidates (empty query + filters),
    then applies `select_ids_for_forgetting` locally and deletes via adapter.
    """
    adapter = await get_vectorstore_adapter()

    # Build filters
    namespace_filter = Namespace(eq=namespace) if namespace else None
    user_id_filter = UserId(eq=user_id) if user_id else None
    session_id_filter = SessionId(eq=session_id) if session_id else None

    # Fetch candidates with an empty query honoring filters
    results = await adapter.search_memories(
        query="",
        namespace=namespace_filter,
        user_id=user_id_filter,
        session_id=session_id_filter,
        limit=limit,
    )

    now = datetime.now(UTC)
    candidate_results = results.memories or []

    # Select IDs for deletion using policy
    to_delete_ids = select_ids_for_forgetting(
        candidate_results,
        policy=policy,
        now=now,
        pinned_ids=set(pinned_ids) if pinned_ids else None,
    )

    deleted = 0
    if to_delete_ids and not dry_run:
        deleted = await adapter.delete_memories(to_delete_ids)

    return {
        "scanned": len(candidate_results),
        "deleted": deleted if not dry_run else len(to_delete_ids),
        "deleted_ids": to_delete_ids,
        "dry_run": dry_run,
    }


async def periodic_forget_long_term_memories(
    *,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 1000,
    dry_run: bool = False,
    perpetual: Perpetual = Perpetual(
        every=timedelta(minutes=settings.forgetting_every_minutes), automatic=True
    ),
) -> dict:
    """Periodic forgetting using defaults from settings.

    This function can be registered with Docket and will run automatically
    according to the `perpetual` schedule when a worker is active.
    """
    # Build default policy from settings
    policy: dict[str, object] = {
        "max_age_days": settings.forgetting_max_age_days,
        "max_inactive_days": settings.forgetting_max_inactive_days,
        "budget": settings.forgetting_budget_keep_top_n,
        "memory_type_allowlist": None,
    }

    # If feature disabled, no-op
    if not settings.forgetting_enabled:
        logger.info("Forgetting is disabled; skipping periodic run")
        return {"scanned": 0, "deleted": 0, "deleted_ids": [], "dry_run": True}

    return await forget_long_term_memories(
        policy,
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        dry_run=dry_run,
    )
