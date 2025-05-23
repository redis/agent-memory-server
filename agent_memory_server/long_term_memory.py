import hashlib
import logging
import time
from functools import reduce
from typing import Any

import nanoid
from redis.asyncio import Redis
from redis.commands.search.query import Query
from redisvl.query import VectorQuery, VectorRangeQuery
from redisvl.utils.vectorize import OpenAITextVectorizer

from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.extraction import extract_discrete_memories, handle_extraction
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
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
)
from agent_memory_server.models import (
    LongTermMemory,
    LongTermMemoryResult,
    LongTermMemoryResults,
)
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import (
    ensure_search_index_exists,
    get_redis_conn,
    get_search_index,
    safe_get,
)


DEFAULT_MEMORY_LIMIT = 1000
MEMORY_INDEX = "memory_idx"


logger = logging.getLogger(__name__)


async def extract_memory_structure(_id: str, text: str, namespace: str | None):
    redis = await get_redis_conn()

    # Process messages for topic/entity extraction
    topics, entities = await handle_extraction(text)

    # Convert lists to comma-separated strings for TAG fields
    topics_joined = ",".join(topics) if topics else ""
    entities_joined = ",".join(entities) if entities else ""

    await redis.hset(
        Keys.memory_key(_id, namespace),
        mapping={
            "topics": topics_joined,
            "entities": entities_joined,
        },
    )  # type: ignore


def generate_memory_hash(memory: dict) -> str:
    """
    Generate a stable hash for a memory based on text, user_id, and session_id.

    Args:
        memory: Dictionary containing memory data

    Returns:
        A stable hash string
    """
    # Create a deterministic string representation of the key fields
    text = memory.get("text", "")
    user_id = memory.get("user_id", "") or ""
    session_id = memory.get("session_id", "") or ""

    # Combine the fields in a predictable order
    hash_content = f"{text}|{user_id}|{session_id}"

    # Create a stable hash
    return hashlib.sha256(hash_content.encode()).hexdigest()


async def merge_memories_with_llm(memories: list[dict], llm_client: Any = None) -> dict:
    """
    Use an LLM to merge similar or duplicate memories.

    Args:
        memories: List of memory dictionaries to merge
        llm_client: Optional LLM client to use for merging

    Returns:
        A merged memory dictionary
    """
    # If there's only one memory, just return it
    if len(memories) == 1:
        return memories[0]

    # Create a unified set of topics and entities
    all_topics = set()
    all_entities = set()

    for memory in memories:
        if memory.get("topics"):
            if isinstance(memory["topics"], str):
                all_topics.update(memory["topics"].split(","))
            else:
                all_topics.update(memory["topics"])

        if memory.get("entities"):
            if isinstance(memory["entities"], str):
                all_entities.update(memory["entities"].split(","))
            else:
                all_entities.update(memory["entities"])

    # Get the memory texts for LLM prompt
    memory_texts = [m["text"] for m in memories]

    # Construct the LLM prompt
    instruction = "Merge these similar memories into a single, coherent memory:"

    prompt = f"{instruction}\n\n"
    for i, text in enumerate(memory_texts, 1):
        prompt += f"Memory {i}: {text}\n\n"

    prompt += "\nMerged memory:"

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

    # Use the earliest creation timestamp
    created_at = min(int(m.get("created_at", int(time.time()))) for m in memories)

    # Use the most recent last_accessed timestamp
    last_accessed = max(int(m.get("last_accessed", int(time.time()))) for m in memories)

    # Prefer non-empty namespace, user_id, session_id from memories
    namespace = next((m["namespace"] for m in memories if m.get("namespace")), None)
    user_id = next((m["user_id"] for m in memories if m.get("user_id")), None)
    session_id = next((m["session_id"] for m in memories if m.get("session_id")), None)

    # Get the memory type from the first memory
    memory_type = next(
        (m["memory_type"] for m in memories if m.get("memory_type")), "semantic"
    )

    # Get the discrete_memory_extracted from the first memory
    discrete_memory_extracted = next(
        (
            m["discrete_memory_extracted"]
            for m in memories
            if m.get("discrete_memory_extracted")
        ),
        "t",
    )

    # Create the merged memory
    merged_memory = {
        "text": merged_text.strip(),
        "id_": nanoid.generate(),
        "user_id": user_id,
        "session_id": session_id,
        "namespace": namespace,
        "created_at": created_at,
        "last_accessed": last_accessed,
        "updated_at": int(time.time()),
        "topics": list(all_topics) if all_topics else None,
        "entities": list(all_entities) if all_entities else None,
        "memory_type": memory_type,
        "discrete_memory_extracted": discrete_memory_extracted,
    }

    # Generate a new hash for the merged memory
    merged_memory["memory_hash"] = generate_memory_hash(merged_memory)

    return merged_memory


async def compact_long_term_memories(
    limit: int = 1000,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    llm_client: OpenAIClientWrapper | AnthropicClientWrapper | None = None,
    redis_client: Redis | None = None,
    vector_distance_threshold: float = 0.12,
    compact_hash_duplicates: bool = True,
    compact_semantic_duplicates: bool = True,
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
            agg_query = (
                f"FT.AGGREGATE {index_name} {filter_str} "
                "GROUPBY 1 @memory_hash "
                "REDUCE COUNT 0 AS count "
                'FILTER "@count>1" '  # Only groups with more than 1 memory
                "SORTBY 2 @count DESC "
                f"LIMIT 0 {limit}"
            )

            # Execute aggregation to find duplicate groups
            duplicate_groups = await redis_client.execute_command(agg_query)

            if duplicate_groups and duplicate_groups[0] > 0:
                num_groups = duplicate_groups[0]
                logger.info(f"Found {num_groups} groups of hash-based duplicates")

                # Process each group of duplicates
                for i in range(1, len(duplicate_groups), 2):
                    try:
                        # Get the hash and count from aggregation results
                        group_data = duplicate_groups[i]
                        memory_hash = None
                        count = 0

                        for j in range(0, len(group_data), 2):
                            if group_data[j] == b"memory_hash":
                                memory_hash = group_data[j + 1].decode()
                            elif group_data[j] == b"count":
                                count = int(group_data[j + 1])

                        if not memory_hash or count <= 1:
                            continue

                        # Find all memories with this hash
                        # Use FT.SEARCH to find the actual memories with this hash
                        # TODO: Use RedisVL index
                        search_query = (
                            f"FT.SEARCH {index_name} "
                            f"(@memory_hash:{{{memory_hash}}}) {' '.join(filters)} "
                            "RETURN 6 id_ text last_accessed created_at user_id session_id "
                            "SORTBY last_accessed ASC"  # Oldest first
                        )

                        search_results = await redis_client.execute_command(
                            search_query
                        )

                        if search_results and search_results[0] > 1:
                            num_duplicates = search_results[0]

                            # Keep the newest memory (last in sorted results)
                            # and delete the rest
                            memories_to_delete = []

                            for j in range(1, len(search_results), 2):
                                # Skip the last item (newest) which we'll keep
                                if j < (int(num_duplicates) - 1) * 2 + 1:
                                    key = search_results[j].decode()
                                    memories_to_delete.append(key)

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

            logger.info(
                f"Completed hash-based deduplication. Merged {memories_merged} memories."
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

        # Check if the index exists before proceeding
        try:
            await redis_client.execute_command(f"FT.INFO {index_name}")
        except Exception as info_e:
            if "unknown index name" in str(info_e).lower():
                logger.info(f"Search index {index_name} doesn't exist, creating it")
                # Ensure 'get_search_index' is called with the correct name to create it if needed
                await ensure_search_index_exists(redis_client, index_name=index_name)
            else:
                logger.warning(
                    f"Error checking index '{index_name}': {info_e} - attempting to proceed."
                )

        # Get all memories matching the filters, using the correct index name
        index = get_search_index(redis_client, index_name=index_name)
        query_str = filter_str if filter_str != "*" else "*"

        # Create a query to get all memories
        q = Query(query_str).paging(0, limit)
        q.return_fields("id_", "text", "vector", "user_id", "session_id", "namespace")

        # Execute the query to get memories
        search_result = None
        try:
            search_result = await index.search(q)
        except Exception as e:
            logger.error(f"Error searching for memories: {e}")

        if search_result and search_result.total > 0:
            logger.info(
                f"Found {search_result.total} memories to check for semantic duplicates"
            )

            # Process memories in batches to avoid overloading Redis
            batch_size = 50
            processed_keys = set()  # Track which memories have been processed

            for i in range(0, len(search_result.docs), batch_size):
                batch = search_result.docs[i : i + batch_size]

                for memory in batch:
                    memory_key = safe_get(memory, "id")  # We get the Redis key as "id"
                    memory_id = safe_get(memory, "id_")  # This is our own generated ID

                    # Skip if already processed
                    if memory_key in processed_keys:
                        continue

                    # Get memory data with error handling
                    memory_data = {}
                    try:
                        memory_data_raw = await redis_client.hgetall(memory_key)  # type: ignore
                        if memory_data_raw:
                            # Convert memory data from bytes to strings
                            memory_data = {
                                k.decode() if isinstance(k, bytes) else k: v
                                if isinstance(v, bytes)
                                and (k == b"vector" or k == "vector")
                                else v.decode()
                                if isinstance(v, bytes)
                                else v
                                for k, v in memory_data_raw.items()
                            }
                    except Exception as e:
                        logger.error(f"Error retrieving memory {memory_key}: {e}")
                        continue

                    # Skip if memory not found
                    if not memory_data:
                        continue

                    # Convert to LongTermMemory object for deduplication
                    memory_obj = LongTermMemory(
                        id_=memory_id,
                        text=memory_data.get("text", ""),
                        user_id=memory_data.get("user_id"),
                        session_id=memory_data.get("session_id"),
                        namespace=memory_data.get("namespace"),
                        created_at=int(memory_data.get("created_at", 0)),
                        last_accessed=int(memory_data.get("last_accessed", 0)),
                        topics=memory_data.get("topics", "").split(",")
                        if memory_data.get("topics")
                        else [],
                        entities=memory_data.get("entities", "").split(",")
                        if memory_data.get("entities")
                        else [],
                        memory_type=memory_data.get("memory_type", "semantic"),
                        discrete_memory_extracted=memory_data.get(
                            "discrete_memory_extracted", "t"
                        ),
                    )

                    # Add this memory to processed list
                    processed_keys.add(memory_key)

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
                        # We need to delete the original memory and save the merged one
                        await redis_client.delete(memory_key)

                        # Re-index the merged memory
                        await index_long_term_memories(
                            [merged_memory],
                            redis_client=redis_client,
                            deduplicate=False,  # Already deduplicated
                        )
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
    memories: list[LongTermMemory],
    redis_client: Redis | None = None,
    deduplicate: bool = False,
    vector_distance_threshold: float = 0.12,
    llm_client: Any = None,
) -> None:
    """
    Index long-term memories in Redis for search, with optional deduplication

    Args:
        memories: List of long-term memories to index
        redis_client: Optional Redis client to use. If None, a new connection will be created.
        deduplicate: Whether to deduplicate memories before indexing
        vector_distance_threshold: Threshold for semantic similarity
        llm_client: Optional LLM client for semantic merging
    """
    redis = redis_client or await get_redis_conn()
    model_client = (
        llm_client or await get_model_client(model_name=settings.generation_model)
        if deduplicate
        else None
    )
    background_tasks = get_background_tasks()

    # Process memories for deduplication if requested
    processed_memories = []
    if deduplicate:
        for memory in memories:
            current_memory = memory
            was_deduplicated = False

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

    # Now proceed with indexing the processed memories
    vectorizer = OpenAITextVectorizer()
    embeddings = await vectorizer.aembed_many(
        [memory.text for memory in processed_memories],
        batch_size=20,
        as_buffer=True,
    )

    async with redis.pipeline(transaction=False) as pipe:
        for idx, vector in enumerate(embeddings):
            memory = processed_memories[idx]
            id_ = memory.id_ if memory.id_ else nanoid.generate()
            key = Keys.memory_key(id_, memory.namespace)

            # Generate memory hash for the memory
            memory_hash = generate_memory_hash(
                {
                    "text": memory.text,
                    "user_id": memory.user_id or "",
                    "session_id": memory.session_id or "",
                }
            )
            print("Memory hash: ", memory_hash)

            await pipe.hset(  # type: ignore
                key,
                mapping={
                    "text": memory.text,
                    "id_": id_,
                    "session_id": memory.session_id or "",
                    "user_id": memory.user_id or "",
                    "last_accessed": memory.last_accessed or int(time.time()),
                    "created_at": memory.created_at or int(time.time()),
                    "namespace": memory.namespace or "",
                    "memory_hash": memory_hash,  # Store the hash for aggregation
                    "memory_type": memory.memory_type,
                    "vector": vector,
                    "discrete_memory_extracted": memory.discrete_memory_extracted,
                },
            )

            await background_tasks.add_task(
                extract_memory_structure, id_, memory.text, memory.namespace
            )

        await pipe.execute()

    logger.info(f"Indexed {len(processed_memories)} memories")
    if settings.enable_discrete_memory_extraction:
        # Extract discrete memories from the indexed messages and persist
        # them as separate long-term memory records. This process also
        # runs deduplication if requested.
        await background_tasks.add_task(
            extract_discrete_memories,
            deduplicate=deduplicate,
        )


async def search_long_term_memories(
    text: str,
    redis: Redis,
    session_id: SessionId | None = None,
    user_id: UserId | None = None,
    namespace: Namespace | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    distance_threshold: float | None = None,
    memory_type: MemoryType | None = None,
    limit: int = 10,
    offset: int = 0,
) -> LongTermMemoryResults:
    """
    Search for long-term memories using vector similarity and filters.
    """
    vectorizer = OpenAITextVectorizer()
    vector = await vectorizer.aembed(text)
    filters = []

    if session_id:
        filters.append(session_id.to_filter())
    if user_id:
        filters.append(user_id.to_filter())
    if namespace:
        filters.append(namespace.to_filter())
    if created_at:
        filters.append(created_at.to_filter())
    if last_accessed:
        filters.append(last_accessed.to_filter())
    if topics:
        filters.append(topics.to_filter())
    if entities:
        filters.append(entities.to_filter())
    if memory_type:
        filters.append(memory_type.to_filter())
    filter_expression = reduce(lambda x, y: x & y, filters) if filters else None

    if distance_threshold is not None:
        q = VectorRangeQuery(
            vector=vector,
            vector_field_name="vector",
            distance_threshold=distance_threshold,
            num_results=limit,
            return_score=True,
            return_fields=[
                "text",
                "id_",
                "dist",
                "created_at",
                "last_accessed",
                "user_id",
                "session_id",
                "namespace",
                "topics",
                "entities",
                "memory_type",
                "memory_hash",
            ],
        )
    else:
        q = VectorQuery(
            vector=vector,
            vector_field_name="vector",
            num_results=limit,
            return_score=True,
            return_fields=[
                "text",
                "id_",
                "dist",
                "created_at",
                "last_accessed",
                "user_id",
                "session_id",
                "namespace",
                "topics",
                "entities",
                "memory_type",
                "memory_hash",
            ],
        )
    if filter_expression:
        q.set_filter(filter_expression)

    q.paging(offset=offset, num=limit)

    index = get_search_index(redis)
    search_result = await index.query(q)

    results = []
    memory_hashes = []

    for doc in search_result:
        if safe_get(doc, "memory_hash") not in memory_hashes:
            memory_hashes.append(safe_get(doc, "memory_hash"))
        else:
            continue

        # NOTE: Because this may not be obvious. We index hashes, and we extract
        # topics and entities separately from main long-term indexing. However,
        # when we store the topics and entities, we store them as comma-separated
        # strings in the hash. Our search index picks these up and indexes them
        # in TAG fields, and we get them back as comma-separated strings.
        doc_topics = safe_get(doc, "topics", [])
        if isinstance(doc_topics, str):
            doc_topics = doc_topics.split(",")  # type: ignore

        doc_entities = safe_get(doc, "entities", [])
        if isinstance(doc_entities, str):
            doc_entities = doc_entities.split(",")  # type: ignore

        results.append(
            LongTermMemoryResult(
                id_=safe_get(doc, "id_"),
                text=safe_get(doc, "text", ""),
                dist=float(safe_get(doc, "vector_distance", 0)),
                created_at=int(safe_get(doc, "created_at", 0)),
                updated_at=int(safe_get(doc, "updated_at", 0)),
                last_accessed=int(safe_get(doc, "last_accessed", 0)),
                user_id=safe_get(doc, "user_id"),
                session_id=safe_get(doc, "session_id"),
                namespace=safe_get(doc, "namespace"),
                topics=doc_topics,
                entities=doc_entities,
                memory_hash=safe_get(doc, "memory_hash"),
                memory_type=safe_get(doc, "memory_type", "message"),
            )
        )

    # Handle different types of search_result
    total_results = len(results)
    if hasattr(search_result, "total"):
        total_results = search_result.total

    logger.info(f"Found {len(results)} results for query")
    return LongTermMemoryResults(
        total=total_results,
        memories=results,
        next_offset=offset + limit if offset + limit < total_results else None,
    )


async def count_long_term_memories(
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Count the total number of long-term memories matching the given filters.

    Args:
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        redis_client: Optional Redis client

    Returns:
        Total count of memories matching filters
    """
    # TODO: Use RedisVL here.
    if not redis_client:
        redis_client = await get_redis_conn()

    # Build filters for the query
    filters = []
    if namespace:
        filters.append(f"@namespace:{{{namespace}}}")
    if user_id:
        filters.append(f"@user_id:{{{user_id}}}")
    if session_id:
        filters.append(f"@session_id:{{{session_id}}}")

    filter_str = " ".join(filters) if filters else "*"

    # Execute a search to get the total count
    index_name = Keys.search_index_name()
    query = f"FT.SEARCH {index_name} {filter_str} LIMIT 0 0"

    try:
        # First try to check if the index exists
        try:
            await redis_client.execute_command(f"FT.INFO {index_name}")
        except Exception as info_e:
            if "unknown index name" in str(info_e).lower():
                # Index doesn't exist, create it
                logger.info(f"Search index {index_name} doesn't exist, creating it")
                await ensure_search_index_exists(redis_client)
            else:
                logger.warning(f"Error checking index: {info_e}")

        result = await redis_client.execute_command(query)
        # First element in the result is the total count
        if result and len(result) > 0:
            return result[0]
        return 0
    except Exception as e:
        logger.error(f"Error counting memories: {e}")
        return 0


async def deduplicate_by_hash(
    memory: LongTermMemory,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[LongTermMemory | None, bool]:
    """
    Check if a memory has hash-based duplicates and handle accordingly.

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
    memory_hash = generate_memory_hash(
        {
            "text": memory.text,
            "user_id": memory.user_id or "",
            "session_id": memory.session_id or "",
        }
    )

    # Build filters for the search
    filters = []
    if namespace or memory.namespace:
        ns = namespace or memory.namespace
        filters.append(f"@namespace:{{{ns}}}")
    if user_id or memory.user_id:
        uid = user_id or memory.user_id
        filters.append(f"@user_id:{{{uid}}}")
    if session_id or memory.session_id:
        sid = session_id or memory.session_id
        filters.append(f"@session_id:{{{sid}}}")

    filter_str = " ".join(filters) if filters else ""

    # Search for existing memories with the same hash
    index_name = Keys.search_index_name()

    # Use FT.SEARCH to find memories with this hash
    # TODO: Use RedisVL
    search_query = (
        f"FT.SEARCH {index_name} "
        f"(@memory_hash:{{{memory_hash}}}) {filter_str} "
        "RETURN 1 id_ "
        "SORTBY last_accessed DESC"  # Newest first
    )

    search_results = await redis_client.execute_command(search_query)

    if search_results and search_results[0] > 0:
        # Found existing memory with the same hash
        logger.info(f"Found existing memory with hash {memory_hash}")

        # Update the last_accessed timestamp of the existing memory
        if search_results[0] >= 1:
            existing_key = search_results[1].decode()
            await redis_client.hset(
                existing_key, "last_accessed", str(int(time.time()))
            )  # type: ignore

            # Don't save this memory, it's a duplicate
            return None, True

    # No duplicates found, return the original memory
    return memory, False


async def deduplicate_by_semantic_search(
    memory: LongTermMemory,
    redis_client: Redis | None = None,
    llm_client: Any = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    vector_distance_threshold: float = 0.12,
) -> tuple[LongTermMemory | None, bool]:
    """
    Check if a memory has semantic duplicates and merge if found.

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

    # Get the vector for the memory
    vectorizer = OpenAITextVectorizer()
    vector = await vectorizer.aembed(memory.text, as_buffer=True)

    # Build filters
    filter_expression = None
    if namespace or memory.namespace:
        ns = namespace or memory.namespace
        filter_expression = Namespace(eq=ns).to_filter()
    if user_id or memory.user_id:
        uid = user_id or memory.user_id
        user_filter = UserId(eq=uid).to_filter()
        filter_expression = (
            user_filter
            if filter_expression is None
            else filter_expression & user_filter
        )
    if session_id or memory.session_id:
        sid = session_id or memory.session_id
        session_filter = SessionId(eq=sid).to_filter()
        filter_expression = (
            session_filter
            if filter_expression is None
            else filter_expression & session_filter
        )

    # Use vector search to find semantically similar memories
    index = get_search_index(redis_client)

    vector_query = VectorRangeQuery(
        vector=vector,
        vector_field_name="vector",
        distance_threshold=vector_distance_threshold,
        num_results=5,
        return_fields=[
            "id_",
            "text",
            "user_id",
            "session_id",
            "namespace",
            "id",
            "created_at",
            "last_accessed",
            "topics",
            "entities",
            "memory_type",
        ],
    )

    if filter_expression:
        vector_query.set_filter(filter_expression)

    vector_search_result = await index.query(vector_query)

    if vector_search_result and len(vector_search_result) > 0:
        # Found semantically similar memories
        similar_memory_keys = []
        for similar_memory in vector_search_result:
            similar_memory_keys.append(similar_memory["id"])
            similar_memory["created_at"] = similar_memory.get(
                "created_at", int(time.time())
            )
            similar_memory["last_accessed"] = similar_memory.get(
                "last_accessed", int(time.time())
            )
            # Merge the memories
            merged_memory = await merge_memories_with_llm(
                [memory.model_dump()] + [similar_memory],
                llm_client=llm_client,
            )

            # Convert back to LongTermMemory
            merged_memory_obj = LongTermMemory(
                id_=memory.id_ or nanoid.generate(),
                text=merged_memory["text"],
                user_id=merged_memory["user_id"],
                session_id=merged_memory["session_id"],
                namespace=merged_memory["namespace"],
                created_at=merged_memory["created_at"],
                last_accessed=merged_memory["last_accessed"],
                topics=merged_memory.get("topics", []),
                entities=merged_memory.get("entities", []),
                memory_type=merged_memory.get("memory_type", "semantic"),
                discrete_memory_extracted=merged_memory.get(
                    "discrete_memory_extracted", "t"
                ),
            )

            # Delete the similar memories if requested
            for key in similar_memory_keys:
                await redis_client.delete(key)

        logger.info(
            f"Merged new memory with {len(similar_memory_keys)} semantic duplicates"
        )
        return merged_memory_obj, True

    # No similar memories found or error occurred
    return memory, False
