import hashlib
import logging
import time
from functools import reduce

import nanoid
from redis.asyncio import Redis
from redisvl.query import VectorQuery, VectorRangeQuery
from redisvl.utils.vectorize import OpenAITextVectorizer

from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.extraction import handle_extraction
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.llms import AnthropicClientWrapper, OpenAIClientWrapper
from agent_memory_server.models import (
    LongTermMemory,
    LongTermMemoryResult,
    LongTermMemoryResults,
)
from agent_memory_server.utils import (
    Keys,
    get_model_client,
    get_redis_conn,
    get_search_index,
    safe_get,
)


logger = logging.getLogger(__name__)


async def extract_memory_structure(_id: str, text: str, namespace: str | None):
    redis = get_redis_conn()

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


async def merge_memories_with_llm(
    memories: list[dict], memory_type: str = "hash"
) -> dict:
    """
    Use an LLM to merge similar or duplicate memories.

    Args:
        memories: List of memory dictionaries to merge
        memory_type: Type of duplication ("hash" for exact matches or "semantic" for similar)

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
    if memory_type == "hash":
        instruction = "Merge these duplicate memories into a single, concise memory:"
    else:
        instruction = "Merge these similar memories into a single, coherent memory:"

    prompt = f"{instruction}\n\n"
    for i, text in enumerate(memory_texts, 1):
        prompt += f"Memory {i}: {text}\n\n"

    prompt += "\nMerged memory:"

    # Use gpt-4o-mini for a good balance of quality and speed
    model_name = "gpt-4o-mini"
    model_client: OpenAIClientWrapper | AnthropicClientWrapper = await get_model_client(
        model_name
    )

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
    created_at = min(m.get("created_at", int(time.time())) for m in memories)

    # Use the most recent last_accessed timestamp
    last_accessed = max(m.get("last_accessed", int(time.time())) for m in memories)

    # Prefer non-empty namespace, user_id, session_id from memories
    namespace = next((m["namespace"] for m in memories if m.get("namespace")), None)
    user_id = next((m["user_id"] for m in memories if m.get("user_id")), None)
    session_id = next((m["session_id"] for m in memories if m.get("session_id")), None)

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
    }

    # Generate a new hash for the merged memory
    merged_memory["memory_hash"] = generate_memory_hash(merged_memory)

    return merged_memory


async def compact_long_term_memories(redis: Redis) -> None:
    """
    Compact long-term memories in Redis

    1. Use a stable hash to identify exact duplicates (same text, user ID, session ID)
    2. Merge exact duplicates using an LLM
    3. Find semantically similar memories using vector search
    4. Merge semantically similar memories using an LLM
    """
    logger.info("Starting memory compaction process")

    # 1. Scan Redis for all memory keys
    memory_keys = []
    cursor = 0
    pattern = "memory:*"

    while True:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
        memory_keys.extend(keys)
        if cursor == 0:
            break

    logger.info(f"Found {len(memory_keys)} memory keys")

    if not memory_keys:
        logger.info("No memories found to compact")
        return

    # 2. Get all memory data and group by hash
    memories_by_hash = {}
    processed_keys = set()
    vectorizer = OpenAITextVectorizer()
    search_index = get_search_index(redis)

    # Fetch all memories in batches to avoid overwhelming Redis
    batch_size = 50
    memory_data = []

    for i in range(0, len(memory_keys), batch_size):
        batch_keys = memory_keys[i : i + batch_size]
        pipeline = redis.pipeline()

        for key in batch_keys:
            pipeline.hgetall(key)

        results = await pipeline.execute()

        for j, result in enumerate(results):
            if not result:
                continue

            # Convert bytes to strings if needed
            memory = {
                k.decode() if isinstance(k, bytes) else k: v.decode()
                if isinstance(v, bytes)
                else v
                for k, v in result.items()
            }

            memory["key"] = batch_keys[j]

            # Generate hash for the memory
            memory["memory_hash"] = generate_memory_hash(memory)

            memory_data.append(memory)

            # Add to hash-based groups
            if memory["memory_hash"] not in memories_by_hash:
                memories_by_hash[memory["memory_hash"]] = []

            memories_by_hash[memory["memory_hash"]].append(memory)

    logger.info(
        f"Processed {len(memory_data)} memories with {len(memories_by_hash)} unique hashes"
    )

    # 3. First compact memories with the same hash
    compacted_memories = []
    keys_to_delete = set()

    for memory_hash, memories in memories_by_hash.items():
        if len(memories) > 1:
            # We found duplicates with the same hash
            logger.info(
                f"Found {len(memories)} duplicate memories with hash {memory_hash[:8]}..."
            )

            # Merge duplicates
            merged_memory = await merge_memories_with_llm(memories, "hash")
            compacted_memories.append(merged_memory)

            # Mark original keys for deletion
            for memory in memories:
                keys_to_delete.add(memory["key"])
                processed_keys.add(memory["key"])
        else:
            # Single memory, keep for semantic deduplication
            compacted_memories.append(memories[0])

    logger.info(f"After hash-based compaction: {len(compacted_memories)} memories")

    # 4. Now handle semantic similarity compaction using RedisVL
    if not compacted_memories:
        logger.info("No memories after hash-based compaction")
        return

    # Using a distance threshold for semantic similarity
    semantic_distance_threshold = 0.1  # Adjust based on testing
    semantic_groups = []
    remaining_memories = list(compacted_memories)

    # We'll process each memory, find similar ones using RedisVL, and group them
    while remaining_memories:
        # Take the first remaining memory as reference
        ref_memory = remaining_memories.pop(0)

        # Skip if this memory has already been processed
        if ref_memory["key"] in processed_keys:
            continue

        # Find semantically similar memories using VectorRangeQuery
        query_text = ref_memory["text"]
        query_vector = await vectorizer.aembed(query_text)

        # Create filter for user_id and session_id matching
        filters = []
        if ref_memory.get("user_id"):
            filters.append(UserId(eq=ref_memory["user_id"]).to_filter())
        if ref_memory.get("session_id"):
            filters.append(SessionId(eq=ref_memory["session_id"]).to_filter())

        filter_expression = reduce(lambda x, y: x & y, filters) if filters else None

        # Create vector query with distance threshold
        q = VectorRangeQuery(
            vector=query_vector,
            vector_field_name="vector",
            distance_threshold=semantic_distance_threshold,
            num_results=100,  # Set a reasonable limit
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
            ],
        )

        if filter_expression:
            q.set_filter(filter_expression)

        # Execute the query
        search_result = await search_index.search(q, q.params)

        # If we found similar memories
        if (
            search_result.docs and len(search_result.docs) > 1
        ):  # More than just the reference memory
            similar_memories = [ref_memory]  # Start with reference memory

            # Process similar memories from search results
            for doc in search_result.docs:
                # Skip if it's the reference memory or already processed
                doc_id = safe_get(doc, "id_")
                if not doc_id or any(m.get("id_") == doc_id for m in similar_memories):
                    continue

                # Find the original memory data
                for memory in remaining_memories[:]:
                    if memory.get("id_") == doc_id:
                        similar_memories.append(memory)
                        remaining_memories.remove(memory)
                        # Mark as processed
                        if memory["key"] not in processed_keys:
                            keys_to_delete.add(memory["key"])
                            processed_keys.add(memory["key"])
                        break

            # If we found similar memories
            if len(similar_memories) > 1:
                semantic_groups.append(similar_memories)
                # Mark reference memory as processed if not already
                if ref_memory["key"] not in processed_keys:
                    keys_to_delete.add(ref_memory["key"])
                    processed_keys.add(ref_memory["key"])

        # If no similar memories found, the reference memory remains unprocessed
        # and will be preserved in the database

    logger.info(f"Found {len(semantic_groups)} semantic similarity groups")

    # 5. Merge semantic groups and store the final compacted memories
    final_compacted_memories = []

    # Process each semantic group
    for group in semantic_groups:
        if len(group) > 1:
            merged_memory = await merge_memories_with_llm(group, "semantic")
            final_compacted_memories.append(merged_memory)
        else:
            # Should never happen as we only create groups with more than one memory
            final_compacted_memories.append(group[0])

    # 6. Index the new compacted memories
    memories_to_index = [
        LongTermMemory(
            text=memory["text"],
            id_=memory.get("id_"),
            session_id=memory.get("session_id"),
            user_id=memory.get("user_id"),
            namespace=memory.get("namespace"),
            created_at=memory.get("created_at"),
            last_accessed=memory.get("last_accessed"),
            topics=memory.get("topics"),
            entities=memory.get("entities"),
        )
        for memory in final_compacted_memories
    ]

    if memories_to_index:
        await index_long_term_memories(memories_to_index)
        logger.info(f"Indexed {len(memories_to_index)} compacted memories")

    # 7. Delete the original memory keys
    if keys_to_delete:
        # Delete in batches to be efficient
        for i in range(0, len(keys_to_delete), batch_size):
            batch_keys = list(keys_to_delete)[i : i + batch_size]
            await redis.delete(*batch_keys)

        logger.info(f"Deleted {len(keys_to_delete)} original memory keys")

    logger.info("Memory compaction completed successfully")


async def index_long_term_memories(
    memories: list[LongTermMemory],
) -> None:
    """
    Index long-term memories in Redis for search
    """
    redis = get_redis_conn()
    background_tasks = get_background_tasks()
    vectorizer = OpenAITextVectorizer()
    embeddings = await vectorizer.aembed_many(
        [memory.text for memory in memories],
        batch_size=20,
        as_buffer=True,
    )

    async with redis.pipeline(transaction=False) as pipe:
        for idx, vector in enumerate(embeddings):
            memory = memories[idx]
            id_ = memory.id_ if memory.id_ else nanoid.generate()
            key = Keys.memory_key(id_, memory.namespace)

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
                    "vector": vector,
                },
            )

            await background_tasks.add_task(
                extract_memory_structure, id_, memory.text, memory.namespace
            )

        await pipe.execute()

    logger.info(f"Indexed {len(memories)} memories")


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
            ],
        )
    if filter_expression:
        q.set_filter(filter_expression)

    q.paging(offset=offset, num=limit)

    index = get_search_index(redis)
    search_result = await index.search(q, q.params)

    results = []

    for doc in search_result.docs:
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
            )
        )
    total_results = search_result.total

    logger.info(f"Found {len(results)} results for query")
    return LongTermMemoryResults(
        total=total_results,
        memories=results,
        next_offset=offset + limit if offset + limit < total_results else None,
    )
