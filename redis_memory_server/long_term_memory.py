import logging
import time

import nanoid
from fastapi import BackgroundTasks
from redis.asyncio import Redis
from redis.commands.search.query import Query

from redis_memory_server.extraction import handle_extraction
from redis_memory_server.models import (
    LongTermMemory,
    LongTermMemoryResult,
    LongTermMemoryResults,
)
from redis_memory_server.utils import (
    REDIS_INDEX_NAME,
    Keys,
    TokenEscaper,
    get_openai_client,
)


logger = logging.getLogger(__name__)
escaper = TokenEscaper()


async def extract_memory_structure(
    redis: Redis, _id: str, text: str, namespace: str | None
):
    # Process messages for topic/entity extraction
    # TODO: Move into background task.
    topics, entities = await handle_extraction(text)

    # Convert lists to comma-separated strings for TAG fields
    topics_joined = ",".join(topics) if topics else ""
    entities_joined = ",".join(entities) if entities else ""

    await redis.hset(
        Keys.memory_key(_id, ""),
        mapping={
            "topics": topics_joined,
            "entities": entities_joined,
        },
    )  # type: ignore


async def index_long_term_memories(
    redis: Redis,
    memories: list[LongTermMemory],
    background_tasks: BackgroundTasks,
) -> None:
    """
    Index long-term memories in Redis for search
    """
    # Currently we only support OpenAI embeddings
    client = await get_openai_client()
    embeddings = await client.create_embedding([memory.text for memory in memories])

    async with redis.pipeline(transaction=False) as pipe:
        for idx, embedding in enumerate(embeddings):
            memory = memories[idx]
            id_ = memory.id_ if memory.id_ else nanoid.generate()
            key = Keys.memory_key(id_, memory.namespace)
            vector = embedding.tobytes()
            id_ = memory.id_ if memory.id_ else nanoid.generate()

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

            background_tasks.add_task(
                extract_memory_structure, redis, id_, memory.text, memory.namespace
            )

        await pipe.execute()

    logger.info(f"Indexed {len(memories)} memories")


class Unset:
    pass


async def search_long_term_memories(
    text: str,
    redis: Redis,
    session_id: str | None = None,
    user_id: str | None = None,
    namespace: str | None = None,
    created_at: int | None = None,
    last_accessed: int | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    distance_threshold: float | type[Unset] = Unset,
    limit: int = 10,
    offset: int = 0,
) -> LongTermMemoryResults:
    """Search for long-term memories using vector similarity and filters"""

    try:
        query = escaper.escape(text)
        if session_id:
            session_id = escaper.escape(session_id)
        if namespace:
            namespace = escaper.escape(namespace)

        client = await get_openai_client()  # Only OpenAI supports embeddings currently
        query_embedding = await client.create_embedding([query])
        vector = query_embedding.tobytes()

        # TODO: Use RedisVL
        params = {"vec": vector}
        namespace_filter = f"@namespace:{{{namespace}}}" if namespace else ""
        session_filter = f"@session_id:{{{session_id}}}" if session_id else ""
        topics_filter = f"@topics:{{{','.join(topics)}}}" if topics else ""
        entities_filter = f"@entities:{{{','.join(entities)}}}" if entities else ""
        user_id_filter = f"@user_id:{{{user_id}}}" if user_id else ""

        # TODO: time filters

        if distance_threshold and distance_threshold is not Unset:
            base_query = Query(
                f"{session_filter} {namespace_filter} {topics_filter} {entities_filter} {user_id_filter} @vector:[VECTOR_RANGE $radius $vec]=>{{$YIELD_DISTANCE_AS: dist}}"
            )
            params = {"vec": vector, "radius": distance_threshold}
        else:
            if not any(
                [
                    session_filter,
                    namespace_filter,
                    topics_filter,
                    entities_filter,
                    user_id_filter,
                ]
            ):
                pre_filter = "(*)"
            else:
                pre_filter = f"({session_filter} {namespace_filter} {topics_filter} {entities_filter} {user_id_filter})"
            base_query = Query(f"{pre_filter}=>[KNN {limit} @vector $vec AS dist]")

        q = (
            base_query.return_fields(
                "text",
                "id_",
                "user_id",
                "session_id",
                "namespace",
                "topics",
                "entities",
                "dist",
                "created_at",
                "last_accessed",
            )
            .sort_by("dist", asc=True)
            .paging(offset, limit)
            .dialect(2)
        )

        # Execute search
        raw_results = await redis.ft(REDIS_INDEX_NAME).search(
            q,
            query_params=params,  # type: ignore
        )

        # Parse results safely
        results = []
        total_results = 0

        # Check if raw_results has the expected attributes
        if hasattr(raw_results, "docs") and isinstance(raw_results.docs, list):
            for doc in raw_results.docs:
                if hasattr(doc, "id") and hasattr(doc, "text") and hasattr(doc, "dist"):
                    results.append(
                        LongTermMemoryResult(
                            id_=doc.id_,
                            text=doc.text,
                            dist=float(doc.dist),
                            created_at=int(doc.created_at),
                            last_accessed=int(doc.last_accessed),
                            user_id=doc.user_id,
                            session_id=doc.session_id,
                            namespace=doc.namespace,
                            topics=doc.topics.split(",") if doc.topics else [],
                            entities=doc.entities.split(",") if doc.entities else [],
                        )
                    )

            total_results = getattr(raw_results, "total", len(results))
        else:
            # Handle the case where raw_results doesn't have the expected structure
            logger.warning("Unexpected search result format")
            total_results = 0

        logger.info(f"Found {len(results)} results for query in session {session_id}")
        return LongTermMemoryResults(total=total_results, memories=results)
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise
