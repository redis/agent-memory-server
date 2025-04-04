import json
import logging
import time

import nanoid
from fastapi import BackgroundTasks
from redis.asyncio import Redis
from redis.commands.search.query import Query

from redis_memory_server.config import settings
from redis_memory_server.extraction import handle_extraction
from redis_memory_server.llms import OpenAIClientWrapper
from redis_memory_server.models import (
    MemoryMessage,
    RedisearchResult,
    SearchResults,
    SessionMemory,
)
from redis_memory_server.summarization import summarize_session
from redis_memory_server.utils import (
    REDIS_INDEX_NAME,
    Keys,
    TokenEscaper,
    get_openai_client,
)


logger = logging.getLogger(__name__)
escaper = TokenEscaper()


async def list_sessions(
    redis: Redis,
    page: int = 1,
    size: int = 20,
    namespace: str | None = None,
) -> list[str]:
    """List sessions"""
    # Calculate start and end indices (0-indexed start, inclusive end)
    start = (page - 1) * size
    end = page * size - 1

    sessions_key = Keys.sessions_key(namespace=namespace)
    session_ids = await redis.zrange(sessions_key, start, end)
    return [s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids]


async def get_session_memory(
    redis: Redis,
    session_id: str,
    window_size: int = settings.window_size,
    namespace: str | None = None,
) -> SessionMemory | None:
    """Get a session's memory"""
    sessions_key = Keys.sessions_key(namespace=namespace)
    messages_key = Keys.messages_key(session_id, namespace=namespace)
    context_key = Keys.context_key(session_id, namespace=namespace)
    metadata_key = Keys.metadata_key(session_id, namespace=namespace)

    session_exists = await redis.zscore(sessions_key, session_id)
    if not session_exists:
        return None

    pipe = redis.pipeline()
    pipe.lrange(messages_key, 0, window_size - 1)  # Get messages
    pipe.get(context_key)  # Get context
    pipe.hgetall(metadata_key)  # Get metadata

    messages_raw, context_raw, metadata_raw = await pipe.execute()

    memory_messages = []
    for msg_raw in messages_raw:
        # Decode if needed
        if isinstance(msg_raw, bytes):
            msg_raw = msg_raw.decode("utf-8")

        # Parse JSON
        msg_dict = json.loads(msg_raw)

        # Convert comma-separated strings back to lists for topics and entities
        if "topics" in msg_dict:
            msg_dict["topics"] = (
                msg_dict["topics"].split(",") if msg_dict["topics"] else []
            )
        if "entities" in msg_dict:
            msg_dict["entities"] = (
                msg_dict["entities"].split(",") if msg_dict["entities"] else []
            )

        memory_messages.append(MemoryMessage(**msg_dict))

    kwargs = {
        "messages": memory_messages,
        "context": context_raw.decode("utf-8") if context_raw else None,
        "namespace": namespace,
        **{k.decode("utf-8"): v.decode("utf-8") for k, v in metadata_raw.items()},
    }

    return SessionMemory(**kwargs)


async def set_session_memory(
    redis: Redis,
    session_id: str,
    memory: SessionMemory,
    background_tasks: BackgroundTasks,
):
    """Create or update a session's memory"""
    messages_key = Keys.messages_key(session_id, namespace=memory.namespace)
    context_key = Keys.context_key(session_id, namespace=memory.namespace)
    sessions_key = Keys.sessions_key(namespace=memory.namespace)

    messages_json = [json.dumps(msg.model_dump()) for msg in memory.messages]

    current_time = int(time.time())
    await redis.zadd(sessions_key, {session_id: current_time})
    if memory.context:
        await redis.set(context_key, memory.context)  # type: ignore
    await redis.rpush(messages_key, *messages_json)  # type: ignore

    # Check if window size is exceeded
    current_size = await redis.llen(messages_key)  # type: ignore
    if current_size > settings.window_size:
        # Handle summarization in background
        background_tasks.add_task(
            summarize_session,
            redis,
            session_id,
            settings.generation_model,
            settings.window_size,
        )

    # If long-term memory is enabled, index messages
    # TODO: Use a distributed background task
    if settings.long_term_memory:
        background_tasks.add_task(
            index_messages,
            redis,
            session_id,
            memory.messages,
            memory.namespace,
        )


async def delete_session_memory(
    redis: Redis,
    session_id: str,
    namespace: str | None = None,
):
    """Delete a session's memory"""
    # Define keys
    messages_key = Keys.messages_key(session_id)
    context_key = Keys.context_key(session_id)
    sessions_key = f"sessions:{namespace}" if namespace else "sessions"

    # Create pipeline for deletion
    pipe = redis.pipeline()
    pipe.delete(messages_key, context_key)
    pipe.zrem(sessions_key, session_id)
    await pipe.execute()


async def index_messages(
    redis: Redis,
    session_id: str,
    messages: list[MemoryMessage],
    namespace: str | None = None,
) -> None:
    """Index messages in Redis for vector search"""
    # Currently we only support OpenAI embeddings
    client = await get_openai_client()

    contents = [msg.content for msg in messages]
    embeddings = await client.create_embedding(contents)

    async with redis.pipeline(transaction=False) as pipe:
        for idx, embedding in enumerate(embeddings):
            id = nanoid.generate()
            key = Keys.memory_key(id, namespace)
            vector = embedding.tobytes()

            # Process messages for topic/entity extraction
            topics, entities = await handle_extraction(contents[idx])
            # Convert lists to comma-separated strings for TAG fields
            topics_joined = ",".join(topics) if topics else ""
            entities_joined = ",".join(entities) if entities else ""

            await pipe.hset(  # type: ignore
                key,
                mapping={
                    "session": session_id or "",
                    "namespace": namespace or "",
                    "vector": vector,
                    "content": contents[idx],
                    "role": messages[idx].role,
                    "topics": topics_joined,
                    "entities": entities_joined,
                },
            )

        await pipe.execute()

    logger.info(f"Indexed {len(messages)} messages for session {session_id}")


class Unset:
    pass


async def search_messages(
    text: str,
    client: OpenAIClientWrapper,  # Only OpenAI supports embeddings currently
    redis_conn: Redis,
    session_id: str | None = None,
    namespace: str | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    distance_threshold: float | type[Unset] = Unset,
    limit: int = 10,
    offset: int = 0,
) -> SearchResults:
    """Search for messages using vector similarity and filters"""
    try:
        query = escaper.escape(text)
        if session_id:
            session_id = escaper.escape(session_id)
        if namespace:
            namespace = escaper.escape(namespace)

        # Get embedding for query
        query_embedding = await client.create_embedding([query])
        vector = query_embedding.tobytes()

        # TODO: Use RedisVL
        params = {"vec": vector}
        namespace_filter = f"@namespace:{{{namespace}}}" if namespace else ""
        session_filter = f"@session:{{{session_id}}}" if session_id else ""
        topics_filter = f"@topics:{{{','.join(topics)}}}" if topics else ""
        entities_filter = f"@entities:{{{','.join(entities)}}}" if entities else ""

        if distance_threshold and distance_threshold is not Unset:
            base_query = Query(
                f"{session_filter} {namespace_filter} {topics_filter} {entities_filter} @vector:[VECTOR_RANGE $radius $vec]=>{{$YIELD_DISTANCE_AS: dist}}"
            )
            params = {"vec": vector, "radius": distance_threshold}
        else:
            base_query = Query(
                f"{session_filter} {namespace_filter} {topics_filter} {entities_filter}=>[KNN {limit} @vector$vec AS dist]"
            )

        q = (
            base_query.return_fields("role", "content", "dist")
            .sort_by("dist", asc=True)
            .paging(offset, limit)
            .dialect(2)
        )

        # Execute search
        raw_results = await redis_conn.ft(REDIS_INDEX_NAME).search(
            q,
            query_params=params,  # type: ignore
        )

        # Parse results safely
        results = []
        total_results = 0

        # Check if raw_results has the expected attributes
        if hasattr(raw_results, "docs") and isinstance(raw_results.docs, list):
            for doc in raw_results.docs:
                if (
                    hasattr(doc, "role")
                    and hasattr(doc, "content")
                    and hasattr(doc, "dist")
                ):
                    results.append(
                        RedisearchResult(
                            role=doc.role, content=doc.content, dist=float(doc.dist)
                        )
                    )

            total_results = getattr(raw_results, "total", len(results))
        else:
            # Handle the case where raw_results doesn't have the expected structure
            logger.warning("Unexpected search result format")
            total_results = 0

        logger.info(f"Found {len(results)} results for query in session {session_id}")
        return SearchResults(total=total_results, docs=results)
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise
