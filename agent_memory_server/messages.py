"""Session memory management functions."""

import json
import logging
import time

from redis import WatchError
from redis.asyncio import Redis

from agent_memory_server.config import settings
from agent_memory_server.dependencies import DocketBackgroundTasks
from agent_memory_server.long_term_memory import index_long_term_memories
from agent_memory_server.models import (
    LongTermMemory,
    MemoryMessage,
    SessionMemory,
)
from agent_memory_server.summarization import summarize_session
from agent_memory_server.utils.keys import Keys


logger = logging.getLogger(__name__)


async def list_sessions(
    redis: Redis,
    limit: int = 10,
    offset: int = 0,
    namespace: str | None = None,
) -> tuple[int, list[str]]:
    """List sessions"""
    # Calculate start and end indices (0-indexed start, inclusive end)
    start = offset
    end = offset + limit - 1

    sessions_key = Keys.sessions_key(namespace=namespace)

    # Check if the sessions key exists
    await redis.exists(sessions_key)

    # Try to get all sessions directly
    await redis.zrange(sessions_key, 0, -1)

    async with redis.pipeline() as pipe:
        pipe.zcard(sessions_key)
        pipe.zrange(sessions_key, start, end)
        total, session_ids = await pipe.execute()

    return total, [
        s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids
    ]


async def get_session_memory(
    redis: Redis,
    session_id: str,
    window_size: int = settings.window_size,
    namespace: str | None = None,
) -> SessionMemory | None:
    """Get a session's memory"""
    sessions_key = Keys.sessions_key(namespace=namespace)
    messages_key = Keys.messages_key(session_id, namespace=namespace)
    metadata_key = Keys.metadata_key(session_id, namespace=namespace)

    session_exists = await redis.zscore(sessions_key, session_id)
    if not session_exists:
        return None

    async with redis.pipeline() as pipe:
        pipe.lrange(messages_key, -window_size, -1)  # Get the most recent messages
        pipe.hgetall(metadata_key)
        messages_data, metadata = await pipe.execute()

    messages = []
    for msg_data in messages_data:
        if isinstance(msg_data, bytes):
            msg_data = msg_data.decode("utf-8")
        msg = json.loads(msg_data)
        messages.append(MemoryMessage(**msg))

    metadata_dict = {}
    for k, v in metadata.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else k
        value = v.decode("utf-8") if isinstance(v, bytes) else v
        metadata_dict[key] = value

    return SessionMemory(messages=messages, **metadata_dict)


async def set_session_memory(
    redis: Redis,
    session_id: str,
    memory: SessionMemory,
    background_tasks: DocketBackgroundTasks,
):
    """
    Create or update a session's memory

    Args:
        redis: The Redis client
        session_id: The session ID
        memory: The session memory to set
        background_tasks: Background tasks instance
    """
    sessions_key = Keys.sessions_key(namespace=memory.namespace)
    messages_key = Keys.messages_key(session_id, namespace=memory.namespace)
    metadata_key = Keys.metadata_key(session_id, namespace=memory.namespace)
    messages_json = [json.dumps(msg.model_dump()) for msg in memory.messages]
    metadata = memory.model_dump(
        exclude_none=True,
        exclude_unset=True,
        exclude={"messages"},
    )

    async with redis.pipeline(transaction=True) as pipe:
        await pipe.watch(messages_key, metadata_key)
        pipe.multi()

        while True:
            try:
                current_time = int(time.time())
                pipe.zadd(sessions_key, {session_id: current_time})
                pipe.rpush(messages_key, *messages_json)  # type: ignore
                pipe.hset(metadata_key, mapping=metadata)  # type: ignore
                await pipe.execute()
            except WatchError:
                continue
            break

    # Verify that the session was added to the sessions set
    await redis.zscore(sessions_key, session_id)

    # List all sessions in the sessions set
    await redis.zrange(sessions_key, 0, -1)

    # Check if window size is exceeded
    current_size = await redis.llen(messages_key)  # type: ignore
    if current_size > settings.window_size:
        # Add summarization task
        await background_tasks.add_task(
            summarize_session,
            session_id,
            settings.generation_model,
            settings.window_size,
        )

    # If long-term memory is enabled, index messages
    if settings.long_term_memory:
        memories = [
            LongTermMemory(
                session_id=session_id,
                text=f"{msg.role}: {msg.content}",
                namespace=memory.namespace,
                memory_type="message",
            )
            for msg in memory.messages
        ]

        await background_tasks.add_task(
            index_long_term_memories,
            memories,
        )


async def delete_session_memory(
    redis: Redis,
    session_id: str,
    namespace: str | None = None,
):
    """Delete a session's memory"""
    # Define keys
    messages_key = Keys.messages_key(session_id, namespace=namespace)
    sessions_key = Keys.sessions_key(namespace=namespace)
    metadata_key = Keys.metadata_key(session_id, namespace=namespace)

    # Create pipeline for deletion
    pipe = redis.pipeline()
    pipe.delete(messages_key, metadata_key)
    pipe.zrem(sessions_key, session_id)
    await pipe.execute()
