"""Working memory management for sessions."""

import json
import logging
import time
from datetime import UTC, datetime

from redis.asyncio import Redis

from agent_memory_server.models import MemoryMessage, MemoryRecord, WorkingMemory
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import get_redis_conn


logger = logging.getLogger(__name__)


def json_datetime_handler(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


async def list_sessions(
    redis,
    limit: int = 10,
    offset: int = 0,
    namespace: str | None = None,
    user_id: str | None = None,
) -> tuple[int, list[str]]:
    """
    List sessions

    Args:
        redis: Redis client
        limit: Maximum number of sessions to return
        offset: Offset for pagination
        namespace: Optional namespace filter
        user_id: Optional user ID filter (not yet implemented - sessions are stored in sorted sets)

    Returns:
        Tuple of (total_count, session_ids)

    Note:
        The user_id parameter is accepted for API compatibility but filtering by user_id
        is not yet implemented. This would require changing how sessions are stored to
        enable efficient user_id-based filtering.
    """
    # Calculate start and end indices (0-indexed start, inclusive end)
    start = offset
    end = offset + limit - 1

    # TODO: This should take a user_id
    sessions_key = Keys.sessions_key(namespace=namespace)

    async with redis.pipeline() as pipe:
        pipe.zcard(sessions_key)
        pipe.zrange(sessions_key, start, end)
        total, session_ids = await pipe.execute()

    return total, [
        s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids
    ]


async def get_working_memory(
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    redis_client: Redis | None = None,
) -> WorkingMemory | None:
    """
    Get working memory for a session.

    Args:
        session_id: The session ID
        namespace: Optional namespace for the session
        redis_client: Optional Redis client

    Returns:
        WorkingMemory object or None if not found
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    key = Keys.working_memory_key(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
    )

    try:
        data = await redis_client.get(key)
        if not data:
            logger.debug(
                f"No working memory found for parameters: {session_id}, {user_id}, {namespace}"
            )
            return None

        # Parse the JSON data
        working_memory_data = json.loads(data)

        # Convert memory records back to MemoryRecord objects
        memories = []
        for memory_data in working_memory_data.get("memories", []):
            memory = MemoryRecord(**memory_data)
            memories.append(memory)

        # Convert messages back to MemoryMessage objects
        messages = []
        for message_data in working_memory_data.get("messages", []):
            message = MemoryMessage(**message_data)
            messages.append(message)

        return WorkingMemory(
            messages=messages,
            memories=memories,
            context=working_memory_data.get("context"),
            user_id=working_memory_data.get("user_id"),
            tokens=working_memory_data.get("tokens", 0),
            session_id=session_id,
            namespace=namespace,
            ttl_seconds=working_memory_data.get("ttl_seconds", None),
            data=working_memory_data.get("data") or {},
            last_accessed=datetime.fromtimestamp(
                working_memory_data.get("last_accessed", int(time.time())), UTC
            ),
            created_at=datetime.fromtimestamp(
                working_memory_data.get("created_at", int(time.time())), UTC
            ),
            updated_at=datetime.fromtimestamp(
                working_memory_data.get("updated_at", int(time.time())), UTC
            ),
        )

    except Exception as e:
        logger.error(f"Error getting working memory for session {session_id}: {e}")
        return None


async def set_working_memory(
    working_memory: WorkingMemory,
    redis_client: Redis | None = None,
) -> None:
    """
    Set working memory for a session with TTL.

    Args:
        working_memory: WorkingMemory object to store
        redis_client: Optional Redis client
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # Validate that all memories have id (Stage 3 requirement)
    for memory in working_memory.memories:
        if not memory.id:
            raise ValueError("All memory records in working memory must have an ID")

    key = Keys.working_memory_key(
        session_id=working_memory.session_id,
        user_id=working_memory.user_id,
        namespace=working_memory.namespace,
    )

    # Update the updated_at timestamp
    working_memory.updated_at = datetime.now(UTC)

    # Convert to JSON-serializable format with timestamp conversion
    data = {
        "messages": [
            message.model_dump(mode="json") for message in working_memory.messages
        ],
        "memories": [
            memory.model_dump(mode="json") for memory in working_memory.memories
        ],
        "context": working_memory.context,
        "user_id": working_memory.user_id,
        "tokens": working_memory.tokens,
        "session_id": working_memory.session_id,
        "namespace": working_memory.namespace,
        "ttl_seconds": working_memory.ttl_seconds,
        "data": working_memory.data or {},
        "last_accessed": int(working_memory.last_accessed.timestamp()),
        "created_at": int(working_memory.created_at.timestamp()),
        "updated_at": int(working_memory.updated_at.timestamp()),
    }

    try:
        if working_memory.ttl_seconds is not None:
            # Store with TTL
            await redis_client.setex(
                key,
                working_memory.ttl_seconds,
                json.dumps(data, default=json_datetime_handler),
            )
            logger.info(
                f"Set working memory for session {working_memory.session_id} with TTL {working_memory.ttl_seconds}s"
            )
        else:
            await redis_client.set(
                key,
                json.dumps(data, default=json_datetime_handler),
            )
            logger.info(
                f"Set working memory for session {working_memory.session_id} with no TTL"
            )
    except Exception as e:
        logger.error(
            f"Error setting working memory for session {working_memory.session_id}: {e}"
        )
        raise


async def delete_working_memory(
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    redis_client: Redis | None = None,
) -> None:
    """
    Delete working memory for a session.

    Args:
        session_id: The session ID
        user_id: Optional user ID for the session
        namespace: Optional namespace for the session
        redis_client: Optional Redis client
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    key = Keys.working_memory_key(
        session_id=session_id, user_id=user_id, namespace=namespace
    )

    try:
        await redis_client.delete(key)
        logger.info(f"Deleted working memory for session {session_id}")

    except Exception as e:
        logger.error(f"Error deleting working memory for session {session_id}: {e}")
        raise
