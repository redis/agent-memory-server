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


async def get_working_memory(
    session_id: str,
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

    key = Keys.working_memory_key(session_id, namespace)

    try:
        data = await redis_client.get(key)
        if not data:
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
            ttl_seconds=working_memory_data.get("ttl_seconds", 3600),
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
            raise ValueError("All memory records in working memory must have an id")

    key = Keys.working_memory_key(working_memory.session_id, working_memory.namespace)

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
        # Store with TTL
        await redis_client.setex(
            key,
            working_memory.ttl_seconds,
            json.dumps(
                data, default=json_datetime_handler
            ),  # Add custom handler for any remaining datetime objects
        )
        logger.info(
            f"Set working memory for session {working_memory.session_id} with TTL {working_memory.ttl_seconds}s"
        )

    except Exception as e:
        logger.error(
            f"Error setting working memory for session {working_memory.session_id}: {e}"
        )
        raise


async def delete_working_memory(
    session_id: str,
    namespace: str | None = None,
    redis_client: Redis | None = None,
) -> None:
    """
    Delete working memory for a session.

    Args:
        session_id: The session ID
        namespace: Optional namespace for the session
        redis_client: Optional Redis client
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    key = Keys.working_memory_key(session_id, namespace)

    try:
        await redis_client.delete(key)
        logger.info(f"Deleted working memory for session {session_id}")

    except Exception as e:
        logger.error(f"Error deleting working memory for session {session_id}: {e}")
        raise
