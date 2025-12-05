"""Working memory management for sessions."""

import json
import logging
import time
from datetime import UTC, datetime

from redis.asyncio import Redis

from agent_memory_server.models import (
    MemoryMessage,
    MemoryRecord,
    MemoryStrategyConfig,
    WorkingMemory,
)
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import get_redis_conn


logger = logging.getLogger(__name__)

# Flag to track if all string keys have been migrated to JSON
# When True, we skip the type() check and go straight to json().get()
_string_keys_migrated: bool = False

# Counter for remaining string keys (avoids re-scanning after each migration)
_remaining_string_keys: int = 0


async def check_and_set_migration_status(redis_client: Redis | None = None) -> bool:
    """
    Check if any working memory keys are still in old string format.
    Sets the global _string_keys_migrated flag and _remaining_string_keys counter.

    Args:
        redis_client: Optional Redis client

    Returns:
        True if all keys are migrated (or no keys exist), False if string keys remain
    """
    global _string_keys_migrated, _remaining_string_keys

    if not redis_client:
        redis_client = await get_redis_conn()

    # Scan for working_memory:* keys
    cursor = 0
    json_keys_found = 0

    while True:
        cursor, keys = await redis_client.scan(
            cursor=cursor, match="working_memory:*", count=1000
        )

        if keys:
            # Use pipeline to batch TYPE calls for better performance
            pipe = redis_client.pipeline()
            for key in keys:
                pipe.type(key)
            types = await pipe.execute()

            for key_type in types:
                if isinstance(key_type, bytes):
                    key_type = key_type.decode("utf-8")

                if key_type == "string":
                    # Early exit: found at least one string key, enable lazy migration
                    logger.info(
                        "Found working memory key in old string format. "
                        "Lazy migration enabled. Run 'agent-memory migrate-working-memory' "
                        "to migrate all keys at once."
                    )
                    _string_keys_migrated = False
                    # We don't know the exact count, so set to -1 to indicate unknown
                    # The counter will be managed differently in this mode
                    _remaining_string_keys = -1
                    return False
                elif key_type == "ReJSON-RL":
                    json_keys_found += 1

        if cursor == 0:
            break

    # No string keys found
    if json_keys_found > 0:
        logger.info(
            f"All {json_keys_found} working memory keys are in JSON format. "
            "Skipping type checks."
        )
    else:
        logger.info("No working memory keys found. Skipping type checks.")
    _string_keys_migrated = True
    _remaining_string_keys = 0
    return True


def _decrement_string_key_count() -> None:
    """
    Decrement the string key counter after a successful migration.

    Note: When _remaining_string_keys is -1, we don't know the exact count
    (early exit mode). In this case, lazy migration stays enabled until
    the migration script is run.
    """
    global _string_keys_migrated, _remaining_string_keys

    # If we don't know the count (-1), we can't track completion
    # The migration script will set the flag when done
    if _remaining_string_keys == -1:
        return

    _remaining_string_keys -= 1
    if _remaining_string_keys <= 0:
        _remaining_string_keys = 0
        _string_keys_migrated = True
        logger.info("All working memory keys have been migrated to JSON format.")


def is_migration_complete() -> bool:
    """Check if migration is complete (for testing purposes)."""
    return _string_keys_migrated


def get_remaining_string_keys() -> int:
    """Get the count of remaining string keys (for testing purposes)."""
    return _remaining_string_keys


def reset_migration_status() -> None:
    """Reset migration status (for testing purposes)."""
    global _string_keys_migrated, _remaining_string_keys
    _string_keys_migrated = False
    _remaining_string_keys = 0


def set_migration_complete() -> None:
    """Mark migration as complete (called by migration script)."""
    global _string_keys_migrated, _remaining_string_keys
    _string_keys_migrated = True
    _remaining_string_keys = 0
    logger.info("Working memory migration marked as complete.")


async def _migrate_string_to_json(
    redis_client: Redis,
    key: str,
    string_data: str,
) -> dict:
    """
    Migrate working memory from old string format to new JSON format.

    Args:
        redis_client: Redis client
        key: The Redis key
        string_data: The JSON string data from the old format

    Returns:
        The parsed dict data
    """
    try:
        data = json.loads(string_data)
        logger.info(f"Migrating working memory key {key} from string to JSON format")

        # Delete the old string key and set as JSON
        await redis_client.delete(key)
        await redis_client.json().set(key, "$", data)

        # Preserve TTL if it was set
        # Note: TTL is lost during migration since we deleted the key
        # The next set_working_memory call will restore it if configured

        logger.info(f"Successfully migrated working memory key {key} to JSON format")

        # Decrement the counter (O(1) instead of re-scanning all keys)
        _decrement_string_key_count()

        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse string data for key {key}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to migrate working memory key {key}: {e}")
        raise


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
    recent_messages_limit: int | None = None,
) -> WorkingMemory | None:
    """
    Get working memory for a session.

    If no working memory exists but index_all_messages_in_long_term_memory is enabled,
    attempts to reconstruct working memory from messages stored in long-term memory.

    Args:
        session_id: The session ID
        namespace: Optional namespace for the session
        redis_client: Optional Redis client
        recent_messages_limit: Optional limit on number of recent messages to return

    Returns:
        WorkingMemory object or None if not found
    """
    from agent_memory_server.config import settings

    if not redis_client:
        redis_client = await get_redis_conn()

    key = Keys.working_memory_key(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
    )

    try:
        working_memory_data = None

        if _string_keys_migrated:
            # Fast path: all keys are already in JSON format
            working_memory_data = await redis_client.json().get(key)
        else:
            # Slow path: check key type to determine storage format
            key_type = await redis_client.type(key)
            if isinstance(key_type, bytes):
                key_type = key_type.decode("utf-8")

            if key_type == "ReJSON-RL":
                # New JSON format
                working_memory_data = await redis_client.json().get(key)
            elif key_type == "string":
                # Old string format - migrate to JSON
                string_data = await redis_client.get(key)
                if string_data:
                    if isinstance(string_data, bytes):
                        string_data = string_data.decode("utf-8")
                    working_memory_data = await _migrate_string_to_json(
                        redis_client, key, string_data
                    )
            # If key_type is "none", the key doesn't exist - working_memory_data stays None

        if not working_memory_data:
            logger.debug(
                f"No working memory found for parameters: {session_id}, {user_id}, {namespace}"
            )

            # Try to reconstruct from long-term memory if enabled
            if settings.index_all_messages_in_long_term_memory:
                reconstructed = await _reconstruct_working_memory_from_long_term(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    recent_messages_limit=recent_messages_limit,
                )
                if reconstructed:
                    logger.info(
                        f"Reconstructed working memory for session {session_id} from long-term storage"
                    )
                    return reconstructed

            return None

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

        # Apply recent messages limit if specified (in-memory slice)
        if recent_messages_limit is not None and recent_messages_limit > 0:
            # Sort messages by created_at timestamp to ensure proper chronological order
            messages.sort(key=lambda m: m.created_at)
            # Get the most recent N messages
            messages = messages[-recent_messages_limit:]

        # Handle memory strategy configuration
        strategy_data = working_memory_data.get("long_term_memory_strategy")
        if strategy_data:
            long_term_memory_strategy = MemoryStrategyConfig(**strategy_data)
        else:
            long_term_memory_strategy = (
                MemoryStrategyConfig()
            )  # Default to discrete strategy

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
            long_term_memory_strategy=long_term_memory_strategy,
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
        "long_term_memory_strategy": working_memory.long_term_memory_strategy.model_dump(),
        "last_accessed": int(working_memory.last_accessed.timestamp()),
        "created_at": int(working_memory.created_at.timestamp()),
        "updated_at": int(working_memory.updated_at.timestamp()),
    }

    try:
        # Use Redis native JSON storage
        await redis_client.json().set(key, "$", data)

        if working_memory.ttl_seconds is not None:
            # Set TTL separately for JSON keys
            await redis_client.expire(key, working_memory.ttl_seconds)
            logger.info(
                f"Set working memory for session {working_memory.session_id} with TTL {working_memory.ttl_seconds}s"
            )
        else:
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


async def _reconstruct_working_memory_from_long_term(
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    recent_messages_limit: int | None = None,
) -> WorkingMemory | None:
    """
    Reconstruct working memory from messages stored in long-term memory.

    This function searches for messages in long-term memory that belong to the
    specified session and reconstructs a WorkingMemory object from them.

    Args:
        session_id: The session ID to reconstruct
        user_id: Optional user ID filter
        namespace: Optional namespace filter
        recent_messages_limit: Optional limit on number of recent messages to return

    Returns:
        Reconstructed WorkingMemory object or None if no messages found
    """
    from agent_memory_server.filters import MemoryType, Namespace, SessionId, UserId
    from agent_memory_server.long_term_memory import search_long_term_memories

    try:
        # Search for message-type memories for this session
        session_filter = SessionId(eq=session_id)
        user_filter = UserId(eq=user_id) if user_id else None
        namespace_filter = Namespace(eq=namespace) if namespace else None
        memory_type_filter = MemoryType(eq="message")

        # Search for messages with appropriate limit
        # We use empty text since we're filtering by session_id and memory_type
        search_limit = recent_messages_limit if recent_messages_limit else 1000
        results = await search_long_term_memories(
            text="",  # Empty query since we're filtering by metadata
            session_id=session_filter,
            user_id=user_filter,
            namespace=namespace_filter,
            memory_type=memory_type_filter,
            limit=search_limit,
            offset=0,
        )

        if not results.memories:
            logger.debug(
                f"No message memories found for session {session_id} in long-term storage"
            )
            return None

        # Convert memory records back to messages
        messages = []
        for memory in results.memories:
            # Parse the message text which should be in format "role: content"
            text = memory.text
            if ": " in text:
                role, content = text.split(": ", 1)
                message = MemoryMessage(
                    id=memory.id,
                    role=role.lower(),
                    content=content,
                    created_at=memory.created_at,  # Use the original creation time
                    persisted_at=memory.persisted_at,  # Mark as already persisted
                )
                messages.append(message)
            else:
                logger.warning(
                    f"Skipping malformed message memory: {memory.id} - {text}"
                )

        if not messages:
            logger.debug(f"No valid messages found for session {session_id}")
            return None

        # Sort messages by creation time to maintain conversation order (most recent first for API response)
        messages.sort(key=lambda m: m.created_at, reverse=True)

        # If we have a limit, take only the most recent N messages
        if recent_messages_limit and len(messages) > recent_messages_limit:
            messages = messages[:recent_messages_limit]

        # Reverse back to chronological order for working memory (oldest first)
        messages.reverse()

        # Create reconstructed working memory
        now = datetime.now(UTC)
        reconstructed = WorkingMemory(
            session_id=session_id,
            namespace=namespace,
            user_id=user_id,
            messages=messages,
            memories=[],  # No structured memories in reconstruction
            context="",  # No context in reconstruction
            data={},  # No session data in reconstruction
            created_at=messages[0].persisted_at or now if messages else now,
            updated_at=now,
            last_accessed=now,
        )

        logger.info(
            f"Reconstructed working memory for session {session_id} with {len(messages)} messages"
        )
        return reconstructed

    except Exception as e:
        logger.error(
            f"Error reconstructing working memory for session {session_id}: {e}"
        )
        return None
