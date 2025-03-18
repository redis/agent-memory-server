import json
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from redis_memory_server.config import settings
from redis_memory_server.extraction import handle_extraction
from redis_memory_server.logging import get_logger
from redis_memory_server.long_term_memory import index_messages
from redis_memory_server.models import (
    AckResponse,
    GetSessionsQuery,
    MemoryMessage,
    MemoryMessagesAndContext,
    MemoryResponse,
)
from redis_memory_server.summarization import handle_compaction
from redis_memory_server.utils import (
    Keys,
    get_model_client,
    get_openai_client,
    get_redis_conn,
)


logger = get_logger(__name__)

router = APIRouter()


@router.get("/sessions/", response_model=list[str])
async def get_sessions(
    pagination: GetSessionsQuery = Depends(),
):
    """
    Get a list of session IDs, with optional pagination

    Args:
        pagination: Pagination parameters (page, size, namespace)

    Returns:
        List of session IDs
    """
    # Check page limit
    if pagination.page > 100:
        raise HTTPException(status_code=400, detail="Page must not exceed 100")

    redis = get_redis_conn()

    # Calculate start and end indices (0-indexed start, inclusive end)
    start = (pagination.page - 1) * pagination.size
    end = pagination.page * pagination.size - 1

    # Set key based on namespace
    sessions_key = (
        f"sessions:{pagination.namespace}" if pagination.namespace else "sessions"
    )

    try:
        # Get session IDs from Redis
        session_ids = await redis.zrange(sessions_key, start, end)

        # Convert from bytes to strings if needed
        return [s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids]

    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/sessions/{session_id}/memory", response_model=MemoryResponse)
async def get_memory(session_id: str):
    """
    Get memory for a session

    Args:
        session_id: The session ID

    Returns:
        Memory response with messages and context
    """
    redis = get_redis_conn()

    try:
        # Define keys
        messages_key = Keys.messages_key(session_id)
        context_key = Keys.context_key(session_id)
        token_count_key = Keys.token_count_key(session_id)

        # Get data from Redis in a pipeline
        pipe = redis.pipeline()
        pipe.lrange(messages_key, 0, settings.window_size - 1)  # Get messages
        pipe.mget(context_key, token_count_key)  # Get context and token count
        results = await pipe.execute()

        # Extract results
        messages_raw = results[0]
        context_and_tokens = results[1]

        # Parse messages
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

        # Extract context and tokens
        context = None
        tokens = None

        if context_and_tokens[0]:
            context_bytes = context_and_tokens[0]
            context = (
                context_bytes.decode("utf-8")
                if isinstance(context_bytes, bytes)
                else context_bytes
            )

        if context_and_tokens[1]:
            tokens_bytes = context_and_tokens[1]
            tokens_str = (
                tokens_bytes.decode("utf-8")
                if isinstance(tokens_bytes, bytes)
                else tokens_bytes
            )
            tokens = int(tokens_str)

        # Build response
        return MemoryResponse(
            messages=memory_messages,
            context=context,
            tokens=tokens,
        )

    except Exception as e:
        logger.error(f"Error getting memory for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/sessions/{session_id}/memory", response_model=AckResponse)
async def post_memory(
    session_id: str,
    memory_messages: MemoryMessagesAndContext,
    background_tasks: BackgroundTasks,
    namespace: str | None = None,
):
    """
    Add messages to a session's memory

    Args:
        session_id: The session ID
        memory_messages: Messages and optional context to add
        namespace: Optional namespace for the session

    Returns:
        Acknowledgement response
    """
    redis = get_redis_conn()

    try:
        # Define keys
        messages_key = Keys.messages_key(session_id)
        context_key = Keys.context_key(session_id)
        sessions_key = f"sessions:{namespace}" if namespace else "sessions"

        # Check if new context is provided
        if memory_messages.context is not None:
            await redis.set(context_key, memory_messages.context)

        # Add session to sessions set with timestamp
        current_time = int(time.time())
        await redis.zadd(sessions_key, {session_id: current_time})

        # Get model client for extraction
        model_client = await get_model_client(settings.generation_model)

        messages_json = []

        # Process messages for topic/entity extraction
        for msg in memory_messages.messages:
            # Handle extraction in background for each message
            msg = await handle_extraction(msg)
            msg_dict = msg.model_dump()
            # Convert lists to comma-separated strings for TAG fields
            msg_dict["topics"] = ",".join(msg.topics) if msg.topics else ""
            msg_dict["entities"] = ",".join(msg.entities) if msg.entities else ""
            messages_json.append(json.dumps(msg_dict))

        # Add messages to list
        await redis.lpush(messages_key, *messages_json)  # type: ignore

        # Check if window size is exceeded
        current_size = await redis.llen(messages_key)  # type: ignore
        if current_size > settings.window_size:
            # Handle compaction in background
            background_tasks.add_task(
                handle_compaction,
                session_id,
                settings.generation_model,
                settings.window_size,
                model_client,
                redis,
            )

        # If long-term memory is enabled, index messages
        if settings.long_term_memory:
            embedding_client = await get_openai_client()
            background_tasks.add_task(
                index_messages,
                memory_messages.messages,
                session_id,
                embedding_client,
                redis,
            )

        return AckResponse(status="ok")
    except Exception as e:
        logger.error(f"Error adding messages for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.delete("/sessions/{session_id}/memory", response_model=AckResponse)
async def delete_memory(
    session_id: str,
    namespace: str | None = None,
):
    """
    Delete a session's memory

    Args:
        session_id: The session ID
        namespace: Optional namespace for the session

    Returns:
        Acknowledgement response
    """
    redis = get_redis_conn()
    try:
        # Define keys
        messages_key = Keys.messages_key(session_id)
        context_key = Keys.context_key(session_id)
        token_count_key = Keys.token_count_key(session_id)
        sessions_key = f"sessions:{namespace}" if namespace else "sessions"

        # Create pipeline for deletion
        pipe = redis.pipeline()
        pipe.delete(messages_key, context_key, token_count_key)
        pipe.zrem(sessions_key, session_id)
        await pipe.execute()

        return AckResponse(status="ok")
    except Exception as e:
        logger.error(f"Error deleting memory for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
