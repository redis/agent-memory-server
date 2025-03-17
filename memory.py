from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from redis.asyncio import Redis
from typing import List, Optional
import json
import logging
import time
import asyncio

from models import (
    MemoryMessage,
    MemoryMessagesAndContext,
    MemoryResponse,
    AckResponse,
    GetSessionsQuery,
)
from reducers import handle_compaction
from long_term_memory import index_messages
from utils import Keys, get_openai_client, get_redis_conn
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions/", response_model=List[str])
async def get_sessions(
    pagination: GetSessionsQuery = Depends(GetSessionsQuery),
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
        session_ids = [
            s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids
        ]

        return session_ids
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}/memory", response_model=MemoryResponse)
async def get_memory(
    session_id: str
):
    """
    Get memory for a session

    Args:
        session_id: The session ID
        request: FastAPI request

    Returns:
        Memory response with messages and context
    """
    redis = get_redis_conn()
    print(f"Redis connection: {redis.connection_pool.connection_kwargs}")

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
            msg = json.loads(msg_raw)
            memory_messages.append(MemoryMessage(**msg))

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
        response = MemoryResponse(
            messages=memory_messages, context=context, tokens=tokens
        )

        return response
    except Exception as e:
        logger.error(f"Error getting memory for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sessions/{session_id}/memory", response_model=AckResponse)
async def post_memory(
    session_id: str,
    memory_messages: MemoryMessagesAndContext,
    background_tasks: BackgroundTasks,
    namespace: Optional[str] = None,
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

        # Add messages to session list
        # TODO: Don't need a pipeline here, lpush takes multiple values.
        pipe = redis.pipeline()
        for msg in memory_messages.messages:
            # Convert to dict and serialize
            msg_json = json.dumps(msg.model_dump())
            pipe.lpush(messages_key, msg_json)

        # Execute pipeline
        await pipe.execute()

        # Check if window size is exceeded
        current_size = await redis.llen(messages_key)
        if current_size > settings.window_size:
            # Handle compaction in background
            background_tasks.add_task(
                handle_compaction,
                session_id,
                settings.generation_model,
                settings.window_size,
                await get_openai_client(),
                redis,
            )

        # If long-term memory is enabled, index messages.
        # 
        # TODO: Add support for custom policies around when to index and/or
        # when to "move" memories from short-term to long-term memory, if
        # that implies deleting them from short-term memory.
        if settings.long_term_memory:
            background_tasks.add_task(
                index_messages,
                memory_messages.messages,
                session_id,
                await get_openai_client(),
                redis,
            )

        return AckResponse(status="ok")
    except Exception as e:
        logger.error(f"Error adding messages for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/sessions/{session_id}/memory", response_model=AckResponse)
async def delete_memory(
    session_id: str,
    namespace: Optional[str] = None,
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
        raise
        raise HTTPException(status_code=500, detail="Internal server error")
