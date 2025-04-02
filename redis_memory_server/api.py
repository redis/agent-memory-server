import json
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from redis_memory_server.config import settings
from redis_memory_server.logging import get_logger
from redis_memory_server.models.extraction import handle_extraction
from redis_memory_server.models.messages import (
    AckResponse,
    GetSessionsQuery,
    MemoryMessage,
    MemoryMessagesAndContext,
    MemoryResponse,
    SearchPayload,
    SearchResults,
    index_messages,
    search_messages,
)
from redis_memory_server.models.summarization import handle_compaction
from redis_memory_server.utils import (
    Keys,
    get_model_client,
    get_openai_client,
    get_redis_conn,
)


logger = get_logger(__name__)

router = APIRouter()


@router.get("/sessions/", response_model=list[str])
async def list_sessions(
    pagination: GetSessionsQuery = Depends(),
):
    """
    Get a list of session IDs, with optional pagination.

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
    sessions_key = Keys.sessions_key(namespace=pagination.namespace)

    try:
        # Get session IDs from Redis
        session_ids = await redis.zrange(sessions_key, start, end)

        # Convert from bytes to strings if needed
        return [s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids]

    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/sessions/{session_id}/memory", response_model=MemoryResponse)
async def get_session_memory(session_id: str, namespace: str | None = None):
    """
    Get memory for a session.

    This includes stored conversation history and context.

    Args:
        session_id: The session ID

    Returns:
        Conversation history and context
    """
    redis = get_redis_conn()

    try:
        # Define keys
        sessions_key = Keys.sessions_key(namespace=namespace)
        messages_key = Keys.messages_key(session_id, namespace=namespace)
        context_key = Keys.context_key(session_id, namespace=namespace)
        token_count_key = Keys.token_count_key(session_id, namespace=namespace)

        # TODO: Use a hash
        session_exists = await redis.zscore(sessions_key, session_id)
        if not session_exists:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get data from Redis in a pipeline
        pipe = redis.pipeline()
        # TODO: Make window size configurable via API parameter
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

    except HTTPException as e:
        raise e
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

        if memory_messages.context is not None:
            await redis.set(context_key, memory_messages.context)

        current_time = int(time.time())
        await redis.zadd(sessions_key, {session_id: current_time})

        model_client = await get_model_client(settings.generation_model)
        messages_json = []

        # Process messages for topic/entity extraction
        # TODO: Use a distributed background task
        for msg in memory_messages.messages:
            # Handle extraction in background for each message
            msg = await handle_extraction(msg)
            msg_dict = msg.model_dump()
            # Convert lists to comma-separated strings for TAG fields
            msg_dict["topics"] = ",".join(msg.topics) if msg.topics else ""
            msg_dict["entities"] = ",".join(msg.entities) if msg.entities else ""
            messages_json.append(json.dumps(msg_dict))

        # Add messages to list
        await redis.rpush(messages_key, *messages_json)  # type: ignore

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
        # TODO: Use a distributed background task
        if settings.long_term_memory:
            embedding_client = await get_openai_client()
            background_tasks.add_task(
                index_messages,
                memory_messages.messages,
                session_id,
                embedding_client,
                redis,
                namespace,
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


@router.post("/sessions/{session_id}/search", response_model=SearchResults)
async def search_session_messages(
    session_id: str,
    payload: SearchPayload,
    namespace: str | None = None,
):
    """
    Run a semantic search on the messages in a session

    Args:
        session_id: The session ID
        payload: Search payload with text to search for
        namespace: Optional namespace for the session

    Returns:
        List of search results
    """
    redis = get_redis_conn()

    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long term memory is disabled")

    # For embeddings, we always use OpenAI models since Anthropic doesn't support embeddings
    client = await get_openai_client()

    try:
        return await search_messages(
            payload.text,
            client,
            redis,
            session_id=session_id,
            namespace=namespace,
        )
    except Exception as e:
        logger.error(f"Error in retrieval API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
