from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from redis_memory_server import messages
from redis_memory_server.config import settings
from redis_memory_server.logging import get_logger
from redis_memory_server.models import (
    AckResponse,
    GetSessionsQuery,
    SearchPayload,
    SearchResults,
    SessionMemory,
    SessionMemoryResponse,
)
from redis_memory_server.utils import (
    get_openai_client,
    get_redis_conn,
)


logger = get_logger(__name__)

router = APIRouter()


@router.get("/sessions/", response_model=list[str])
async def list_sessions(
    options: GetSessionsQuery = Depends(),
):
    """
    Get a list of session IDs, with optional pagination.

    Args:
        options: Query parameters (page, size, namespace)

    Returns:
        List of session IDs
    """
    # TODO: Pydantic should validate this
    if options.page > 100:
        raise HTTPException(status_code=400, detail="Page must not exceed 100")

    redis = get_redis_conn()

    return await messages.list_sessions(
        redis=redis,
        page=options.page,
        size=options.size,
        namespace=options.namespace,
    )


@router.get("/sessions/{session_id}/memory", response_model=SessionMemoryResponse)
async def get_session_memory(
    session_id: str,
    namespace: str | None = None,
    window_size: int = settings.window_size,
):
    """
    Get memory for a session.

    This includes stored conversation history and context.

    Args:
        session_id: The session ID
        window_size: The number of messages to include in the response
        namespace: The namespace to use for the session

    Returns:
        Conversation history and context
    """
    redis = get_redis_conn()

    session = await messages.get_session_memory(
        redis=redis,
        session_id=session_id,
        window_size=window_size,
        namespace=namespace,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.put("/sessions/{session_id}/memory", response_model=AckResponse)
async def put_session_memory(
    session_id: str,
    memory: SessionMemory,
    background_tasks: BackgroundTasks,
):
    """
    Set session memory. Replaces existing session memory.

    Args:
        session_id: The session ID
        memory: Messages and context to save

    Returns:
        Acknowledgement response
    """
    redis = get_redis_conn()

    await messages.set_session_memory(
        redis=redis,
        session_id=session_id,
        memory=memory,
        background_tasks=background_tasks,
    )
    return AckResponse(status="ok")


@router.delete("/sessions/{session_id}/memory", response_model=AckResponse)
async def delete_session_memory(
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
    await messages.delete_session_memory(
        redis=redis,
        session_id=session_id,
        namespace=namespace,
    )
    return AckResponse(status="ok")


@router.post("/messages/search", response_model=SearchResults)
async def messages_search(payload: SearchPayload):
    """
    Run a semantic search on messages

    TODO: Infer topics for `text`

    Args:
        payload: Search payload

    Returns:
        List of search results
    """
    redis = get_redis_conn()

    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long term memory is disabled")

    # For embeddings, we always use OpenAI models since Anthropic doesn't support embeddings
    client = await get_openai_client()

    return await messages.search_messages(
        client=client,
        redis_conn=redis,
        **payload.model_dump(exclude_none=True),
    )
