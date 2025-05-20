from fastapi import APIRouter, Depends, HTTPException
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server import long_term_memory, messages
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.llms import get_model_config
from agent_memory_server.logging import get_logger
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryRequest,
    GetSessionsQuery,
    LongTermMemoryResultsResponse,
    MemoryPromptRequest,
    MemoryPromptResponse,
    ModelNameLiteral,
    SearchRequest,
    SessionListResponse,
    SessionMemory,
    SessionMemoryResponse,
    SystemMessage,
)
from agent_memory_server.utils.redis import get_redis_conn


logger = get_logger(__name__)

router = APIRouter()


def _get_effective_window_size(
    window_size: int,
    context_window_max: int | None,
    model_name: ModelNameLiteral | None,
) -> int:
    # If context_window_max is explicitly provided, use that
    if context_window_max is not None:
        effective_window_size = min(window_size, context_window_max)
    # If model_name is provided, get its max_tokens from our config
    elif model_name is not None:
        model_config = get_model_config(model_name)
        effective_window_size = min(window_size, model_config.max_tokens)
    # Otherwise use the default window_size
    else:
        effective_window_size = window_size
    return effective_window_size


@router.get("/sessions/", response_model=SessionListResponse)
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
    redis = await get_redis_conn()

    total, session_ids = await messages.list_sessions(
        redis=redis,
        limit=options.limit,
        offset=options.offset,
        namespace=options.namespace,
    )

    return SessionListResponse(
        sessions=session_ids,
        total=total,
    )


@router.get("/sessions/{session_id}/memory", response_model=SessionMemoryResponse)
async def get_session_memory(
    session_id: str,
    namespace: str | None = None,
    window_size: int = settings.window_size,
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
):
    """
    Get memory for a session.

    This includes stored conversation history and context.

    Args:
        session_id: The session ID
        namespace: The namespace to use for the session
        window_size: The number of messages to include in the response
        model_name: The client's LLM model name (will determine context window size if provided)
        context_window_max: Direct specification of the context window max tokens (overrides model_name)

    Returns:
        Conversation history and context
    """
    redis = await get_redis_conn()
    effective_window_size = _get_effective_window_size(
        window_size=window_size,
        context_window_max=context_window_max,
        model_name=model_name,
    )

    session = await messages.get_session_memory(
        redis=redis,
        session_id=session_id,
        window_size=effective_window_size,
        namespace=namespace,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.put("/sessions/{session_id}/memory", response_model=AckResponse)
async def put_session_memory(
    session_id: str,
    memory: SessionMemory,
    background_tasks=Depends(get_background_tasks),
):
    """
    Set session memory. Replaces existing session memory.

    Args:
        session_id: The session ID
        memory: Messages and context to save
        background_tasks: DocketBackgroundTasks instance (injected automatically)

    Returns:
        Acknowledgement response
    """
    redis = await get_redis_conn()

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
    redis = await get_redis_conn()
    await messages.delete_session_memory(
        redis=redis,
        session_id=session_id,
        namespace=namespace,
    )
    return AckResponse(status="ok")


@router.post("/long-term-memory", response_model=AckResponse)
async def create_long_term_memory(
    payload: CreateLongTermMemoryRequest,
    background_tasks=Depends(get_background_tasks),
):
    """
    Create a long-term memory

    Args:
        payload: Long-term memory payload
        background_tasks: DocketBackgroundTasks instance (injected automatically)

    Returns:
        Acknowledgement response
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    await background_tasks.add_task(
        long_term_memory.index_long_term_memories,
        memories=payload.memories,
    )
    return AckResponse(status="ok")


@router.post("/long-term-memory/search", response_model=LongTermMemoryResultsResponse)
async def search_long_term_memory(payload: SearchRequest):
    """
    Run a semantic search on long-term memory with filtering options.

    Args:
        payload: Search payload with filter objects for precise queries

    Returns:
        List of search results
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    redis = await get_redis_conn()

    # Extract filter objects from the payload
    filters = payload.get_filters()

    kwargs = {
        "redis": redis,
        "distance_threshold": payload.distance_threshold,
        "limit": payload.limit,
        "offset": payload.offset,
        **filters,
    }

    if payload.text:
        kwargs["text"] = payload.text

    # Pass text, redis, and filter objects to the search function
    return await long_term_memory.search_long_term_memories(**kwargs)


@router.post("/memory-prompt", response_model=MemoryPromptResponse)
async def memory_prompt(params: MemoryPromptRequest) -> MemoryPromptResponse:
    """
    Hydrate a user query with memory context and return a prompt
    ready to send to an LLM.

    `query` is the input text that the caller of this API wants to use to find
    relevant context. If `session_id` is provided and matches an existing
    session, the resulting prompt will include those messages as the immediate
    history of messages leading to a message containing `query`.

    If `long_term_search_payload` is provided, the resulting prompt will include
    relevant long-term memories found via semantic search with the options
    provided in the payload.

    Args:
        params: MemoryPromptRequest

    Returns:
        List of messages to send to an LLM, hydrated with relevant memory context
    """
    if not params.session and not params.long_term_search:
        raise HTTPException(
            status_code=400,
            detail="Either session or long_term_search must be provided",
        )

    redis = await get_redis_conn()
    _messages = []

    if params.session:
        effective_window_size = _get_effective_window_size(
            window_size=params.session.window_size,
            context_window_max=params.session.context_window_max,
            model_name=params.session.model_name,
        )
        session_memory = await messages.get_session_memory(
            redis=redis,
            session_id=params.session.session_id,
            window_size=effective_window_size,
            namespace=params.session.namespace,
        )

        if session_memory:
            if session_memory.context:
                # TODO: Weird to use MCP types here?
                _messages.append(
                    SystemMessage(
                        content=TextContent(
                            type="text",
                            text=f"## A summary of the conversation so far\n{session_memory.context}",
                        ),
                    )
                )
            # Ignore past system messages as the latest context may have changed
            for msg in session_memory.messages:
                if msg.role == "user":
                    msg_class = base.UserMessage
                else:
                    msg_class = base.AssistantMessage
                _messages.append(
                    msg_class(
                        content=TextContent(type="text", text=msg.content),
                    )
                )

    if params.long_term_search:
        # TODO: Exclude session messages if we already included them from session memory
        long_term_memories = await search_long_term_memory(
            params.long_term_search,
        )

        if long_term_memories.total > 0:
            long_term_memories_text = "\n".join(
                [f"- {m.text}" for m in long_term_memories.memories]
            )
            _messages.append(
                SystemMessage(
                    content=TextContent(
                        type="text",
                        text=f"## Long term memories related to the user's query\n {long_term_memories_text}",
                    ),
                )
            )

    _messages.append(
        base.UserMessage(
            content=TextContent(type="text", text=params.query),
        )
    )

    return MemoryPromptResponse(messages=_messages)
