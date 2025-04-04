import logging

from fastapi import BackgroundTasks, HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from redis_memory_server.api import (
    delete_session_memory as core_delete_memory,
    get_session_memory as core_get_session_memory,
    list_sessions as core_list_sessions,
    messages_search as core_search_messages,
    put_session_memory as core_put_session_memory,
)
from redis_memory_server.models import (
    AckResponse,
    GetSessionsQuery,
    MemoryMessage,
    SearchPayload,
    SearchResults,
    SessionMemory,
)


logger = logging.getLogger(__name__)
mcp_app = FastMCP("Redis Agent Memory Server")


@mcp_app.tool()
async def list_sessions(
    page: int = 1, size: int = 10, namespace: str | None = None
) -> list[str]:
    """List available memory sessions"""
    return await core_list_sessions(
        GetSessionsQuery(page=page, size=size, namespace=namespace)
    )


@mcp_app.resource("memory://{session_id}/memory")
async def get_session_memory(session_id: str) -> SessionMemory:
    """Get memory for a specific session"""
    return await core_get_session_memory(session_id)


@mcp_app.tool()
async def add_memory(
    session_id: str,
    memory: str,
    context: str | None = None,
    namespace: str | None = None,
) -> AckResponse:
    """Add a memory to a session"""
    background_tasks = BackgroundTasks()
    session_memory = SessionMemory(
        messages=[MemoryMessage(role="user", content=memory)],
        context=context,
        namespace=namespace,
    )

    result = await core_put_session_memory(session_id, session_memory, background_tasks)

    logger.warning(f"Background tasks: {background_tasks.tasks}")

    return result


@mcp_app.tool()
async def delete_session_memory(
    session_id: str,
    namespace: str | None = None,
) -> AckResponse:
    """Delete a session's memory"""
    return await core_delete_memory(session_id, namespace)


@mcp_app.tool()
async def search_memory(
    session_id: str,
    query: str,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
    namespace: str | None = None,
) -> SearchResults:
    """Search through a session's memory"""
    payload = SearchPayload(
        session_id=session_id,
        text=query,
        namespace=namespace,
        topics=topics,
        entities=entities,
        distance_threshold=distance_threshold,
        limit=limit,
        offset=offset,
    )
    return await core_search_messages(payload)


@mcp_app.prompt()
async def memory_prompt(
    session_id: str,
    query: str,
    namespace: str | None = None,
) -> list[base.Message]:
    """A prompt to enrich a user query with context from memory"""
    messages = []
    try:
        session_memory = await core_get_session_memory(session_id)
    except HTTPException:
        session_memory = None

    if session_memory:
        if session_memory.context:
            messages.append(
                base.AssistantMessage(
                    content=TextContent(
                        type="text",
                        text=f"## A summary of the conversation so far\n{session_memory.context}",
                    ),
                )
            )
        for msg in session_memory.messages:
            if msg.role == "user":
                msg_class = base.UserMessage
            else:
                msg_class = base.AssistantMessage
            messages.append(
                msg_class(
                    content=TextContent(type="text", text=msg.content),
                )
            )

    payload = SearchPayload(
        session_id=session_id,
        text=query,
        namespace=namespace,
    )
    long_term_memories = await core_search_messages(payload)
    if long_term_memories.total > 0:
        long_term_memories_text = "\n".join(
            [
                f"Role: {m.role}\nContent: {m.content}\nDistance: {m.dist}"
                for m in long_term_memories.docs
            ]
        )
        messages.append(
            base.AssistantMessage(
                content=TextContent(
                    type="text",
                    text=f"## Long term memories related to the user's query\n {long_term_memories_text}",
                ),
            )
        )

    messages.append(
        base.UserMessage(
            content=TextContent(type="text", text=query),
        )
    )

    return messages
