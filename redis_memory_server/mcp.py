import logging

from fastapi import BackgroundTasks
from mcp.server.fastmcp import FastMCP
from mcp.types import Prompt, PromptArgument, PromptMessage, TextContent

from redis_memory_server.api import (
    delete_memory as core_delete_memory,
    get_session_memory as core_get_session_memory,
    list_sessions as core_list_sessions,
    post_memory as core_post_memory,
    search_session_messages as core_search_messages,
)
from redis_memory_server.models.messages import (
    AckResponse,
    GetSessionsQuery,
    MemoryMessagesAndContext,
    MemoryResponse,
    SearchPayload,
    SearchResults,
)


logger = logging.getLogger(__name__)
mcp_app = FastMCP("Redis Agentic Memory Server")


@mcp_app.tool()
async def list_sessions(
    page: int = 1, size: int = 10, namespace: str | None = None
) -> list[str]:
    """List available memory sessions"""
    return await core_list_sessions(
        GetSessionsQuery(page=page, size=size, namespace=namespace)
    )


@mcp_app.resource("memory://{session_id}/memory")
async def get_session_memory(session_id: str) -> MemoryResponse:
    """Get memory for a specific session"""
    return await core_get_session_memory(session_id)


@mcp_app.tool()
async def add_memory(
    session_id: str,
    memory_messages: MemoryMessagesAndContext,
    namespace: str | None = None,
) -> AckResponse:
    """Add messages to a session's memory"""
    background_tasks = BackgroundTasks()

    return await core_post_memory(
        session_id, memory_messages, background_tasks, namespace
    )


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
    namespace: str | None = None,
) -> SearchResults:
    """Search through a session's memory"""
    return await core_search_messages(session_id, SearchPayload(text=query), namespace)


@mcp_app.prompt()
async def memory_prompt(
    session_id: str,
    query: str,
    namespace: str | None = None,
) -> Prompt:
    """A prompt to enrich a user query with memory context"""
    messages = []

    try:
        memory = await core_get_session_memory(session_id)
        if memory.context:
            messages.append(
                PromptMessage(
                    role="assistant",
                    content=TextContent(
                        type="text",
                        text=f"## Context related to the current conversation\n{memory.context}",
                    ),
                )
            )
        for msg in memory.messages:
            messages.append(
                PromptMessage(
                    role="user" if msg.role == "user" else "assistant",
                    content=TextContent(type="text", text=msg.content),
                )
            )
    except Exception:
        logger.exception(f"Could not load memory for session {session_id}")

    long_term_memories = await core_search_messages(
        session_id, SearchPayload(text=query), namespace
    )
    if long_term_memories.total > 0:
        long_term_memories_text = "\n".join(
            [
                f"Role: {m.role}\nContent: {m.content}\nDistance: {m.dist}"
                for m in long_term_memories.docs
            ]
        )
        messages.append(
            PromptMessage(
                role="assistant",
                content=TextContent(
                    type="text",
                    text=f"## Long term memories related to the user's query\n {long_term_memories_text}",
                ),
            )
        )

    messages.append(
        PromptMessage(
            role="user",
            content=TextContent(type="text", text=query),
        )
    )

    return Prompt(
        name="memory-prompt",
        description="A prompt containing the user's query enriched with memory context",
        arguments=[
            PromptArgument(
                name="session_id",
                description="The session ID to interact with",
                required=False,
            ),
            PromptArgument(
                name="query",
                description="The query or message to process",
                required=False,
            ),
        ],
    )
