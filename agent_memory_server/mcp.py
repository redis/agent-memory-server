import logging
import sys

from fastapi import BackgroundTasks, HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server.api import (
    create_long_term_memory as core_create_long_term_memory,
    get_session_memory as core_get_session_memory,
    search_long_term_memory as core_search_long_term_memory,
)
from agent_memory_server.config import settings
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryPayload,
    LongTermMemory,
    LongTermMemoryResults,
    SearchPayload,
)


logger = logging.getLogger(__name__)
mcp_app = FastMCP("Redis Agent Memory Server", port=settings.mcp_port)


@mcp_app.tool()
async def create_long_term_memories(
    memories: list[LongTermMemory],
) -> AckResponse:
    """
    Create a long-term memory.

    Args:
        memories: A list of long-term memories to create.

    Returns:
        An acknowledgement of the creation.
    """
    payload = CreateLongTermMemoryPayload(memories=memories)
    return await core_create_long_term_memory(
        payload, background_tasks=BackgroundTasks()
    )


@mcp_app.tool()
async def search_long_term_memory(
    query: str,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
    namespace: str | None = None,
) -> LongTermMemoryResults:
    """
    Search for long-term memories relevant to a query.

    Args:
        query: The query to search for.
        topics: A list of topics to filter by.
        entities: A list of entities to filter by.
        distance_threshold: The distance threshold to use for the search.
        limit: The maximum number of results to return.
        offset: The offset to use for the search.

    Returns:
        A list of long-term memories that match the query.
    """
    payload = SearchPayload(
        text=query,
        namespace=namespace,
        topics=topics,
        entities=entities,
        distance_threshold=distance_threshold,
        limit=limit,
        offset=offset,
    )
    return await core_search_long_term_memory(payload)


@mcp_app.prompt()
async def memory_prompt(
    session_id: str,
    query: str,
    namespace: str | None = None,
) -> list[base.Message]:
    """
    A prompt to enrich a user query with context from memory.

    Args:
        query: The query to enrich.
        namespace: The namespace to use for the search.

    Returns:
        A list of messages with the enriched query.
    """
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

    long_term_memories = await core_search_long_term_memory(payload)
    if long_term_memories.total > 0:
        long_term_memories_text = "\n".join(
            [f"- {m.text}" for m in long_term_memories.memories]
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        mcp_app.run(transport="sse")
    else:
        mcp_app.run(transport="stdio")
