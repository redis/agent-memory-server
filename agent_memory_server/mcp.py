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
    # Simple filters for backward compatibility
    session_id: str | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    # Extended filter options
    # Session ID filters
    session_id_ne: str | None = None,
    session_ids_any: list[str] | None = None,
    session_ids_all: list[str] | None = None,
    # Namespace filters
    namespace_ne: str | None = None,
    namespaces_any: list[str] | None = None,
    namespaces_all: list[str] | None = None,
    # Topics filters
    topics_ne: str | None = None,
    topics_any: list[str] | None = None,
    topics_all: list[str] | None = None,
    # Entities filters
    entities_ne: str | None = None,
    entities_any: list[str] | None = None,
    entities_all: list[str] | None = None,
    # User ID filters
    user_id_ne: str | None = None,
    user_ids_any: list[str] | None = None,
    user_ids_all: list[str] | None = None,
    # Time range filters
    created_at_gt: int | None = None,
    created_at_lt: int | None = None,
    created_at_gte: int | None = None,
    created_at_lte: int | None = None,
    created_at_eq: int | None = None,
    created_at_ne: int | None = None,
    created_at_between: list[float] | None = None,
    last_accessed_gt: int | None = None,
    last_accessed_lt: int | None = None,
    last_accessed_gte: int | None = None,
    last_accessed_lte: int | None = None,
    last_accessed_eq: int | None = None,
    last_accessed_ne: int | None = None,
    last_accessed_between: list[float] | None = None,
    # Pagination and other options
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
) -> LongTermMemoryResults:
    """
    Search for long-term memories relevant to a query with advanced filtering.

    Args:
        query: The query to search for.

        # Basic filters
        session_id: Filter by exact session ID.
        namespace: Filter by exact namespace.
        user_id: Filter by exact user ID.
        topics: Filter by topics (contains any).
        entities: Filter by entities (contains any).

        # Session ID filters
        session_id_ne: Exclude this session ID.
        session_ids_any: Sessions containing any of these IDs.
        session_ids_all: Sessions containing all of these IDs.

        # Namespace filters
        namespace_ne: Exclude this namespace.
        namespaces_any: Namespaces containing any of these.
        namespaces_all: Namespaces containing all of these.

        # Topics filters
        topics_ne: Exclude this topic.
        topics_any: Contain any of these topics.
        topics_all: Contain all of these topics.

        # Entities filters
        entities_ne: Exclude this entity.
        entities_any: Contain any of these entities.
        entities_all: Contain all of these entities.

        # User ID filters
        user_id_ne: Exclude this user ID.
        user_ids_any: Any of these user IDs.
        user_ids_all: All of these user IDs.

        # Time range filters
        created_at_gt: Created after this timestamp.
        created_at_lt: Created before this timestamp.
        created_at_gte: Created at or after this timestamp.
        created_at_lte: Created at or before this timestamp.
        created_at_eq: Created at exactly this timestamp.
        created_at_ne: Not created at this timestamp.
        created_at_between: Created between these timestamps (inclusive).

        # Last accessed filters
        last_accessed_gt: Last accessed after this timestamp.
        last_accessed_lt: Last accessed before this timestamp.
        last_accessed_gte: Last accessed at or after this timestamp.
        last_accessed_lte: Last accessed at or before this timestamp.
        last_accessed_eq: Last accessed at exactly this timestamp.
        last_accessed_ne: Not last accessed at this timestamp.
        last_accessed_between: Last accessed between these timestamps (inclusive).

        # Other options
        distance_threshold: Maximum semantic distance for results.
        limit: Maximum number of results to return.
        offset: Offset for pagination.

    Returns:
        A list of long-term memories that match the query and filters.
    """
    # Create a payload with proper filter objects
    payload = SearchPayload.create_with_primitives(
        text=query,
        # Session ID filters
        session_id=session_id,
        session_id_ne=session_id_ne,
        session_ids_any=session_ids_any,
        session_ids_all=session_ids_all,
        # Namespace filters
        namespace=namespace,
        namespace_ne=namespace_ne,
        namespaces_any=namespaces_any,
        namespaces_all=namespaces_all,
        # Topics filters
        topics=topics,
        topics_ne=topics_ne,
        topics_any=topics_any,
        topics_all=topics_all,
        # Entities filters
        entities=entities,
        entities_ne=entities_ne,
        entities_any=entities_any,
        entities_all=entities_all,
        # User ID filters
        user_id=user_id,
        user_id_ne=user_id_ne,
        user_ids_any=user_ids_any,
        user_ids_all=user_ids_all,
        # Time range filters
        created_at_gt=created_at_gt,
        created_at_lt=created_at_lt,
        created_at_gte=created_at_gte,
        created_at_lte=created_at_lte,
        created_at_eq=created_at_eq,
        created_at_ne=created_at_ne,
        created_at_between=created_at_between,
        last_accessed_gt=last_accessed_gt,
        last_accessed_lt=last_accessed_lt,
        last_accessed_gte=last_accessed_gte,
        last_accessed_lte=last_accessed_lte,
        last_accessed_eq=last_accessed_eq,
        last_accessed_ne=last_accessed_ne,
        last_accessed_between=last_accessed_between,
        # Other options
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

    long_term_memories = []
    try:
        # Create a search payload with proper filter objects
        search_payload = SearchPayload.create_with_primitives(
            text=query,
            session_id=session_id,
            namespace=namespace,
        )
        long_term_memories = await core_search_long_term_memory(search_payload)
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
    except Exception as e:
        logger.error(f"Error searching long-term memory: {e}")

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
