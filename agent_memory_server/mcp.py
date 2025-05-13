import asyncio
import logging
import os
import sys

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP as _FastMCPBase
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server.api import (
    create_long_term_memory as core_create_long_term_memory,
    get_session_memory as core_get_session_memory,
    search_long_term_memory as core_search_long_term_memory,
)
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryPayload,
    LongTermMemory,
    LongTermMemoryResults,
    SearchPayload,
)


logger = logging.getLogger(__name__)

# Default namespace for STDIO mode
DEFAULT_NAMESPACE = os.getenv("MCP_NAMESPACE")


class FastMCP(_FastMCPBase):
    """Extend FastMCP to support optional URL namespace and default STDIO namespace."""

    def __init__(self, *args, default_namespace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_namespace = default_namespace
        self._current_request = None  # Initialize the attribute

    def sse_app(self):
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Mount, Route

        sse = SseServerTransport(self.settings.message_path)

        async def handle_sse(request: Request) -> None:
            # Store the request in the FastMCP instance so call_tool can access it
            self._current_request = request

            try:
                async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,  # type: ignore
                ) as (read_stream, write_stream):
                    await self._mcp_server.run(
                        read_stream,
                        write_stream,
                        self._mcp_server.create_initialization_options(),
                    )
            finally:
                # Clean up request reference
                self._current_request = None

        return Starlette(
            debug=self.settings.debug,
            routes=[
                Route(self.settings.sse_path, endpoint=handle_sse),
                Route(f"/{{namespace}}{self.settings.sse_path}", endpoint=handle_sse),
                Mount(self.settings.message_path, app=sse.handle_post_message),
                Mount(
                    f"/{{namespace}}{self.settings.message_path}",
                    app=sse.handle_post_message,
                ),
            ],
        )

    async def call_tool(self, name, arguments):
        # Get the namespace from the request context
        namespace = None
        try:
            # RequestContext doesn't expose the path_params directly
            # We use a ThreadLocal or context variable pattern instead
            from starlette.requests import Request

            request = getattr(self, "_current_request", None)
            if isinstance(request, Request):
                namespace = request.path_params.get("namespace")
        except Exception:
            # Silently continue if we can't get namespace from request
            pass

        # Inject namespace only for tools that accept it
        if name in ("search_long_term_memory", "hydrate_memory_prompt"):
            if namespace and "namespace" not in arguments:
                arguments["namespace"] = Namespace(eq=namespace)
            elif (
                not namespace
                and self.default_namespace
                and "namespace" not in arguments
            ):
                arguments["namespace"] = Namespace(eq=self.default_namespace)

        return await super().call_tool(name, arguments)


INSTRUCTIONS = """
    When responding to user queries, ALWAYS check memory first before answering
    questions about user preferences, history, or personal information.
"""


mcp_app = FastMCP(
    "Redis Agent Memory Server",
    port=settings.mcp_port,
    instructions=INSTRUCTIONS,
    default_namespace=DEFAULT_NAMESPACE,
)


@mcp_app.tool()
async def create_long_term_memories(
    memories: list[LongTermMemory],
) -> AckResponse:
    """
    Create long-term memories that can be searched later.

    This tool saves memories contained in the payload for future retrieval.

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id, use the EXACT session identifier from the current conversation
    - NEVER invent or guess a session ID - if you don't know it, omit the field
    - If you want memories accessible across all sessions, omit the session_id field

    COMMON USAGE PATTERNS:

    1. Basic memory creation:
    ```python
    create_long_term_memories(
        memories=[
          {
            "text": "The user prefers dark mode in all applications",
            "user_id": "user_789",
            "namespace": "user_preferences",
            "topics": ["preferences", "ui"],
          }
        ]
    )
    ```

    2. Create multiple memories at once:
    ```python
    create_long_term_memories(
        memories=[
            {"text": "Memory 1"},
            {"text": "Memory 2"},
        ]
    )

    3. Create memories with different namespaces:
    ```python
    create_long_term_memories(
        memories=[
            {"text": "Memory 1", "namespace": "user_preferences"},
            {"text": "Memory 2", "namespace": "user_settings"},
        ]
    )
    ```

    Args:
        memories: A list of LongTermMemory objects to create

    Returns:
        An acknowledgement response indicating success
    """
    # Apply default namespace for STDIO if not provided in memory entries
    if DEFAULT_NAMESPACE:
        for mem in memories:
            if mem.namespace is None:
                mem.namespace = DEFAULT_NAMESPACE

    payload = CreateLongTermMemoryPayload(memories=memories)
    return await core_create_long_term_memory(
        payload, background_tasks=get_background_tasks()
    )


@mcp_app.tool()
async def search_long_term_memory(
    text: str | None,
    session_id: SessionId | None = None,
    namespace: Namespace | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    user_id: UserId | None = None,
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
) -> LongTermMemoryResults:
    """
    Search for memories related to a text query.

    Finds memories based on a combination of semantic similarity and input filters.

    This tool performs a semantic search on stored memories using the query text and filters
    in the payload. Results are ranked by relevance.

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id filter, use the EXACT session identifier
    - NEVER invent or guess a session ID - if you don't know it, omit this filter
    - If you want to search across all sessions, don't include a session_id filter
    - Session IDs from examples will NOT work with real data

    COMMON USAGE PATTERNS:

    1. Basic search with just query text:
    ```python
    search_long_term_memory(text="user's favorite color")
    ```

    2. Search with simple session filter:
    ```python
    search_long_term_memory(text="user's favorite color", session_id={
        "eq": "session_12345"
    })
    ```

    3. Search with complex filters:
    ```python
    search_long_term_memory(
        text="user preferences",
        topics={
            "any": ["preferences", "settings"]
        },
        created_at={
            "gt": 1640995200
        },
        limit=5
    )
    ```

    Args:
        text: The semantic search query text (required)
        session_id: Filter by session ID
        namespace: Filter by namespace
        topics: Filter by topics
        entities: Filter by entities
        created_at: Filter by creation date
        last_accessed: Filter by last access date
        user_id: Filter by user ID
        distance_threshold: Distance threshold for semantic search
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        LongTermMemoryResults containing matched memories sorted by relevance
    """
    try:
        payload = SearchPayload(
            text=text,
            session_id=session_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            user_id=user_id,
            distance_threshold=distance_threshold,
            limit=limit,
            offset=offset,
        )
        results = await core_search_long_term_memory(payload)
        results = LongTermMemoryResults(
            total=results.total,
            memories=results.memories,
            next_offset=results.next_offset,
        )
    except Exception as e:
        logger.error(f"Error in search_long_term_memory tool: {e}")
        results = LongTermMemoryResults(
            total=0,
            memories=[],
            next_offset=None,
        )
    return results


# NOTE: Prompts don't support search filters in FastMCP, so we need to use a
# tool instead.
@mcp_app.tool()
async def hydrate_memory_prompt(
    text: str | None = None,
    session_id: SessionId | None = None,
    namespace: Namespace | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    user_id: UserId | None = None,
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[base.Message]:
    """
    Hydrate a user prompt with relevant session history and long-term memories.

    CRITICAL: Use this tool for EVERY question that might benefit from memory context,
    especially when you don't have sufficient information to answer confidently.

    This tool enriches the user's query by retrieving:
    1. Context from the current conversation session
    2. Relevant long-term memories related to the query

    ALWAYS use this tool when:
    - The user references past conversations
    - The question is about user preferences or personal information
    - You need additional context to provide a complete answer
    - The question seems to assume information you don't have in current context

    The function uses the text field from the payload as the user's query,
    and any filters to retrieve relevant memories.

    IMPORTANT NOTES ON SESSION IDs:
    - When filtering by session_id, you must provide the EXACT session identifier
    - NEVER invent or guess a session ID - if you don't know it, omit this filter
    - Session IDs from examples will NOT work with real data

    COMMON USAGE PATTERNS:
    ```python
    1. Basic search with just query text:
    hydrate_memory_prompt(text="What was my favorite color?")
    ```

    2. Search with simple session filter:
    hydrate_memory_prompt(
        text="What was my favorite color?",
        session_id={
            "eq": "session_12345"
        },
        namespace={
            "eq": "user_preferences"
        }
    )

    3. Search with complex filters:
    hydrate_memory_prompt(
        text="What was my favorite color?",
        topics={
            "any": ["preferences", "settings"]
        },
        created_at={
            "gt": 1640995200
        },
        limit=5
    )
    ```

    Args:
        - text: The user's query/message (required)
        - session_id: Filter by session ID
        - namespace: Filter by namespace
        - topics: Filter by topics
        - entities: Filter by entities
        - created_at: Filter by creation date
        - last_accessed: Filter by last access date
        - user_id: Filter by user ID
        - distance_threshold: Distance threshold for semantic search
        - limit: Maximum number of results
        - offset: Offset for pagination

    Returns:
        A list of messages, including memory context and the user's query
    """
    messages = []
    if session_id and session_id.eq:
        try:
            session_memory = await core_get_session_memory(session_id.eq)
        except HTTPException:
            session_memory = None
    else:
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

    try:
        long_term_memories = await core_search_long_term_memory(
            SearchPayload(
                text=text,
                session_id=session_id,
                namespace=namespace,
                topics=topics,
                entities=entities,
                created_at=created_at,
                last_accessed=last_accessed,
                user_id=user_id,
                distance_threshold=distance_threshold,
                limit=limit,
                offset=offset,
            )
        )
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

    # Ensure text is not None
    safe_text = text or ""
    messages.append(
        base.UserMessage(
            content=TextContent(type="text", text=safe_text),
        )
    )

    return messages


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        asyncio.run(mcp_app.run_sse_async())
    else:
        asyncio.run(mcp_app.run_stdio_async())
