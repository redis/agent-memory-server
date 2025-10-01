"""
LangChain integration for agent-memory-client.

This module provides automatic conversion of memory client tools to LangChain-compatible
tools, eliminating the need for manual wrapping with @tool decorators.

Example:
    ```python
    from agent_memory_client import create_memory_client
    from agent_memory_client.integrations.langchain import get_memory_tools
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # Get LangChain-compatible tools (no manual wrapping needed!)
    tools = get_memory_tools(
        memory_client=memory_client,
        session_id="my_session",
        user_id="user_123"
    )

    # Use with LangChain agent
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    # Run the agent
    result = await executor.ainvoke({"input": "Remember that I love pizza"})
    ```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent_memory_client import MemoryAPIClient

try:
    from langchain_core.tools import StructuredTool  # type: ignore  # noqa: F401

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class StructuredTool:  # type: ignore[no-redef]
        """Placeholder for when LangChain is not installed."""

        pass


def _check_langchain_available() -> None:
    """Check if LangChain is installed and raise helpful error if not."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required to use this integration. "
            "Install it with: pip install langchain-core"
        )


def get_memory_tools(
    memory_client: MemoryAPIClient,
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    tools: Sequence[str] | Literal["all"] = "all",
) -> list[StructuredTool]:
    """
    Get LangChain-compatible tools from a memory client.

    This function automatically converts memory client tools to LangChain StructuredTool
    instances, eliminating the need for manual @tool decorator wrapping.

    Args:
        memory_client: Initialized MemoryAPIClient instance
        session_id: Session ID to use for working memory operations
        user_id: Optional user ID for memory operations
        namespace: Optional namespace for memory operations
        tools: Which tools to include. Either "all" or a list of tool names.
               Available tools:
               - "search_memory"
               - "get_or_create_working_memory"
               - "add_memory_to_working_memory"
               - "update_working_memory_data"
               - "get_long_term_memory"
               - "create_long_term_memory"
               - "edit_long_term_memory"
               - "delete_long_term_memories"
               - "get_current_datetime"

    Returns:
        List of LangChain StructuredTool instances ready to use with agents

    Raises:
        ImportError: If langchain-core is not installed

    Example:
        ```python
        # Get all memory tools
        tools = get_memory_tools(
            memory_client=client,
            session_id="chat_session",
            user_id="alice"
        )

        # Get specific tools only
        tools = get_memory_tools(
            memory_client=client,
            session_id="chat_session",
            user_id="alice",
            tools=["search_memory", "create_long_term_memory"]
        )
        ```
    """
    _check_langchain_available()

    # Define all available tools with their configurations
    tool_configs = {
        "search_memory": {
            "name": "search_memory",
            "description": "Search long-term memory for relevant information using semantic search. Use this to recall past conversations, user preferences, or stored facts. Returns memories ranked by relevance with scores.",
            "func": _create_search_memory_func(memory_client),
        },
        "get_or_create_working_memory": {
            "name": "get_or_create_working_memory",
            "description": "Get the current working memory state including recent messages, temporarily stored memories, and session-specific data. Creates a new session if one doesn't exist. Use this to check what's already in the current conversation context.",
            "func": _create_get_working_memory_func(
                memory_client, session_id, namespace, user_id
            ),
        },
        "add_memory_to_working_memory": {
            "name": "add_memory_to_working_memory",
            "description": "Store new important information as a structured memory. Use this when users share preferences, facts, or important details that should be remembered for future conversations. The system automatically promotes important memories to long-term storage.",
            "func": _create_add_memory_func(
                memory_client, session_id, namespace, user_id
            ),
        },
        "update_working_memory_data": {
            "name": "update_working_memory_data",
            "description": "Store or update structured session data (JSON objects) in working memory. Use this for complex session-specific information that needs to be accessed and modified during the conversation.",
            "func": _create_update_memory_data_func(
                memory_client, session_id, namespace, user_id
            ),
        },
        "get_long_term_memory": {
            "name": "get_long_term_memory",
            "description": "Retrieve a specific long-term memory by its unique ID to see full details. Use this when you have a memory ID from search_memory results and need complete information.",
            "func": _create_get_long_term_memory_func(memory_client),
        },
        "create_long_term_memory": {
            "name": "create_long_term_memory",
            "description": "Create long-term memories directly for immediate storage and retrieval. Use this for important information that should be permanently stored without going through working memory.",
            "func": _create_create_long_term_memory_func(
                memory_client, namespace, user_id
            ),
        },
        "edit_long_term_memory": {
            "name": "edit_long_term_memory",
            "description": "Update an existing long-term memory with new or corrected information. Use this when users provide corrections, updates, or additional details. First call search_memory to get the memory ID.",
            "func": _create_edit_long_term_memory_func(memory_client),
        },
        "delete_long_term_memories": {
            "name": "delete_long_term_memories",
            "description": "Permanently delete long-term memories that are outdated, incorrect, or no longer needed. First call search_memory to get the memory IDs. This action cannot be undone.",
            "func": _create_delete_long_term_memories_func(memory_client),
        },
        "get_current_datetime": {
            "name": "get_current_datetime",
            "description": "Return the current datetime in UTC to ground relative time expressions. Use this before setting event_date or including a human-readable date in text when the user says 'today', 'yesterday', 'last week', etc.",
            "func": _create_get_current_datetime_func(memory_client),
        },
    }

    # Determine which tools to include
    if tools == "all":
        selected_tools = list(tool_configs.keys())
    else:
        selected_tools = list(tools)
        # Validate tool names
        invalid_tools = set(selected_tools) - set(tool_configs.keys())
        if invalid_tools:
            raise ValueError(
                f"Invalid tool names: {invalid_tools}. "
                f"Available tools: {list(tool_configs.keys())}"
            )

    # Create LangChain tools
    langchain_tools = []
    for tool_name in selected_tools:
        config = tool_configs[tool_name]

        # Use StructuredTool.from_function to create the tool
        # The function's type hints will automatically generate the args_schema
        langchain_tool = StructuredTool.from_function(
            func=config["func"],
            name=config["name"],
            description=config["description"],
            coroutine=config["func"],  # All our functions are async
        )
        langchain_tools.append(langchain_tool)

    return langchain_tools


# Alias for clarity
get_memory_tools_langchain = get_memory_tools


# === Tool Function Factories ===
# These create the actual async functions that LangChain will call


def _create_search_memory_func(client: MemoryAPIClient) -> Any:
    """Create search_memory function."""

    async def search_memory(
        query: str,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        memory_type: str | None = None,
        max_results: int = 10,
        min_relevance: float | None = None,
        user_id: str | None = None,
    ) -> str:
        """Search long-term memory for relevant information."""
        result = await client.search_memory_tool(
            query=query,
            topics=topics,
            entities=entities,
            memory_type=memory_type,
            max_results=max_results,
            min_relevance=min_relevance,
            user_id=user_id,
        )
        return str(result.get("summary", str(result)))

    return search_memory


def _create_get_working_memory_func(
    client: MemoryAPIClient,
    session_id: str,
    namespace: str | None,
    user_id: str | None,
) -> Any:
    """Create get_or_create_working_memory function."""

    async def get_or_create_working_memory() -> str:
        """Get the current working memory state."""
        result = await client.get_or_create_working_memory_tool(
            session_id=session_id,
            namespace=namespace,
            user_id=user_id,
        )
        return str(result.get("summary", str(result)))

    return get_or_create_working_memory


def _create_add_memory_func(
    client: MemoryAPIClient,
    session_id: str,
    namespace: str | None,
    user_id: str | None,
) -> Any:
    """Create add_memory_to_working_memory function."""

    async def add_memory_to_working_memory(
        text: str,
        memory_type: Literal["episodic", "semantic"],
        topics: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> str:
        """Store new important information as a structured memory."""
        result = await client.add_memory_tool(
            session_id=session_id,
            text=text,
            memory_type=memory_type,
            topics=topics,
            entities=entities,
            namespace=namespace,
            user_id=user_id,
        )
        return str(result.get("summary", str(result)))

    return add_memory_to_working_memory


def _create_update_memory_data_func(
    client: MemoryAPIClient,
    session_id: str,
    namespace: str | None,
    user_id: str | None,
) -> Any:
    """Create update_working_memory_data function."""

    async def update_working_memory_data(
        data: dict[str, Any],
        merge_strategy: Literal["replace", "merge", "deep_merge"] = "merge",
    ) -> str:
        """Store or update structured session data in working memory."""
        result = await client.update_memory_data_tool(
            session_id=session_id,
            data=data,
            merge_strategy=merge_strategy,
            namespace=namespace,
            user_id=user_id,
        )
        return str(result.get("summary", str(result)))

    return update_working_memory_data


def _create_get_long_term_memory_func(client: MemoryAPIClient) -> Any:
    """Create get_long_term_memory function."""

    async def get_long_term_memory(memory_id: str) -> str:
        """Retrieve a specific long-term memory by its unique ID."""
        result = await client.resolve_function_call(
            function_name="get_long_term_memory",
            function_arguments={"memory_id": memory_id},
            session_id="",  # Not needed for long-term memory retrieval
        )
        if result["success"]:
            return str(result["formatted_response"])
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return get_long_term_memory


def _create_create_long_term_memory_func(
    client: MemoryAPIClient,
    namespace: str | None,
    user_id: str | None,
) -> Any:
    """Create create_long_term_memory function."""

    async def create_long_term_memory(memories: list[dict[str, Any]]) -> str:
        """Create long-term memories directly for immediate storage."""
        result = await client.resolve_function_call(
            function_name="create_long_term_memory",
            function_arguments={"memories": memories},
            session_id="",  # Not needed for direct long-term memory creation
            namespace=namespace,
            user_id=user_id,
        )
        if result["success"]:
            return str(result["formatted_response"])
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return create_long_term_memory


def _create_edit_long_term_memory_func(client: MemoryAPIClient) -> Any:
    """Create edit_long_term_memory function."""

    async def edit_long_term_memory(
        memory_id: str,
        text: str | None = None,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        memory_type: Literal["episodic", "semantic"] | None = None,
        event_date: str | None = None,
    ) -> str:
        """Update an existing long-term memory with new or corrected information."""
        # Build update dict with only provided fields
        updates: dict[str, Any] = {"memory_id": memory_id}
        if text is not None:
            updates["text"] = text
        if topics is not None:
            updates["topics"] = topics
        if entities is not None:
            updates["entities"] = entities
        if memory_type is not None:
            updates["memory_type"] = memory_type
        if event_date is not None:
            updates["event_date"] = event_date

        result = await client.resolve_function_call(
            function_name="edit_long_term_memory",
            function_arguments=updates,
            session_id="",  # Not needed for long-term memory editing
        )
        if result["success"]:
            return str(result["formatted_response"])
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return edit_long_term_memory


def _create_delete_long_term_memories_func(client: MemoryAPIClient) -> Any:
    """Create delete_long_term_memories function."""

    async def delete_long_term_memories(memory_ids: list[str]) -> str:
        """Permanently delete long-term memories."""
        result = await client.resolve_function_call(
            function_name="delete_long_term_memories",
            function_arguments={"memory_ids": memory_ids},
            session_id="",  # Not needed for long-term memory deletion
        )
        if result["success"]:
            return str(result["formatted_response"])
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return delete_long_term_memories


def _create_get_current_datetime_func(client: MemoryAPIClient) -> Any:
    """Create get_current_datetime function."""

    async def get_current_datetime() -> str:
        """Return the current datetime in UTC."""
        result = await client.resolve_function_call(
            function_name="get_current_datetime",
            function_arguments={},
            session_id="",  # Not needed for datetime
        )
        if result["success"]:
            return str(result["formatted_response"])
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return get_current_datetime
