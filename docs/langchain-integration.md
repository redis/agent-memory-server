# LangChain Integration

The agent-memory-client provides seamless integration with LangChain, eliminating the need for manual tool wrapping. This integration automatically converts memory client tools into LangChain-compatible `StructuredTool` instances.

## Why Use This Integration?

### Before (Manual Wrapping) ❌

Users had to manually wrap every memory tool with LangChain's `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
async def create_long_term_memory(memories: List[dict]) -> str:
    """Store important information in long-term memory."""
    result = await memory_client.resolve_function_call(
        function_name="create_long_term_memory",
        args={"memories": memories},
        session_id=session_id,
        user_id=student_id
    )
    return f"✅ Stored {len(memories)} memory(ies): {result}"

@tool
async def search_long_term_memory(text: str, limit: int = 5) -> str:
    """Search for relevant memories using semantic search."""
    result = await memory_client.resolve_function_call(
        function_name="search_long_term_memory",
        args={"text": text, "limit": limit},
        session_id=session_id,
        user_id=student_id
    )
    return str(result)

# ... repeat for every tool you want to use
```

**Problems:**
- Tedious boilerplate code
- Error-prone (easy to forget session_id, user_id, etc.)
- Hard to maintain
- Duplicates logic across projects

### After (Automatic Integration) ✅

With the LangChain integration, you get all tools with one function call:

```python
from agent_memory_client.integrations.langchain import get_memory_tools

tools = get_memory_tools(
    memory_client=memory_client,
    session_id=session_id,
    user_id=user_id
)

# That's it! All tools are ready to use with LangChain agents
```

**Benefits:**
- ✅ No manual wrapping needed
- ✅ Automatic type conversion and validation
- ✅ Session and user context automatically injected
- ✅ Works seamlessly with LangChain agents
- ✅ Consistent behavior across all tools

## Installation

The LangChain integration requires `langchain-core`:

```bash
pip install agent-memory-client langchain-core
```

For the full LangChain experience with agents:

```bash
pip install agent-memory-client langchain langchain-openai
```

## Quick Start

Here's a complete example of creating a memory-enabled LangChain agent:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

async def main():
    # 1. Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # 2. Get LangChain-compatible tools (automatic conversion!)
    tools = get_memory_tools(
        memory_client=memory_client,
        session_id="my_session",
        user_id="alice"
    )

    # 3. Create LangChain agent
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with persistent memory."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    # 4. Use the agent
    result = await executor.ainvoke({
        "input": "Remember that I love pizza and work at TechCorp"
    })
    print(result["output"])

    # Later conversation - agent can recall the information
    result = await executor.ainvoke({
        "input": "What do you know about my food preferences?"
    })
    print(result["output"])

    await memory_client.close()

asyncio.run(main())
```

## API Reference

### `get_memory_tools()`

Convert memory client tools to LangChain-compatible tools.

```python
def get_memory_tools(
    memory_client: MemoryAPIClient,
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    tools: Sequence[str] | Literal["all"] = "all",
) -> list[StructuredTool]:
```

**Parameters:**

- `memory_client` (MemoryAPIClient): Initialized memory client instance
- `session_id` (str): Session ID for working memory operations
- `user_id` (str | None): Optional user ID for memory operations
- `namespace` (str | None): Optional namespace for memory operations
- `tools` (Sequence[str] | "all"): Which tools to include (default: "all")

**Returns:**

List of LangChain `StructuredTool` instances ready to use with agents.

**Available Tools:**

- `search_memory` - Search long-term memory using semantic search
- `get_or_create_working_memory` - Get current working memory state
- `add_memory_to_working_memory` - Store new structured memories
- `update_working_memory_data` - Update session data
- `get_long_term_memory` - Retrieve specific memory by ID
- `create_long_term_memory` - Create long-term memories directly
- `edit_long_term_memory` - Update existing memories
- `delete_long_term_memories` - Delete memories permanently
- `get_current_datetime` - Get current UTC datetime

## Usage Examples

### Example 1: All Memory Tools

Get all available memory tools:

```python
from agent_memory_client.integrations.langchain import get_memory_tools

tools = get_memory_tools(
    memory_client=client,
    session_id="chat_session",
    user_id="alice"
)

# Returns all 9 memory tools
print(f"Created {len(tools)} tools")
```

### Example 2: Selective Tools

Get only specific tools you need:

```python
tools = get_memory_tools(
    memory_client=client,
    session_id="chat_session",
    user_id="alice",
    tools=["search_memory", "create_long_term_memory"]
)

# Returns only the 2 specified tools
```

### Example 3: Combining with Custom Tools

Combine memory tools with your own custom tools:

```python
from langchain_core.tools import tool
from agent_memory_client.integrations.langchain import get_memory_tools

# Get memory tools
memory_tools = get_memory_tools(
    memory_client=client,
    session_id="session",
    user_id="user"
)

# Define custom tools
@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Your weather API logic here
    return f"Weather in {city}: Sunny, 72°F"

# Combine all tools
all_tools = memory_tools + [calculate, get_weather]

# Use with agent
agent = create_tool_calling_agent(llm, all_tools, prompt)
executor = AgentExecutor(agent=agent, tools=all_tools)
```

### Example 4: Multi-User Application

Handle multiple users with different sessions:

```python
async def create_user_agent(user_id: str, session_id: str):
    """Create a memory-enabled agent for a specific user."""

    tools = get_memory_tools(
        memory_client=shared_memory_client,
        session_id=session_id,
        user_id=user_id,
        namespace=f"app:{user_id}"  # User-specific namespace
    )

    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are assisting user {user_id}."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

# Create agents for different users
alice_agent = await create_user_agent("alice", "alice_session_1")
bob_agent = await create_user_agent("bob", "bob_session_1")

# Each agent has isolated memory
await alice_agent.ainvoke({"input": "I love pizza"})
await bob_agent.ainvoke({"input": "I love sushi"})
```

## Advanced Usage

### Custom Tool Selection

Choose exactly which memory capabilities your agent needs:

```python
# Minimal agent - only search and create
minimal_tools = get_memory_tools(
    memory_client=client,
    session_id="minimal",
    user_id="user",
    tools=["search_memory", "create_long_term_memory"]
)

# Read-only agent - only search
readonly_tools = get_memory_tools(
    memory_client=client,
    session_id="readonly",
    user_id="user",
    tools=["search_memory", "get_long_term_memory"]
)

# Full control agent - all tools
full_tools = get_memory_tools(
    memory_client=client,
    session_id="full",
    user_id="user",
    tools="all"
)
```

### Error Handling

The integration handles errors gracefully:

```python
try:
    tools = get_memory_tools(
        memory_client=client,
        session_id="session",
        user_id="user",
        tools=["invalid_tool_name"]  # This will raise ValueError
    )
except ValueError as e:
    print(f"Invalid tool selection: {e}")
```

## Comparison with Direct SDK Usage

| Feature | Direct SDK | LangChain Integration |
|---------|-----------|----------------------|
| Setup complexity | Low | Very Low |
| Tool wrapping | Manual | Automatic |
| Type safety | Manual | Automatic |
| Context injection | Manual | Automatic |
| Agent compatibility | Requires wrapping | Native |
| Code maintenance | High | Low |
| Best for | Custom workflows | LangChain agents |

## See Also

- [Memory Integration Patterns](memory-integration-patterns.md) - Overview of different integration approaches
- [Python SDK](python-sdk.md) - Direct SDK usage without LangChain
- [Agent Examples](agent-examples.md) - More agent implementation examples
- [LangChain Integration Example](https://github.com/redis/agent-memory-server/blob/main/examples/langchain_integration_example.py) - Complete working example
