# LangChain Integration

The Python SDK (agent-memory-client) provides a LangChain integration that helps you use the memory server with LangChain applications. This integration automatically converts memory operations into LangChain-compatible tools.

## Memory Tools for LangChain

The SDK provides a `get_memory_tools()` function that returns a list of LangChain `StructuredTool` instances. These tools give your LangChain LLMs and agents access to the memory server's capabilities.

For details on available memory operations, see the [Tool Methods](python-sdk.md#tool-methods) section of the Python SDK documentation.

### Direct LLM Integration

You can bind memory tools directly to a LangChain LLM:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool


async def main():
    # Initialize the memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # Get memory tools as LangChain StructuredTool instances
    tools: list[StructuredTool] = get_memory_tools(
        memory_client=memory_client,
        session_id="user_session_123",
        user_id="alice"
    )

    # Bind tools to an LLM
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    # Use the LLM with memory capabilities
    response = await llm_with_tools.ainvoke(
        "Remember that I prefer morning meetings and I work remotely"
    )
    print(response)


asyncio.run(main())
```

The LLM can now automatically use memory tools to store and retrieve information during conversations.

## Installation

Install the Python SDK with LangChain support:

```bash
pip install agent-memory-client langchain-core
```

For LangChain agents and LangGraph:

```bash
pip install agent-memory-client langchain langchain-openai langgraph
```

## Using with LangChain

Here's a complete example of creating a memory-enabled LangChain agent:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI


async def main():
    # Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # Get memory tools
    tools: list[StructuredTool] = get_memory_tools(
        memory_client=memory_client,
        session_id="my_session",
        user_id="alice"
    )

    # Create LangChain agent
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with persistent memory."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    # Use the agent
    result = await executor.ainvoke({
        "input": "Remember that I love pizza and work at TechCorp"
    })
    print(result["output"])

    # Later conversation - agent can recall the information
    result = await executor.ainvoke({
        "input": "What do you know about my food preferences?"
    })
    print(result["output"])


asyncio.run(main())
```

## Using with LangGraph

You can use memory tools in LangGraph workflows:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


async def main():
    # Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # Get memory tools
    tools: list[StructuredTool] = get_memory_tools(
        memory_client=memory_client,
        session_id="langgraph_session",
        user_id="alice"
    )

    # Create a LangGraph agent with memory tools
    llm = ChatOpenAI(model="gpt-4o")
    graph = create_react_agent(llm, tools)

    # Use the agent
    result = await graph.ainvoke({
        "messages": [("user", "Remember that I'm learning Python and prefer visual examples")]
    })
    print(result["messages"][-1].content)

    # Continue the conversation
    result = await graph.ainvoke({
        "messages": [("user", "What programming language am I learning?")]
    })
    print(result["messages"][-1].content)


asyncio.run(main())
```

## Advanced Usage

### Selective Tools

Get only specific tools you need:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain_core.tools import StructuredTool


async def main():
    # Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    tools: list[StructuredTool] = get_memory_tools(
        memory_client=memory_client,
        session_id="chat_session",
        user_id="alice",
        tools=["search_memory", "eagerly_create_long_term_memory"]
    )


asyncio.run(main())
```

### Combining with Custom Tools

Combine memory tools with your own custom tools:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# Define custom tools
@tool
async def calculate(expression: str) -> str:
    """Evaluate a simple mathematical expression safely."""
    import ast
    # Use ast.literal_eval for safe evaluation of simple expressions
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except (ValueError, SyntaxError):
        return "Error: Invalid expression"


@tool
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Your weather API logic here
    return f"Weather in {city}: Sunny, 72°F"


async def main():
    # Initialize memory client
    memory_client = await create_memory_client("http://localhost:8000")

    # Get memory tools
    memory_tools = get_memory_tools(
        memory_client=memory_client,
        session_id="session",
        user_id="user"
    )

    # Combine all tools
    all_tools = memory_tools + [calculate, get_weather]

    # Create agent with combined tools
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory and additional capabilities."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools)

    # Use the agent
    result = await executor.ainvoke({
        "input": "What's 2+2? Also remember that I like math."
    })
    print(result["output"])


asyncio.run(main())
```

### Multi-User Application

Handle multiple users with different sessions:

```python
import asyncio
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


async def main():
    # Initialize shared memory client
    shared_memory_client = await create_memory_client("http://localhost:8000")

    async def create_user_agent(user_id: str, session_id: str):
        """Create a memory-enabled agent for a specific user."""
        tools: list[StructuredTool] = get_memory_tools(
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


asyncio.run(main())
```

## See Also

- [Python SDK Documentation](python-sdk.md) - Complete SDK reference and tool methods
- [Memory Integration Patterns](memory-integration-patterns.md) - Overview of different integration approaches
- [LangChain Integration Example](https://github.com/redis/agent-memory-server/blob/main/examples/langchain_integration_example.py) - Complete working example
