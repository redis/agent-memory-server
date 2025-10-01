#!/usr/bin/env python3
"""
LangChain Integration Example

This example demonstrates how to use the agent-memory-client LangChain integration
to create memory-enabled agents WITHOUT manual tool wrapping.

Before (manual wrapping):
    @tool
    async def search_memory(text: str) -> str:
        result = await memory_client.resolve_function_call(...)
        return str(result)

After (automatic integration):
    tools = get_memory_tools(memory_client, session_id, user_id)

Environment variables:
- MEMORY_SERVER_URL (default: http://localhost:8000)
- OPENAI_API_KEY (required for this example)
"""

from __future__ import annotations

import asyncio
import os

# Import memory client
from agent_memory_client import create_memory_client

# Import LangChain integration (no manual wrapping needed!)
from agent_memory_client.integrations.langchain import get_memory_tools
from dotenv import load_dotenv

# Import LangChain components
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


load_dotenv()

MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")


async def main():
    """Run the LangChain integration example."""

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print("🚀 LangChain Integration Example")
    print("=" * 60)

    # 1. Initialize memory client
    print("\n1️⃣  Initializing memory client...")
    memory_client = await create_memory_client(base_url=MEMORY_SERVER_URL)
    print(f"   ✅ Connected to memory server at {MEMORY_SERVER_URL}")

    # 2. Get LangChain-compatible tools (NO MANUAL WRAPPING!)
    print("\n2️⃣  Creating LangChain tools (automatic conversion)...")
    session_id = "langchain_demo"
    user_id = "demo_user"

    tools = get_memory_tools(
        memory_client=memory_client,
        session_id=session_id,
        user_id=user_id,
    )
    print(f"   ✅ Created {len(tools)} LangChain tools:")
    for tool in tools:
        print(f"      - {tool.name}")

    # 3. Create LangChain agent with memory tools
    print("\n3️⃣  Creating LangChain agent...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with persistent memory. "
                "Use the memory tools to remember important information and recall past conversations. "
                "When users share preferences or important facts, store them using add_memory_to_working_memory. "
                "When you need to recall information, use search_memory.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("   ✅ Agent created with memory capabilities")

    # 4. Run example conversations
    print("\n4️⃣  Running example conversations...")
    print("-" * 60)

    # Conversation 1: Store preferences
    print(
        "\n💬 User: Hi! I'm Alice and I love Italian food, especially pasta carbonara."
    )
    result1 = await executor.ainvoke(
        {"input": "Hi! I'm Alice and I love Italian food, especially pasta carbonara."}
    )
    print(f"🤖 Assistant: {result1['output']}")

    # Conversation 2: Store more information
    print("\n💬 User: I also work as a software engineer at TechCorp.")
    result2 = await executor.ainvoke(
        {"input": "I also work as a software engineer at TechCorp."}
    )
    print(f"🤖 Assistant: {result2['output']}")

    # Conversation 3: Recall information
    print("\n💬 User: What do you know about my food preferences?")
    result3 = await executor.ainvoke(
        {"input": "What do you know about my food preferences?"}
    )
    print(f"🤖 Assistant: {result3['output']}")

    # Conversation 4: Recall work information
    print("\n💬 User: Where do I work?")
    result4 = await executor.ainvoke({"input": "Where do I work?"})
    print(f"🤖 Assistant: {result4['output']}")

    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("\n💡 Key Benefits:")
    print("   • No manual @tool decorator wrapping needed")
    print("   • Automatic type conversion and validation")
    print("   • Session and user context automatically injected")
    print("   • Works seamlessly with LangChain agents")

    # Cleanup
    await memory_client.close()


async def selective_tools_example():
    """Example showing how to use only specific memory tools."""

    print("\n🎯 Selective Tools Example")
    print("=" * 60)

    memory_client = await create_memory_client(base_url=MEMORY_SERVER_URL)

    # Get only specific tools instead of all tools
    tools = get_memory_tools(
        memory_client=memory_client,
        session_id="selective_demo",
        user_id="demo_user",
        tools=["search_memory", "create_long_term_memory"],  # Only these two
    )

    print(f"✅ Created {len(tools)} selected tools:")
    for tool in tools:
        print(f"   - {tool.name}")

    await memory_client.close()


async def custom_agent_example():
    """Example showing integration with custom LangChain workflows."""

    print("\n🔧 Custom Workflow Example")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping (OPENAI_API_KEY not set)")
        return

    memory_client = await create_memory_client(base_url=MEMORY_SERVER_URL)

    # Get memory tools
    memory_tools = get_memory_tools(
        memory_client=memory_client,
        session_id="custom_demo",
        user_id="demo_user",
    )

    # You can combine memory tools with your own custom tools
    from langchain_core.tools import tool

    @tool
    async def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)  # Note: Use safely in production!
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    # Combine memory tools with custom tools
    all_tools = memory_tools + [calculate]

    print(f"✅ Created {len(all_tools)} total tools:")
    print(f"   - {len(memory_tools)} memory tools")
    print("   - 1 custom tool (calculate)")

    # Use with your agent...
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with memory and calculation abilities.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False)

    # Test it
    result = await executor.ainvoke(
        {
            "input": "Calculate 42 * 137 and remember that this is my favorite calculation."
        }
    )
    print(
        "\n💬 User: Calculate 42 * 137 and remember that this is my favorite calculation."
    )
    print(f"🤖 Assistant: {result['output']}")

    await memory_client.close()


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment to run additional examples:
    # asyncio.run(selective_tools_example())
    # asyncio.run(custom_agent_example())
