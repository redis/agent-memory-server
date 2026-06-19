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
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver


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

    # 3. Create agent with memory tools
    print("\n3. Creating agent...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are a helpful assistant with persistent memory. "
        "ALWAYS use the memory tools proactively:\n"
        "- When users share preferences, facts, or personal details, IMMEDIATELY store them "
        "using lazily_create_long_term_memory before responding.\n"
        "- When users ask about past information, ALWAYS use search_memory first "
        "(supports semantic, keyword, and hybrid search modes).\n"
        "- Never rely on conversation context alone -- always store and retrieve via tools."
    )

    # Use a checkpointer so the agent remembers conversation across turns
    checkpointer = MemorySaver()
    agent = create_agent(
        llm, tools, system_prompt=system_prompt, checkpointer=checkpointer
    )
    print("   Agent created with memory capabilities")

    # Config with a thread_id so the checkpointer tracks this conversation
    config = {"configurable": {"thread_id": "langchain_demo_thread"}}

    # Helper to invoke the agent and extract the final assistant message
    async def ask(user_input: str) -> str:
        print(f"\nUser: {user_input}")
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        # The last message in the result is the assistant's final response
        reply = result["messages"][-1].content
        print(f"Assistant: {reply}")
        return reply

    # 4. Run example conversations
    print("\n4. Running example conversations...")
    print("-" * 60)

    await ask("Hi! I'm Alice and I love Italian food, especially pasta carbonara.")
    await ask("I also work as a software engineer at TechCorp.")
    await ask("What do you know about my food preferences?")
    await ask("Where do I work?")

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
        tools=["search_memory", "eagerly_create_long_term_memory"],  # Only these two
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
        """Evaluate a simple mathematical expression (numbers and basic operators only)."""
        import ast
        import operator

        # Safe evaluation using AST - only allows basic math operations
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }

        try:

            def eval_node(node: ast.AST) -> float:
                if isinstance(node, ast.Constant):
                    return float(node.value)
                if isinstance(node, ast.BinOp):
                    op = allowed_operators.get(type(node.op))
                    if op is None:
                        raise ValueError(f"Unsupported operator: {type(node.op)}")
                    return op(eval_node(node.left), eval_node(node.right))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                    return -eval_node(node.operand)
                raise ValueError(f"Unsupported expression: {type(node)}")

            tree = ast.parse(expression, mode="eval")
            result = eval_node(tree.body)
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
    system_prompt = "You are a helpful assistant with memory and calculation abilities."

    agent = create_agent(llm, all_tools, system_prompt=system_prompt)

    # Test it
    user_msg = "Calculate 42 * 137 and remember that this is my favorite calculation."
    print(f"\nUser: {user_msg}")
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_msg)]})
    print(f"Assistant: {result['messages'][-1].content}")

    await memory_client.close()


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment to run additional examples:
    # asyncio.run(selective_tools_example())
    # asyncio.run(custom_agent_example())
