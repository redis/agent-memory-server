"""Test that create_long_term_memory accepts both single object and array."""

import asyncio
from unittest.mock import AsyncMock

from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.integrations.langchain import get_memory_tools


async def main():
    # Create mock client
    config = MemoryClientConfig(base_url="http://localhost:8000")
    client = MemoryAPIClient(config)

    # Mock the resolve_function_call method
    client.resolve_function_call = AsyncMock(
        return_value={
            "success": True,
            "formatted_response": "Created 1 memory successfully",
        }
    )

    # Get the tool
    tools = get_memory_tools(
        memory_client=client,
        session_id="test",
        user_id="alice",
        tools=["create_long_term_memory"],
    )

    create_tool = tools[0]

    # Test 1: Single memory object (what LLMs often do)
    print("Test 1: Single memory object")
    single_memory = {
        "text": "User loves pizza",
        "memory_type": "semantic",
        "topics": ["food", "preferences"],
    }

    result = await create_tool.func(memories=single_memory)
    print(f"Result: {result}")

    # Check what was actually called
    call_args = client.resolve_function_call.call_args
    print(f"Called with: {call_args}")
    memories_arg = call_args.kwargs["function_arguments"]["memories"]
    print(f"Memories argument type: {type(memories_arg)}")
    print(f"Memories argument value: {memories_arg}")
    assert isinstance(memories_arg, list), "Should convert single object to list"
    assert len(memories_arg) == 1, "Should have one memory"
    print("✅ Single object correctly converted to array\n")

    # Test 2: Array of memories (correct format)
    print("Test 2: Array of memories")
    client.resolve_function_call.reset_mock()

    memory_array = [
        {"text": "User loves pizza", "memory_type": "semantic", "topics": ["food"]},
        {
            "text": "User works at TechCorp",
            "memory_type": "semantic",
            "topics": ["work"],
        },
    ]

    result = await create_tool.func(memories=memory_array)
    print(f"Result: {result}")

    call_args = client.resolve_function_call.call_args
    memories_arg = call_args.kwargs["function_arguments"]["memories"]
    print(f"Memories argument type: {type(memories_arg)}")
    print(f"Memories argument value: {memories_arg}")
    assert isinstance(memories_arg, list), "Should keep as list"
    assert len(memories_arg) == 2, "Should have two memories"
    print("✅ Array correctly passed through\n")

    print("All tests passed! ✅")


if __name__ == "__main__":
    asyncio.run(main())
