#!/usr/bin/env python3
"""
Demonstration of the recent_messages_limit feature.

This script shows how to use the new recent_messages_limit parameter
to efficiently retrieve only the most recent N messages from working memory.
"""

import asyncio

from agent_memory_server.models import MemoryMessage, WorkingMemory
from agent_memory_server.utils.redis import get_redis_conn
from agent_memory_server.working_memory import get_working_memory, set_working_memory


async def demo_recent_messages_limit():
    """Demonstrate the recent_messages_limit functionality"""
    print("🚀 Recent Messages Limit Demo")
    print("=" * 50)

    # Get Redis connection
    redis_client = await get_redis_conn()

    # Create a session with many messages
    session_id = "demo-session"
    user_id = "demo-user"
    namespace = "demo"

    print("📝 Creating working memory with 10 messages...")

    # Create 10 messages with automatic created_at timestamps
    messages = []
    for i in range(10):
        messages.append(
            MemoryMessage(
                id=f"msg-{i}",
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is message number {i}. It contains some conversation content.",
                # created_at is automatically set to current time
            )
        )

    # Create working memory
    working_memory = WorkingMemory(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        messages=messages,
        context="This is a demo conversation",
        data={"demo": True, "total_messages": 10},
    )

    # Store the working memory
    await set_working_memory(working_memory, redis_client=redis_client)
    print(f"✅ Stored working memory with {len(messages)} messages")

    print("\n" + "=" * 50)
    print("🔍 Testing different message limits:")
    print("=" * 50)

    # Test 1: Get all messages (no limit)
    print("\n1️⃣ Getting ALL messages (no limit):")
    result = await get_working_memory(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        redis_client=redis_client,
    )
    print(f"   📊 Retrieved {len(result.messages)} messages")
    print(f"   📝 First message: {result.messages[0].content}")
    print(f"   📝 Last message: {result.messages[-1].content}")

    # Test 2: Get last 3 messages
    print("\n2️⃣ Getting last 3 messages:")
    result = await get_working_memory(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        redis_client=redis_client,
        recent_messages_limit=3,
    )
    print(f"   📊 Retrieved {len(result.messages)} messages")
    for i, msg in enumerate(result.messages):
        print(f"   📝 Message {i}: {msg.content}")

    # Test 3: Get last 5 messages
    print("\n3️⃣ Getting last 5 messages:")
    result = await get_working_memory(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        redis_client=redis_client,
        recent_messages_limit=5,
    )
    print(f"   📊 Retrieved {len(result.messages)} messages")
    print(f"   📝 First of limited: {result.messages[0].content}")
    print(f"   📝 Last of limited: {result.messages[-1].content}")

    # Test 4: Get more messages than available
    print("\n4️⃣ Getting 20 messages (more than available):")
    result = await get_working_memory(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        redis_client=redis_client,
        recent_messages_limit=20,
    )
    print(f"   📊 Retrieved {len(result.messages)} messages (all available)")

    # Test 5: Verify other data is preserved
    print("\n5️⃣ Verifying other data is preserved:")
    result = await get_working_memory(
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        redis_client=redis_client,
        recent_messages_limit=2,
    )
    print(f"   📊 Retrieved {len(result.messages)} messages")
    print(f"   🗂️ Context preserved: {result.context}")
    print(f"   🗂️ Data preserved: {result.data}")
    print(f"   🗂️ Session ID: {result.session_id}")

    print("\n" + "=" * 50)
    print("🎯 Key Benefits:")
    print("=" * 50)
    print("✨ Efficient: Limits messages returned to client applications")
    print("✨ Chronological: Uses created_at timestamps for proper message ordering")
    print("✨ Simple: Uses in-memory slicing for working memory data")
    print("✨ Flexible: Works with both working memory and long-term reconstruction")
    print("✨ Safe: Preserves all other working memory data")
    print("✨ Compatible: Available in both REST API and MCP server")

    print("\n" + "=" * 50)
    print("📚 Usage Examples:")
    print("=" * 50)
    print("🌐 REST API:")
    print("   GET /v1/working-memory/{session_id}?recent_messages_limit=5")
    print("\n🔧 MCP Tool:")
    print("   get_working_memory(session_id='...', recent_messages_limit=5)")
    print("\n🐍 Python:")
    print("   await get_working_memory(..., recent_messages_limit=5)")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_recent_messages_limit())
