#!/usr/bin/env python3
"""
Demonstration of the recent_messages_limit feature.

This script shows how to use the recent_messages_limit parameter
to efficiently retrieve only the most recent N messages from working memory.

Prerequisites:
    - Redis running: docker-compose up redis -d
    - API server running: DISABLE_AUTH=true uv run agent-memory api --port 8000
"""

import asyncio
import os

import httpx
from agent_memory_client import create_memory_client
from agent_memory_client.models import MemoryMessage, WorkingMemory


BASE_URL = os.getenv("AGENT_MEMORY_URL", "http://localhost:8000")


async def demo_recent_messages_limit():
    """Demonstrate the recent_messages_limit functionality."""
    print("Recent Messages Limit Demo\n" + "=" * 50)

    session_id = "demo-recent-limit-session"
    namespace = "demo"

    # Create a client for storing working memory
    client = await create_memory_client(base_url=BASE_URL)

    print("Creating working memory with 10 messages...")

    # Create 10 messages
    messages = []
    for i in range(10):
        messages.append(
            MemoryMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is message number {i}. It contains some conversation content.",
            )
        )

    # Store working memory using the client SDK
    working_memory = WorkingMemory(
        session_id=session_id,
        messages=messages,
        namespace=namespace,
        context="This is a demo conversation",
        data={"demo": True, "total_messages": 10},
    )
    await client.put_working_memory(session_id, working_memory)
    print(f"Stored working memory with {len(messages)} messages")

    # Use httpx directly to test recent_messages_limit
    # (this parameter is not yet exposed in the client SDK's high-level methods)
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        # Test 1: Get all messages (no limit)
        print("\n1. Getting ALL messages (no limit):")
        resp = await http.get(
            f"/v1/working-memory/{session_id}",
            params={"namespace": namespace},
        )
        resp.raise_for_status()
        result = resp.json()
        msgs = result["messages"]
        print(f"   Retrieved {len(msgs)} messages")

        # Test 2: Get last 3 messages
        print("\n2. Getting last 3 messages:")
        resp = await http.get(
            f"/v1/working-memory/{session_id}",
            params={"namespace": namespace, "recent_messages_limit": 3},
        )
        resp.raise_for_status()
        result = resp.json()
        msgs = result["messages"]
        print(f"   Retrieved {len(msgs)} messages")
        for i, msg in enumerate(msgs):
            print(f"   Message {i}: {msg['content']}")

        # Test 3: Get last 5 messages
        print("\n3. Getting last 5 messages:")
        resp = await http.get(
            f"/v1/working-memory/{session_id}",
            params={"namespace": namespace, "recent_messages_limit": 5},
        )
        resp.raise_for_status()
        result = resp.json()
        msgs = result["messages"]
        print(f"   Retrieved {len(msgs)} messages")

        # Test 4: Get more messages than available
        print("\n4. Getting 20 messages (more than available):")
        resp = await http.get(
            f"/v1/working-memory/{session_id}",
            params={"namespace": namespace, "recent_messages_limit": 20},
        )
        resp.raise_for_status()
        result = resp.json()
        msgs = result["messages"]
        print(f"   Retrieved {len(msgs)} messages (all available)")

        # Test 5: Verify other data is preserved
        print("\n5. Verifying other data is preserved:")
        resp = await http.get(
            f"/v1/working-memory/{session_id}",
            params={"namespace": namespace, "recent_messages_limit": 2},
        )
        resp.raise_for_status()
        result = resp.json()
        msgs = result["messages"]
        print(f"""   Retrieved {len(msgs)} messages
   Context preserved: {result.get("context")}
   Data preserved: {result.get("data")}""")

    # Summary of usage
    print(f"""
{"=" * 50}
Usage Examples:
{"=" * 50}
REST API:
   GET /v1/working-memory/{{session_id}}?recent_messages_limit=5

MCP Tool:
   get_working_memory(session_id='...', recent_messages_limit=5)

Note: The Python SDK does not yet expose recent_messages_limit in its
high-level methods. Use the REST API directly via httpx as shown above,
or use the MCP server interface.""")

    await client.close()
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_recent_messages_limit())
