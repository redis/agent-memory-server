import asyncio

from agent_memory_server.client.api import MemoryAPIClient, MemoryClientConfig
from agent_memory_server.models import (
    LongTermMemory,
    MemoryMessage,
    SessionMemory,
)


async def example():
    # Create a client
    base_url = "http://localhost:8000"  # Adjust to your server URL

    # Using context manager for automatic cleanup
    async with MemoryAPIClient(
        MemoryClientConfig(base_url=base_url, default_namespace="example-namespace")
    ) as client:
        # Check server health
        health = await client.health_check()
        print(f"Server is healthy, current time: {health.now}")

        # Store a conversation
        session_id = "example-session"
        memory = SessionMemory(
            messages=[
                MemoryMessage(role="user", content="What is the weather like today?"),
                MemoryMessage(role="assistant", content="It's sunny and warm!"),
            ]
        )
        await client.put_session_memory(session_id, memory)
        print(f"Stored conversation in session {session_id}")

        # Retrieve the conversation
        session = await client.get_session_memory(session_id)
        print(f"Retrieved {len(session.messages)} messages from session {session_id}")

        for message in session.messages:
            print(f"- {message.role}: {message.content}")

        # Create long-term memory
        memories = [
            LongTermMemory(
                text="User lives in San Francisco",
                topics=["location", "personal_info"],
            ),
        ]
        await client.create_long_term_memory(memories)
        print("Created long-term memory")

        # Search for relevant memories
        results = await client.search_long_term_memory(
            text="Where does the user live?",
            limit=5,
        )
        print(f"Found {results.total} relevant memories")
        for memory in results.memories:
            print(f"- {memory.text} (relevance: {1.0 - memory.dist:.2f})")

        # Clean up
        await client.delete_session_memory(session_id)
        print(f"Deleted session {session_id}")


# Run the example
asyncio.run(example())
