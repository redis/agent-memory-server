"""
Test file for the Redis Memory Server API Client.

This file contains tests that demonstrate how to use the Memory API client.
"""

from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent_memory_server.api import router as memory_router
from agent_memory_server.client.api import MemoryAPIClient, MemoryClientConfig
from agent_memory_server.filters import Namespace
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.models import (
    LongTermMemory,
    LongTermMemoryResult,
    LongTermMemoryResultsResponse,
    MemoryMessage,
    SessionMemory,
    SessionMemoryResponse,
)


@pytest.fixture
def memory_app() -> FastAPI:
    """Create a test FastAPI app with memory routers for testing the client."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(memory_router)
    return app


@pytest.fixture
async def memory_test_client(
    memory_app: FastAPI,
) -> AsyncGenerator[MemoryAPIClient, None]:
    """Create a memory client that uses the test FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=memory_app),
        base_url="http://test",
    ) as http_client:
        # Create the memory client with our test http client
        config = MemoryClientConfig(
            base_url="http://test", default_namespace="test-namespace"
        )
        client = MemoryAPIClient(config)

        # Replace the internal http client with our test client
        client._client = http_client

        yield client


@pytest.mark.asyncio
async def test_health_check(memory_test_client: MemoryAPIClient):
    """Test the health check endpoint"""
    # Mock the response from the health endpoint
    response = await memory_test_client.health_check()
    assert response.now > 0


@pytest.mark.asyncio
async def test_session_lifecycle(memory_test_client: MemoryAPIClient):
    """Test the complete lifecycle of a session"""
    # For this test, we need to set up mocks for all the API calls
    session_id = "test-client-session"

    # Mock memory data
    memory = SessionMemory(
        messages=[
            MemoryMessage(role="user", content="Hello from the client!"),
            MemoryMessage(role="assistant", content="Hi there, I'm the memory server!"),
        ],
        context="This is a test session created by the API client.",
    )

    # First, mock PUT response for creating a session
    with patch("agent_memory_server.messages.set_session_memory") as mock_set_memory:
        mock_set_memory.return_value = None

        # Step 1: Create new session memory
        response = await memory_test_client.put_session_memory(session_id, memory)
        assert response.status == "ok"

    # Next, mock GET response for retrieving session memory
    with patch("agent_memory_server.messages.get_session_memory") as mock_get_memory:
        # Get memory data and explicitly exclude session_id to avoid duplicate parameter
        memory_data = memory.model_dump(exclude={"session_id"})
        mock_response = SessionMemoryResponse(**memory_data, session_id=session_id)
        mock_get_memory.return_value = mock_response

        # Step 2: Retrieve the session memory
        session = await memory_test_client.get_session_memory(session_id)
        assert len(session.messages) == 2
        assert session.messages[0].content == "Hello from the client!"
        assert session.messages[1].content == "Hi there, I'm the memory server!"
        assert session.context == "This is a test session created by the API client."

    # Mock list sessions
    with patch("agent_memory_server.messages.list_sessions") as mock_list_sessions:
        mock_list_sessions.return_value = (1, [session_id])

        # Step 3: List sessions and verify our test session is included
        sessions = await memory_test_client.list_sessions()
        assert session_id in sessions.sessions

    # Mock delete session
    with patch("agent_memory_server.messages.delete_session_memory") as mock_delete:
        mock_delete.return_value = None

        # Step 4: Delete the session
        response = await memory_test_client.delete_session_memory(session_id)
        assert response.status == "ok"

    # Verify it's gone by mocking a 404 response
    with patch("agent_memory_server.messages.get_session_memory") as mock_get_memory:
        mock_get_memory.return_value = None

        # This should raise an httpx.HTTPStatusError (404) since we return None from the mock
        from httpx import HTTPStatusError

        with pytest.raises(HTTPStatusError) as excinfo:
            await memory_test_client.get_session_memory(session_id)

        # Verify it's the correct error (404 Not Found)
        assert excinfo.value.response.status_code == 404


@pytest.mark.asyncio
async def test_long_term_memory(memory_test_client: MemoryAPIClient):
    """Test long-term memory creation and search"""
    # Create some test memories
    memories = [
        LongTermMemory(
            text="The user prefers dark mode in all applications",
            topics=["preferences", "ui"],
            user_id="test-user",
        ),
        LongTermMemory(
            text="The user's favorite color is blue",
            topics=["preferences", "colors"],
            user_id="test-user",
        ),
    ]

    # Mock the memory creation
    with patch(
        "agent_memory_server.long_term_memory.index_long_term_memories"
    ) as mock_index:
        mock_index.return_value = None

        # Store the memories
        with patch("agent_memory_server.api.settings.long_term_memory", True):
            response = await memory_test_client.create_long_term_memory(memories)
            assert response.status == "ok"

    # Mock the search results
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = LongTermMemoryResultsResponse(
            memories=[
                LongTermMemoryResult(
                    id_="1",
                    text="The user's favorite color is blue",
                    dist=0.2,
                    topics=["preferences", "colors"],
                    user_id="test-user",
                ),
                LongTermMemoryResult(
                    id_="2",
                    text="The user prefers dark mode in all applications",
                    dist=0.4,
                    topics=["preferences", "ui"],
                    user_id="test-user",
                ),
            ],
            total=2,
        )

        # Search with various filters
        with patch("agent_memory_server.api.settings.long_term_memory", True):
            results = await memory_test_client.search_long_term_memory(
                text="What color does the user prefer?",
                user_id={"eq": "test-user"},
                topics={"any": ["colors", "preferences"]},
            )

            assert results.total == 2
            # The "favorite color" memory should be the most relevant
            assert any("blue" in memory.text.lower() for memory in results.memories)

            # Try another search using filter objects instead of dictionaries
            results = await memory_test_client.search_long_term_memory(
                text="dark mode",
                namespace=Namespace(eq="test-namespace"),
            )

            assert results.total == 2
            assert any(
                "dark mode" in memory.text.lower() for memory in results.memories
            )


@pytest.mark.asyncio
async def test_client_with_context_manager(memory_app: FastAPI):
    """Test using the client with a context manager"""
    async with (
        AsyncClient(
            transport=ASGITransport(app=memory_app),
            base_url="http://test",
        ) as http_client,
        MemoryAPIClient(MemoryClientConfig(base_url="http://test")) as client,
    ):
        # Replace the internal client
        client._client = http_client

        # Perform a simple health check
        response = await client.health_check()
        assert response.now > 0

        # The client will be automatically closed when the context block exits


# Example usage is left in the file for documentation purposes,
# but commented out so it doesn't run during tests
"""
# This example demonstrates basic usage of the API client
if __name__ == "__main__":
    async def example():
        # Create a client
        base_url = "http://localhost:8000"  # Adjust to your server URL

        # Using context manager for automatic cleanup
        async with MemoryAPIClient(
            MemoryClientConfig(
                base_url=base_url,
                default_namespace="example-namespace"
            )
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
"""
