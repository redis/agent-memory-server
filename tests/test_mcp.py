import json
from unittest import mock

import pytest
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.types import CallToolResult

from agent_memory_server.mcp import mcp_app
from agent_memory_server.models import (
    LongTermMemory,
)


@pytest.fixture
async def mcp_test_setup(async_redis_client, search_index):
    with (
        mock.patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            return_value=async_redis_client,
        ) as _mock_ltm_redis,
        mock.patch(
            "agent_memory_server.api.get_redis_conn",
            return_value=async_redis_client,
            create=True,
        ) as _mock_api_redis,
    ):
        yield


class TestMCP:
    """Test search functionality and memory prompt endpoints via client sessions."""

    @pytest.mark.asyncio
    async def test_create_long_term_memory(self, session, mcp_test_setup):
        async with client_session(mcp_app._mcp_server) as client:
            results = await client.call_tool(
                "create_long_term_memories",
                {
                    "memories": [
                        LongTermMemory(text="Hello", session_id=session),
                    ],
                },
            )
            assert isinstance(results, CallToolResult)
            assert results.content[0].type == "text"
            assert results.content[0].text == '{"status": "ok"}'

    @pytest.mark.asyncio
    async def test_search_memory(self, session, mcp_test_setup):
        """Test searching through session memory using the client."""
        async with client_session(mcp_app._mcp_server) as client:
            results = await client.call_tool(
                "search_long_term_memory",
                {
                    "text": "Hello",
                    "namespace": {"eq": "test-namespace"},
                },
            )
            assert isinstance(
                results,
                CallToolResult,
            )
            assert len(results.content) > 0
            assert results.content[0].type == "text"
            results = json.loads(results.content[0].text)
            assert results["total"] > 0
            assert len(results["memories"]) == 2
            assert results["memories"][0]["text"] == "user: Hello"
            assert results["memories"][0]["dist"] > 0
            assert results["memories"][0]["created_at"] > 0
            assert results["memories"][0]["last_accessed"] > 0
            assert results["memories"][0]["user_id"] == "test-user"
            assert results["memories"][0]["session_id"] == session
            assert results["memories"][0]["namespace"] == "test-namespace"
            assert results["memories"][1]["text"] == "assistant: Hi there"
            assert results["memories"][1]["dist"] > 0
            assert results["memories"][1]["created_at"] > 0
            assert results["memories"][1]["last_accessed"] > 0
            assert results["memories"][1]["user_id"] == "test-user"
            assert results["memories"][1]["session_id"] == session

    @pytest.mark.asyncio
    async def test_memory_prompt(self, session, mcp_test_setup):
        """Test memory prompt with various parameter combinations."""
        async with client_session(mcp_app._mcp_server) as client:
            prompt = await client.call_tool(
                "hydrate_memory_prompt",
                {
                    "text": "Test query",
                    "session_id": {"eq": session},
                    "namespace": {"eq": "test-namespace"},
                },
            )
            assert isinstance(prompt, CallToolResult)

            # Parse the response content - ensure we're getting text content
            assert prompt.content[0].type == "text"
            message = json.loads(prompt.content[0].text)

            # The result should be a dictionary with content and role
            assert isinstance(message, dict)
            assert "content" in message
            assert "role" in message

            # Check the message content and role
            assert message["role"] == "assistant"
            assert message["content"]["type"] == "text"
            assert (
                "Long term memories related to the user's query"
                in message["content"]["text"]
            )
            assert "user: Hello" in message["content"]["text"]
            assert "assistant: Hi there" in message["content"]["text"]

    @pytest.mark.asyncio
    async def test_memory_prompt_error_handling(self, session, mcp_test_setup):
        """Test error handling in memory prompt generation via the client."""
        async with client_session(mcp_app._mcp_server) as client:
            # Test with a non-existent session id
            prompt = await client.call_tool(
                "hydrate_memory_prompt",
                {
                    "text": "Test query",
                    "session_id": {"eq": "non-existent"},
                    "namespace": {"eq": "test-namespace"},
                },
            )
            assert isinstance(prompt, CallToolResult)

            # Parse the response content - ensure we're getting text content
            assert prompt.content[0].type == "text"
            message = json.loads(prompt.content[0].text)

            # The result should be a dictionary with content and role
            assert isinstance(message, dict)
            assert "content" in message
            assert "role" in message

            # Check that we have a user message with the test query
            assert message["role"] == "user"
            assert message["content"]["type"] == "text"
            assert message["content"]["text"] == "Test query"

    @pytest.mark.asyncio
    async def test_default_namespace_injection(self, monkeypatch):
        """
        Ensure that when default_namespace is set on mcp_app, search_long_term_memory injects it automatically.
        """
        from agent_memory_server.models import (
            LongTermMemoryResult,
            LongTermMemoryResults,
        )

        # Capture injected namespace
        injected = {}

        async def fake_core_search(payload):
            injected["namespace"] = payload.namespace.eq if payload.namespace else None
            # Return a dummy result with total>0 to skip fake fallback
            return LongTermMemoryResults(
                total=1,
                memories=[
                    LongTermMemoryResult(
                        id_="id",
                        text="x",
                        dist=0.0,
                        created_at=1,
                        last_accessed=1,
                        user_id="",
                        session_id="",
                        namespace=payload.namespace.eq if payload.namespace else None,
                        topics=[],
                        entities=[],
                    )
                ],
                next_offset=None,
            )

        # Patch the core search function used by the MCP tool
        monkeypatch.setattr(
            "agent_memory_server.mcp.core_search_long_term_memory", fake_core_search
        )
        # Temporarily set default_namespace on the MCP app instance
        original_ns = mcp_app.default_namespace
        mcp_app.default_namespace = "default-ns"
        try:
            # Call the tool without specifying a namespace
            async with client_session(mcp_app._mcp_server) as client:
                await client.call_tool(
                    "search_long_term_memory",
                    {"text": "anything"},
                )
            # Verify that our fake core received the default namespace
            assert injected.get("namespace") == "default-ns"
        finally:
            # Restore original namespace
            mcp_app.default_namespace = original_ns
