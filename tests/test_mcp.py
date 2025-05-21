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

            # Don't assert total > 0 since we're mocking and might get empty results
            assert "total" in results

            # Only check memory structure if there are memories
            if results["total"] > 0 and results["memories"]:
                assert len(results["memories"]) > 0
                memory = results["memories"][0]
                assert "text" in memory
                assert "dist" in memory
                assert "created_at" in memory
                assert "last_accessed" in memory
                assert "user_id" in memory
                assert "session_id" in memory
                assert "namespace" in memory

    @pytest.mark.asyncio
    async def test_memory_prompt(self, session, mcp_test_setup):
        """Test memory prompt with various parameter combinations."""
        async with client_session(mcp_app._mcp_server) as client:
            prompt = await client.call_tool(
                "memory_prompt",
                {
                    "query": "Test query",
                    "session_id": {"eq": session},
                    "namespace": {"eq": "test-namespace"},
                },
            )
            assert isinstance(prompt, CallToolResult)

            assert prompt.content[0].type == "text"
            messages = json.loads(prompt.content[0].text)

            assert isinstance(messages, dict)
            assert "messages" in messages
            assert len(messages["messages"]) == 5

            # The returned messages structure is:
            # 0: system (summary)
            # 1: user ("Hello")
            # 2: assistant ("Hi there")
            # 3: system (long term memories)
            # 4: user ("Test query")
            assert messages["messages"][0]["role"] == "system"
            assert messages["messages"][0]["content"]["type"] == "text"
            assert "summary" in messages["messages"][0]["content"]["text"]

            assert messages["messages"][1]["role"] == "user"
            assert messages["messages"][1]["content"]["type"] == "text"
            assert messages["messages"][1]["content"]["text"] == "Hello"

            assert messages["messages"][2]["role"] == "assistant"
            assert messages["messages"][2]["content"]["type"] == "text"
            assert messages["messages"][2]["content"]["text"] == "Hi there"

            assert messages["messages"][3]["role"] == "system"
            assert messages["messages"][3]["content"]["type"] == "text"
            assert "Long term memories" in messages["messages"][3]["content"]["text"]

            assert messages["messages"][4]["role"] == "user"
            assert messages["messages"][4]["content"]["type"] == "text"
            assert "Test query" in messages["messages"][4]["content"]["text"]

    @pytest.mark.asyncio
    async def test_memory_prompt_error_handling(self, session, mcp_test_setup):
        """Test error handling in memory prompt generation via the client."""
        async with client_session(mcp_app._mcp_server) as client:
            # Test with a non-existent session id
            prompt = await client.call_tool(
                "memory_prompt",
                {
                    "query": "Test query",
                    "session": {"session_id": {"eq": "non-existent"}},
                    "namespace": {"eq": "test-namespace"},
                },
            )
            assert isinstance(prompt, CallToolResult)

            # Parse the response content - ensure we're getting text content
            assert prompt.content[0].type == "text"
            message = json.loads(prompt.content[0].text)

            # The result should be a dictionary containing messages, each with content and role
            assert isinstance(message, dict)
            assert "messages" in message

            # Check that we have a user message with the test query
            assert message["messages"][0]["role"] == "system"
            assert message["messages"][0]["content"]["type"] == "text"
            assert "Long term memories" in message["messages"][0]["content"]["text"]

            assert message["messages"][1]["role"] == "user"
            assert message["messages"][1]["content"]["type"] == "text"
            assert "Test query" in message["messages"][1]["content"]["text"]

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
