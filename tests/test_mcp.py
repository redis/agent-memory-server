import json

import pytest
from mcp import GetPromptResult
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.types import CallToolResult

from redis_memory_server.mcp import mcp_app
from redis_memory_server.models import LongTermMemory


class TestMCP:
    """Test search functionality and memory prompt endpoints via client sessions."""

    @pytest.mark.asyncio
    async def test_create_long_term_memory(self, session):
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
    async def test_search_memory(self, session):
        """Test searching through session memory using the client."""
        async with client_session(mcp_app._mcp_server) as client:
            results = await client.call_tool(
                "search_long_term_memory",
                {
                    "query": "Hello",
                    "namespace": "test-namespace",
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
            assert results["memories"][0]["text"] == "User: Hello"
            assert results["memories"][0]["dist"] > 0
            assert results["memories"][0]["created_at"] > 0
            assert results["memories"][0]["last_accessed"] > 0
            assert results["memories"][0]["user_id"] == ""
            assert results["memories"][0]["session_id"] == session
            assert results["memories"][0]["namespace"] == "test-namespace"
            assert results["memories"][1]["text"] == "Assistant: Hi there"
            assert results["memories"][1]["dist"] > 0
            assert results["memories"][1]["created_at"] > 0
            assert results["memories"][1]["last_accessed"] > 0
            assert results["memories"][1]["user_id"] == ""
            assert results["memories"][1]["session_id"] == session

    @pytest.mark.asyncio
    async def test_memory_prompt(self, session):
        """Test memory prompt with various parameter combinations."""
        async with client_session(mcp_app._mcp_server) as client:
            prompt = await client.get_prompt(
                "memory_prompt",
                {
                    "session_id": session,
                    "query": "Test query",
                    "namespace": "test-namespace",
                },
            )
            assert isinstance(prompt, GetPromptResult)
            assert prompt.messages[0].role == "assistant"  # the summary message
            assert prompt.messages[0].content.type == "text"
            assert prompt.messages[0].content.text.startswith(
                "## Long term memories related to the user's query\n - User: Hello\n- Assistant: Hi there"
            )
            assert prompt.messages[1].role == "user"
            assert prompt.messages[1].content.type == "text"
            assert prompt.messages[1].content.text == "Test query"

    @pytest.mark.asyncio
    async def test_memory_prompt_error_handling(self, session):
        """Test error handling in memory prompt generation via the client."""
        async with client_session(mcp_app._mcp_server) as client:
            # Test with a non-existent session id
            prompt = await client.get_prompt(
                "memory_prompt",
                {
                    "session_id": "non-existent",
                    "query": "Test query",
                    "namespace": "test-namespace",
                },
            )
            assert isinstance(prompt, GetPromptResult)
            assert prompt.messages[0].role == "user"
            assert prompt.messages[0].content.type == "text"
            assert prompt.messages[0].content.text == "Test query"
