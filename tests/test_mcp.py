import json

import pytest
from mcp import GetPromptResult
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.types import CallToolResult
from pydantic import AnyUrl

from redis_memory_server.mcp import mcp_app
from redis_memory_server.models.messages import (
    MemoryMessage,
    MemoryMessagesAndContext,
)


class TestMemoryOperations:
    """Test basic memory operations in FastMCP server using client sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions(self, session):
        """Test listing sessions using client_session."""
        async with client_session(mcp_app._mcp_server) as client:
            sessions = await client.call_tool("list_sessions", {"page": 1, "size": 10})
            assert session in [s.text for s in sessions.content]

    @pytest.mark.asyncio
    async def test_get_session_memory(self, session):
        """Test getting session memory through a client call."""
        async with client_session(mcp_app._mcp_server) as client:
            memory = await client.read_resource(AnyUrl(f"memory://{session}/memory"))
            response = json.loads(memory.contents[0].text)
            assert response["context"] == "Sample context"
            assert response["tokens"] == 150
            assert response["messages"][0]["role"] == "user"
            assert response["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_add_memory(self, session):
        """Test adding memory and verifying its persistence via the client."""
        session_id = session

        messages = MemoryMessagesAndContext(
            messages=[
                MemoryMessage(role="user", content="Test message"),
                MemoryMessage(role="assistant", content="Test response"),
            ],
            context="Test context",
        )
        async with client_session(mcp_app._mcp_server) as client:
            response = await client.call_tool(
                "add_memory", {"session_id": session_id, "memory_messages": messages}
            )
            assert response.content[0].text == '{"status": "ok"}'
            # Verify stored memory
            stored_memory = await client.read_resource(
                AnyUrl(f"memory://{session_id}/memory")
            )
            memory = json.loads(stored_memory.contents[0].text)

            assert memory["context"] == messages.context
            assert len(memory["messages"]) == 4  # Includes the initial messages
            assert memory["messages"][-1]["content"] == messages.messages[1].content
            assert memory["messages"][-2]["content"] == messages.messages[0].content

    @pytest.mark.asyncio
    async def test_delete_session_memory(self, session):
        """Test deleting session memory via client and ensuring it is removed."""
        async with client_session(mcp_app._mcp_server) as client:
            initial_memory = await client.read_resource(
                AnyUrl(f"memory://{session}/memory")
            )
            assert initial_memory is not None
            response = await client.call_tool(
                "delete_session_memory", {"session_id": session}
            )
            assert response.content[0].text == '{"status": "ok"}'
            # Verify memory deletion
            with pytest.raises(Exception, match="Session not found"):
                res = await client.read_resource(AnyUrl(f"memory://{session}/memory"))
                print(res)


class TestSearchAndPrompts:
    """Test search functionality and memory prompt endpoints via client sessions."""

    @pytest.mark.asyncio
    async def test_search_memory(self, session):
        """Test searching through session memory using the client."""
        async with client_session(mcp_app._mcp_server) as client:
            results = await client.call_tool(
                "search_memory", {"session_id": session, "query": "Hello"}
            )
            assert isinstance(
                results,
                CallToolResult,
            )
            assert len(results.content) > 0
            assert results.content[0].type == "text"
            assert (
                results.content[0].text
                == '{"docs": [{"role": "user", "content": "Hello", "dist": 3.57627868652e-07}, {"role": "assistant", "content": "Hi there", "dist": 0.103089511395}], "total": 2}'
            )

    @pytest.mark.asyncio
    async def test_memory_prompt(self, session):
        """Test memory prompt with various parameter combinations."""
        async with client_session(mcp_app._mcp_server) as client:
            prompt = await client.get_prompt(
                "memory_prompt",
                {"session_id": session, "query": "Test query"},
            )
            assert isinstance(prompt, GetPromptResult)
            assert prompt.messages[0].role == "user"
            assert prompt.messages[0].content.type == "text"
            assert (
                prompt.messages[0].content.text
                == '{"name": "memory-prompt", "description": "A prompt containing the user\'s query enriched with memory context", "arguments": [{"name": "session_id", "description": "The session ID to interact with", "required": false}, {"name": "query", "description": "The query or message to process", "required": false}]}'
            )

    @pytest.mark.asyncio
    async def test_memory_prompt_error_handling(self):
        """Test error handling in memory prompt generation via the client."""
        async with client_session(mcp_app._mcp_server) as client:
            # Test with a non-existent session id
            prompt = await client.get_prompt(
                "memory_prompt", {"session_id": "non-existent", "query": "Test query"}
            )
            assert isinstance(prompt, GetPromptResult)
            assert prompt.messages[0].role == "user"
            assert prompt.messages[0].content.type == "text"
            assert (
                prompt.messages[0].content.text
                == '{"name": "memory-prompt", "description": "A prompt containing the user\'s query enriched with memory context", "arguments": [{"name": "session_id", "description": "The session ID to interact with", "required": false}, {"name": "query", "description": "The query or message to process", "required": false}]}'
            )
