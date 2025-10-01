"""
Tests for LangChain integration.

These tests verify that the automatic tool conversion works correctly
and that the tools can be used with LangChain agents.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test imports
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

# Helper functions


def _langchain_available() -> bool:
    """Check if LangChain is available."""
    try:
        import langchain_core.tools  # noqa: F401

        return True
    except ImportError:
        return False


def _create_mock_client() -> MemoryAPIClient:
    """Create a mock MemoryAPIClient for testing."""
    config = MemoryClientConfig(base_url="http://localhost:8000")
    client = MemoryAPIClient(config)

    # Mock the HTTP client to avoid actual requests
    client._client = MagicMock()

    return client


class TestLangChainIntegration:
    """Tests for LangChain integration module."""

    def test_import_without_langchain(self):
        """Test that importing without langchain installed raises helpful error."""
        with patch.dict(
            "sys.modules", {"langchain_core": None, "langchain_core.tools": None}
        ):
            # Re-import to trigger the check
            import importlib

            import agent_memory_client.integrations.langchain as lc_module

            importlib.reload(lc_module)

            # Should raise ImportError with helpful message
            with pytest.raises(ImportError, match="LangChain is required"):
                lc_module._check_langchain_available()

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    def test_get_memory_tools_all(self):
        """Test getting all memory tools."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        # Create mock client
        client = _create_mock_client()

        # Get all tools
        tools = get_memory_tools(
            memory_client=client, session_id="test_session", user_id="test_user"
        )

        # Should return all 9 tools
        assert len(tools) == 9

        # Verify tool names
        tool_names = {tool.name for tool in tools}
        expected_names = {
            "search_memory",
            "get_or_create_working_memory",
            "add_memory_to_working_memory",
            "update_working_memory_data",
            "get_long_term_memory",
            "create_long_term_memory",
            "edit_long_term_memory",
            "delete_long_term_memories",
            "get_current_datetime",
        }
        assert tool_names == expected_names

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    def test_get_memory_tools_selective(self):
        """Test getting only specific tools."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        client = _create_mock_client()

        # Get only specific tools
        tools = get_memory_tools(
            memory_client=client,
            session_id="test_session",
            user_id="test_user",
            tools=["search_memory", "create_long_term_memory"],
        )

        # Should return only 2 tools
        assert len(tools) == 2

        tool_names = {tool.name for tool in tools}
        assert tool_names == {"search_memory", "create_long_term_memory"}

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    def test_get_memory_tools_invalid_tool_name(self):
        """Test that invalid tool names raise ValueError."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        client = _create_mock_client()

        with pytest.raises(ValueError, match="Invalid tool names"):
            get_memory_tools(
                memory_client=client,
                session_id="test_session",
                user_id="test_user",
                tools=["invalid_tool", "another_invalid"],
            )

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    @pytest.mark.asyncio
    async def test_search_memory_tool_execution(self):
        """Test that search_memory tool executes correctly."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        # Create mock client with search_memory_tool method
        client = _create_mock_client()
        client.search_memory_tool = AsyncMock(
            return_value={
                "summary": "Found 2 memories",
                "memories": [
                    {"text": "User loves pizza", "relevance_score": 0.95},
                    {"text": "User works at TechCorp", "relevance_score": 0.87},
                ],
            }
        )

        # Get tools
        tools = get_memory_tools(
            memory_client=client,
            session_id="test_session",
            user_id="test_user",
            tools=["search_memory"],
        )

        # Execute the tool
        search_tool = tools[0]
        result = await search_tool.ainvoke(
            {"query": "user information", "max_results": 5}
        )

        # Verify the result
        assert "Found 2 memories" in result

        # Verify the client method was called correctly
        client.search_memory_tool.assert_called_once()
        call_kwargs = client.search_memory_tool.call_args.kwargs
        assert call_kwargs["query"] == "user information"
        assert call_kwargs["max_results"] == 5

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    @pytest.mark.asyncio
    async def test_add_memory_tool_execution(self):
        """Test that add_memory_to_working_memory tool executes correctly."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        client = _create_mock_client()
        client.add_memory_tool = AsyncMock(
            return_value={
                "summary": "Successfully stored semantic memory",
                "success": True,
            }
        )

        tools = get_memory_tools(
            memory_client=client,
            session_id="test_session",
            user_id="test_user",
            tools=["add_memory_to_working_memory"],
        )

        add_tool = tools[0]
        result = await add_tool.ainvoke(
            {
                "text": "User loves pizza",
                "memory_type": "semantic",
                "topics": ["food", "preferences"],
            }
        )

        assert "Successfully stored" in result

        # Verify the client method was called
        client.add_memory_tool.assert_called_once()
        call_kwargs = client.add_memory_tool.call_args.kwargs
        assert call_kwargs["text"] == "User loves pizza"
        assert call_kwargs["memory_type"] == "semantic"
        assert call_kwargs["topics"] == ["food", "preferences"]
        assert call_kwargs["session_id"] == "test_session"
        assert call_kwargs["user_id"] == "test_user"

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    @pytest.mark.asyncio
    async def test_create_long_term_memory_tool_execution(self):
        """Test that create_long_term_memory tool executes correctly."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        client = _create_mock_client()
        client.resolve_function_call = AsyncMock(
            return_value={
                "success": True,
                "formatted_response": "Created 2 long-term memories",
            }
        )

        tools = get_memory_tools(
            memory_client=client,
            session_id="test_session",
            user_id="test_user",
            tools=["create_long_term_memory"],
        )

        create_tool = tools[0]
        result = await create_tool.ainvoke(
            {
                "memories": [
                    {"text": "User loves pizza", "memory_type": "semantic"},
                    {"text": "User works at TechCorp", "memory_type": "semantic"},
                ]
            }
        )

        assert "Created 2 long-term memories" in result

        # Verify resolve_function_call was called correctly
        client.resolve_function_call.assert_called_once()
        call_kwargs = client.resolve_function_call.call_args.kwargs
        assert call_kwargs["function_name"] == "create_long_term_memory"
        assert len(call_kwargs["function_arguments"]["memories"]) == 2

    @pytest.mark.skipif(not _langchain_available(), reason="LangChain not installed")
    def test_tool_has_correct_schema(self):
        """Test that generated tools have correct schemas."""
        from agent_memory_client.integrations.langchain import get_memory_tools

        client = _create_mock_client()

        tools = get_memory_tools(
            memory_client=client,
            session_id="test_session",
            user_id="test_user",
            tools=["search_memory"],
        )

        search_tool = tools[0]

        # Verify tool has required attributes
        assert search_tool.name == "search_memory"
        assert search_tool.description
        assert "search" in search_tool.description.lower()

        # Verify args_schema exists and has expected fields
        assert search_tool.args_schema is not None
        schema = search_tool.args_schema.model_json_schema()
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
