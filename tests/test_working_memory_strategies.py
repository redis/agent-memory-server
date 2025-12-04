"""Tests for working memory strategy integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.models import (
    MemoryMessage,
    MemoryStrategyConfig,
    WorkingMemory,
    WorkingMemoryRequest,
)
from agent_memory_server.working_memory import get_working_memory, set_working_memory


class TestMemoryStrategyConfig:
    """Test memory strategy configuration model."""

    def test_default_strategy_config(self):
        """Test default strategy configuration."""
        config = MemoryStrategyConfig()
        assert config.strategy == "discrete"
        assert config.config == {}

    def test_custom_strategy_config(self):
        """Test custom strategy configuration."""
        config = MemoryStrategyConfig(
            strategy="summary", config={"max_summary_length": 300}
        )
        assert config.strategy == "summary"
        assert config.config == {"max_summary_length": 300}

    def test_model_dump(self):
        """Test model dump for JSON serialization."""
        config = MemoryStrategyConfig(
            strategy="preferences", config={"custom_param": "value"}
        )
        dumped = config.model_dump()
        assert dumped == {
            "strategy": "preferences",
            "config": {"custom_param": "value"},
        }


class TestWorkingMemoryStrategyIntegration:
    """Test working memory integration with strategy configuration."""

    def test_working_memory_default_strategy(self):
        """Test working memory has default strategy configuration."""
        memory = WorkingMemory(
            session_id="test-session",
            messages=[],
            memories=[],
        )
        assert memory.long_term_memory_strategy.strategy == "discrete"
        assert memory.long_term_memory_strategy.config == {}

    def test_working_memory_custom_strategy(self):
        """Test working memory with custom strategy configuration."""
        strategy_config = MemoryStrategyConfig(
            strategy="summary", config={"max_summary_length": 200}
        )
        memory = WorkingMemory(
            session_id="test-session",
            messages=[],
            memories=[],
            long_term_memory_strategy=strategy_config,
        )
        assert memory.long_term_memory_strategy.strategy == "summary"
        assert memory.long_term_memory_strategy.config == {"max_summary_length": 200}

    def test_working_memory_request_strategy(self):
        """Test working memory request with strategy configuration."""
        strategy_config = MemoryStrategyConfig(
            strategy="preferences", config={"focus_area": "user_interface"}
        )
        request = WorkingMemoryRequest(
            session_id="test-session", long_term_memory_strategy=strategy_config
        )
        assert request.long_term_memory_strategy.strategy == "preferences"
        assert request.long_term_memory_strategy.config == {
            "focus_area": "user_interface"
        }


class TestWorkingMemoryToolGeneration:
    """Test working memory MCP tool generation."""

    def test_get_create_long_term_memory_tool_description(self):
        """Test strategy-aware tool description generation."""
        strategy_config = MemoryStrategyConfig(
            strategy="summary", config={"max_summary_length": 300}
        )
        memory = WorkingMemory(
            session_id="test-session",
            messages=[],
            memories=[],
            long_term_memory_strategy=strategy_config,
        )

        with patch(
            "agent_memory_server.memory_strategies.get_memory_strategy"
        ) as mock_get_strategy:
            # Create a mock strategy with a synchronous method
            mock_strategy = AsyncMock()
            mock_strategy.get_extraction_description.return_value = (
                "Creates summaries (max 300 words)"
            )
            # Make the method synchronous
            mock_strategy.get_extraction_description = (
                lambda: "Creates summaries (max 300 words)"
            )
            mock_get_strategy.return_value = mock_strategy

            description = memory.get_create_long_term_memory_tool_description()

            assert "Creates summaries (max 300 words)" in description
            assert "MEMORY EXTRACTION BEHAVIOR:" in description
            assert "SEMANTIC MEMORIES" in description
            assert "EPISODIC MEMORIES" in description
            mock_get_strategy.assert_called_once_with("summary", max_summary_length=300)

    def test_create_long_term_memory_tool(self):
        """Test strategy-aware MCP tool generation."""
        strategy_config = MemoryStrategyConfig(strategy="discrete", config={})
        memory = WorkingMemory(
            session_id="test-session",
            messages=[],
            memories=[],
            long_term_memory_strategy=strategy_config,
        )

        with patch(
            "agent_memory_server.memory_strategies.get_memory_strategy"
        ) as mock_get_strategy:
            mock_strategy = AsyncMock()
            mock_strategy.get_extraction_description = lambda: "Extracts discrete facts"
            mock_get_strategy.return_value = mock_strategy

            tool_func = memory.create_long_term_memory_tool()

            assert callable(tool_func)
            assert tool_func.__name__ == "create_long_term_memories_discrete"
            assert "Extracts discrete facts" in tool_func.__doc__

    @pytest.mark.asyncio
    async def test_create_long_term_memory_tool_execution(self):
        """Test strategy-aware MCP tool execution."""
        strategy_config = MemoryStrategyConfig(strategy="preferences", config={})
        memory = WorkingMemory(
            session_id="test-session",
            messages=[],
            memories=[],
            long_term_memory_strategy=strategy_config,
        )

        with (
            patch(
                "agent_memory_server.memory_strategies.get_memory_strategy"
            ) as mock_get_strategy,
            patch("agent_memory_server.api.create_long_term_memory") as mock_create,
            patch(
                "agent_memory_server.dependencies.get_background_tasks"
            ) as mock_get_tasks,
        ):
            mock_strategy = AsyncMock()
            mock_strategy.get_extraction_description = lambda: "Extracts preferences"
            mock_get_strategy.return_value = mock_strategy

            # Create a simple mock response object
            class MockResponse:
                def model_dump(self):
                    return {"status": "ok"}

            mock_create.return_value = MockResponse()
            mock_get_tasks.return_value = AsyncMock()

            tool_func = memory.create_long_term_memory_tool()

            test_memories = [
                {
                    "text": "User prefers dark mode",
                    "memory_type": "semantic",
                    "topics": ["preferences"],
                }
            ]

            result = await tool_func(test_memories)

            assert result == {"status": "ok"}
            mock_create.assert_called_once()


class TestWorkingMemoryStorageWithStrategy:
    """Test working memory storage and retrieval with strategy configuration."""

    @pytest.mark.asyncio
    async def test_set_working_memory_with_strategy(self):
        """Test storing working memory with strategy configuration."""
        strategy_config = MemoryStrategyConfig(
            strategy="summary", config={"max_summary_length": 400}
        )
        memory = WorkingMemory(
            session_id="test-session-123",
            namespace="test-namespace",
            user_id="test-user",
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there!"),
            ],
            memories=[],
            long_term_memory_strategy=strategy_config,
        )

        with patch(
            "agent_memory_server.working_memory.get_redis_conn"
        ) as mock_get_redis:
            mock_redis = MagicMock()
            mock_redis.expire = AsyncMock()
            # json() is synchronous but returns an object with async methods
            mock_json = MagicMock()
            mock_json.set = AsyncMock()
            mock_redis.json.return_value = mock_json
            mock_get_redis.return_value = mock_redis

            await set_working_memory(memory, mock_redis)

            # Verify Redis JSON set was called
            mock_json.set.assert_called_once()
            call_args = mock_json.set.call_args

            # The data is passed directly as a dict (not JSON string) to redis.json().set()
            stored_data = call_args[0][2]  # Third positional arg is the data
            assert "long_term_memory_strategy" in stored_data
            assert stored_data["long_term_memory_strategy"]["strategy"] == "summary"
            assert (
                stored_data["long_term_memory_strategy"]["config"]["max_summary_length"]
                == 400
            )

    @pytest.mark.asyncio
    async def test_get_working_memory_with_strategy(self):
        """Test retrieving working memory with strategy configuration."""
        # Mock stored data that includes strategy configuration
        stored_data = {
            "messages": [{"role": "user", "content": "Hello", "id": "msg-1"}],
            "memories": [],
            "context": None,
            "user_id": "test-user",
            "tokens": 0,
            "session_id": "test-session-123",
            "namespace": "test-namespace",
            "ttl_seconds": None,
            "data": {},
            "long_term_memory_strategy": {
                "strategy": "preferences",
                "config": {"focus_area": "ui"},
            },
            "last_accessed": 1640995200,  # Unix timestamp
            "created_at": 1640995200,
            "updated_at": 1640995200,
        }

        with patch(
            "agent_memory_server.working_memory.get_redis_conn"
        ) as mock_get_redis:
            mock_redis = MagicMock()
            # json() is synchronous but returns an object with async methods
            mock_json = MagicMock()
            # Redis JSON returns dict directly, not bytes
            mock_json.get = AsyncMock(return_value=stored_data)
            mock_redis.json.return_value = mock_json
            mock_get_redis.return_value = mock_redis

            result = await get_working_memory(
                session_id="test-session-123",
                namespace="test-namespace",
                user_id="test-user",
                redis_client=mock_redis,
            )

            assert result is not None
            assert result.session_id == "test-session-123"
            assert result.long_term_memory_strategy.strategy == "preferences"
            assert result.long_term_memory_strategy.config == {"focus_area": "ui"}

    @pytest.mark.asyncio
    async def test_get_working_memory_without_strategy_uses_default(self):
        """Test retrieving working memory without strategy uses default."""
        # Mock stored data that doesn't include strategy configuration (legacy)
        stored_data = {
            "messages": [],
            "memories": [],
            "context": None,
            "user_id": "test-user",
            "tokens": 0,
            "session_id": "test-session-123",
            "namespace": "test-namespace",
            "ttl_seconds": None,
            "data": {},
            "last_accessed": 1640995200,
            "created_at": 1640995200,
            "updated_at": 1640995200,
        }

        with patch(
            "agent_memory_server.working_memory.get_redis_conn"
        ) as mock_get_redis:
            mock_redis = MagicMock()
            # json() is synchronous but returns an object with async methods
            mock_json = MagicMock()
            # Redis JSON returns dict directly, not bytes
            mock_json.get = AsyncMock(return_value=stored_data)
            mock_redis.json.return_value = mock_json
            mock_get_redis.return_value = mock_redis

            result = await get_working_memory(
                session_id="test-session-123",
                namespace="test-namespace",
                user_id="test-user",
                redis_client=mock_redis,
            )

            assert result is not None
            assert result.session_id == "test-session-123"
            # Should use default strategy when none is stored
            assert result.long_term_memory_strategy.strategy == "discrete"
            assert result.long_term_memory_strategy.config == {}
