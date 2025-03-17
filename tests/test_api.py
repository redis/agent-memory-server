from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from redis_memory_server.config import Settings
from redis_memory_server.extraction import handle_extraction
from redis_memory_server.long_term_memory import index_messages
from redis_memory_server.models import (
    RedisearchResult,
    SearchResults,
)
from redis_memory_server.summarization import handle_compaction


@pytest.fixture
def mock_openai_client_wrapper():
    """Create a mock OpenAIClientWrapper that doesn't need an API key"""
    with patch("redis_memory_server.models.OpenAIClientWrapper") as mock_wrapper:
        # Create a mock instance
        mock_instance = AsyncMock()
        mock_wrapper.return_value = mock_instance

        # Mock the create_embedding and create_chat_completion methods
        mock_instance.create_embedding.return_value = np.array(
            [[0.1] * 1536], dtype=np.float32
        )
        mock_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 100},
        }

        yield mock_wrapper


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test the health endpoint"""
        response = await client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert "now" in data
        assert isinstance(data["now"], int)


class TestMemoryEndpoints:
    async def test_get_sessions_empty(self, client):
        """Test the get_sessions endpoint with no sessions"""
        response = await client.get("/sessions/?page=1&size=10")

        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    async def test_get_sessions_with_sessions(self, client, test_session_setup):
        """Test the get_sessions endpoint with a session"""
        response = await client.get("/sessions/?page=1&size=10")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0] == test_session_setup

    async def test_get_memory(self, client, test_session_setup):
        """Test the get_memory endpoint"""
        session_id = test_session_setup

        response = await client.get(f"/sessions/{session_id}/memory")

        assert response.status_code == 200

        data = response.json()
        assert "messages" in data
        assert "context" in data
        assert "tokens" in data

        assert len(data["messages"]) == 2

        # Note: The order may be reversed due to LPUSH in Redis
        roles = [msg["role"] for msg in data["messages"]]
        contents = [msg["content"] for msg in data["messages"]]
        assert "user" in roles
        assert "assistant" in roles
        assert "Hello" in contents
        assert "Hi there" in contents

        # Check context and tokens
        assert data["context"] == "Sample context"
        assert int(data["tokens"]) == 150  # Convert string to int for comparison

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_post_memory(self, client):
        """Test the post_memory endpoint"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "context": "Previous context",
        }

        response = await client.post("/sessions/test-session/memory", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_post_memory_stores_in_long_term_memory(self, client):
        """Test the post_memory endpoint"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "context": "Previous context",
        }
        mock_settings = Settings(long_term_memory=True)
        mock_add_task = MagicMock()

        with (
            patch("redis_memory_server.memory.settings", mock_settings),
            patch("redis_memory_server.memory.BackgroundTasks.add_task", mock_add_task),
        ):
            response = await client.post("/sessions/test-session/memory", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        # Check that background tasks were called
        # We expect 3 tasks:
        # 1. Topic/entity extraction for first message
        # 2. Topic/entity extraction for second message
        # 3. Long-term memory indexing
        assert mock_add_task.call_count == 3

        # Check that the last call was for long-term memory indexing
        assert mock_add_task.call_args_list[-1][0][0] == index_messages

        # Check that the first two calls were for topic/entity extraction
        assert mock_add_task.call_args_list[0][0][0] == handle_extraction
        assert mock_add_task.call_args_list[1][0][0] == handle_extraction

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_post_memory_compacts_long_conversation(self, client):
        """Test the post_memory endpoint"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "context": "Previous context",
        }
        mock_settings = Settings(window_size=1, long_term_memory=False)
        mock_add_task = MagicMock()

        with (
            patch("redis_memory_server.memory.settings", mock_settings),
            patch("redis_memory_server.memory.BackgroundTasks.add_task", mock_add_task),
        ):
            response = await client.post("/sessions/test-session/memory", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        # Check that background tasks were called
        # We expect 3 tasks:
        # 1. Topic/entity extraction for first message
        # 2. Topic/entity extraction for second message
        # 3. Compaction
        assert mock_add_task.call_count == 3

        # Check that the last call was for compaction
        assert mock_add_task.call_args_list[-1][0][0] == handle_compaction

        # Check that the first two calls were for topic/entity extraction
        assert mock_add_task.call_args_list[0][0][0] == handle_extraction
        assert mock_add_task.call_args_list[1][0][0] == handle_extraction

    @pytest.mark.asyncio
    async def test_delete_memory(self, client, test_session_setup):
        """Test the delete_memory endpoint"""
        session_id = test_session_setup

        response = await client.get(f"/sessions/{session_id}/memory")

        assert response.status_code == 200

        data = response.json()
        assert len(data["messages"]) == 2

        response = await client.delete(f"/sessions/{session_id}/memory")

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        response = await client.get(f"/sessions/{session_id}/memory")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 0


@pytest.mark.requires_api_keys
class TestRetrievalEndpoint:
    @patch("redis_memory_server.retrieval.search_messages")
    @pytest.mark.asyncio
    async def test_retrieval(self, mock_search, client):
        """Test the retrieval endpoint"""
        mock_search.return_value = SearchResults(
            docs=[
                RedisearchResult(role="user", content="Hello, world!", dist=0.25),
                RedisearchResult(role="assistant", content="Hi there!", dist=0.75),
            ],
            total=2,
        )

        # Create payload
        payload = {"text": "What is the capital of France?"}

        # Call endpoint with the correct URL format (matching the router definition)
        response = await client.post("/sessions/test-session/retrieval", json=payload)

        # Check status code
        assert response.status_code == 200, response.text

        # Check response structure
        data = response.json()
        assert "docs" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["docs"]) == 2

        # Check first result
        assert data["docs"][0]["role"] == "user"
        assert data["docs"][0]["content"] == "Hello, world!"
        assert data["docs"][0]["dist"] == 0.25

        # Check second result
        assert data["docs"][1]["role"] == "assistant"
        assert data["docs"][1]["content"] == "Hi there!"
        assert data["docs"][1]["dist"] == 0.75
