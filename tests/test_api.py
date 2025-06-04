from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent_memory_server.config import Settings
from agent_memory_server.long_term_memory import (
    index_long_term_memories,
    promote_working_memory_to_long_term,
)
from agent_memory_server.models import (
    MemoryMessage,
    MemoryRecordResult,
    MemoryRecordResultsResponse,
    MemoryTypeEnum,
    SessionListResponse,
    WorkingMemory,
    WorkingMemoryResponse,
)


@pytest.fixture
def mock_openai_client_wrapper():
    """Create a mock OpenAIClientWrapper that doesn't need an API key"""
    with patch("agent_memory_server.models.OpenAIClientWrapper") as mock_wrapper:
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
        response = await client.get("/v1/health")

        assert response.status_code == 200

        data = response.json()
        assert "now" in data
        assert isinstance(data["now"], int)


class TestMemoryEndpoints:
    async def test_list_sessions_empty(self, client):
        """Test the list_sessions endpoint with no sessions"""
        response = await client.get("/v1/working-memory/?offset=0&limit=10")

        assert response.status_code == 200

        data = response.json()
        response = SessionListResponse(**data)
        assert response.sessions == []
        assert response.total == 0

    async def test_list_sessions_with_sessions(self, client, session):
        """Test the list_sessions endpoint with a session"""
        response = await client.get(
            "/v1/working-memory/?offset=0&limit=10&namespace=test-namespace"
        )
        assert response.status_code == 200

        data = response.json()
        response = SessionListResponse(**data)
        assert response.sessions == [session]
        assert response.total == 1

    async def test_get_memory(self, client, session):
        """Test the get_memory endpoint"""
        session_id = session

        response = await client.get(
            f"/v1/working-memory/{session_id}?namespace=test-namespace"
        )

        assert response.status_code == 200

        data = response.json()
        response = WorkingMemoryResponse(**data)
        assert response.messages == [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

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
    async def test_put_memory(self, client):
        """Test the post_memory endpoint"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "memories": [],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }

        response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        # Should return the working memory, not just a status
        assert "messages" in data
        assert "context" in data
        assert "namespace" in data
        assert data["context"] == "Previous context"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Hi there"

        # Verify we can still retrieve the session memory
        updated_session = await client.get(
            "/v1/working-memory/test-session?namespace=test-namespace"
        )
        assert updated_session.status_code == 200
        assert updated_session.json()["messages"] == payload["messages"]

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_put_memory_stores_messages_in_long_term_memory(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test the put_memory endpoint"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "memories": [],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }
        mock_settings = Settings(long_term_memory=True)

        with patch("agent_memory_server.api.settings", mock_settings):
            response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        # Should return the working memory, not just a status
        assert "messages" in data
        assert "context" in data
        assert data["context"] == "Previous context"

        # Check that background tasks were called
        assert mock_background_tasks.add_task.call_count == 1

        # Check that the last call was for long-term memory indexing
        assert (
            mock_background_tasks.add_task.call_args_list[-1][0][0]
            == index_long_term_memories
        )

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_put_memory_with_structured_memories_triggers_promotion(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test that structured memories trigger background promotion task"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [],
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "id": "test-memory-1",
                    "memory_type": "semantic",
                    "namespace": "test-namespace",
                }
            ],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }
        mock_settings = Settings(long_term_memory=True)

        with patch("agent_memory_server.api.settings", mock_settings):
            response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "memories" in data
        assert len(data["memories"]) == 1
        assert data["memories"][0]["text"] == "User prefers dark mode"

        # Check that promotion background task was called
        assert mock_background_tasks.add_task.call_count == 1

        # Check that it was the promotion task, not indexing
        assert (
            mock_background_tasks.add_task.call_args_list[0][0][0]
            == promote_working_memory_to_long_term
        )

        # Check the arguments passed to the promotion task
        task_args = mock_background_tasks.add_task.call_args_list[0][0]
        assert task_args[1] == "test-session"  # session_id
        assert task_args[2] == "test-namespace"  # namespace

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_post_memory_compacts_long_conversation(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test the post_memory endpoint with window size exceeded"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "memories": [],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }
        mock_settings = Settings(window_size=1, long_term_memory=False)

        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api._summarize_working_memory"
            ) as mock_summarize,
        ):
            # Mock the summarization to return the working memory with updated context
            mock_summarized_memory = WorkingMemory(
                messages=[
                    MemoryMessage(role="assistant", content="Hi there")
                ],  # Only keep last message
                memories=[],
                context="Summary: User greeted and assistant responded.",
                session_id="test-session",
                namespace="test-namespace",
            )
            mock_summarize.return_value = mock_summarized_memory

            response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        # Should return the summarized working memory
        assert "messages" in data
        assert "context" in data
        # Should have been summarized (only 1 message kept due to window_size=1)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Hi there"
        assert "Summary:" in data["context"]

        # Verify summarization was called
        mock_summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_memory(self, client, session):
        """Test the delete_memory endpoint"""
        session_id = session

        response = await client.get(
            f"/v1/working-memory/{session_id}?namespace=test-namespace"
        )

        assert response.status_code == 200

        data = response.json()
        assert len(data["messages"]) == 2

        response = await client.delete(
            f"/v1/working-memory/{session_id}?namespace=test-namespace"
        )

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        response = await client.get(
            f"/v1/working-memory/{session_id}?namespace=test-namespace"
        )
        assert response.status_code == 200

        # Should return empty working memory after deletion
        data = response.json()
        assert len(data["messages"]) == 0


@pytest.mark.requires_api_keys
class TestSearchEndpoint:
    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_search(self, mock_search, client):
        """Test the search endpoint"""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=2,
            memories=[
                MemoryRecordResult(id="1", text="User: Hello, world!", dist=0.25),
                MemoryRecordResult(id="2", text="Assistant: Hi there!", dist=0.75),
            ],
            next_offset=None,
        )

        # Create payload
        payload = {"text": "What is the capital of France?"}

        # Call endpoint with the correct URL format (matching the router definition)
        response = await client.post("/v1/v1/long-term-memory//search", json=payload)

        # Check status code
        assert response.status_code == 200, response.text

        # Check response structure
        data = response.json()
        assert "memories" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["memories"]) == 2

        # Check first result
        assert data["memories"][0]["id"] == "1"
        assert data["memories"][0]["text"] == "User: Hello, world!"
        assert data["memories"][0]["dist"] == 0.25

        # Check second result
        assert data["memories"][1]["id"] == "2"
        assert data["memories"][1]["text"] == "Assistant: Hi there!"
        assert data["memories"][1]["dist"] == 0.75


@pytest.mark.requires_api_keys
class TestMemoryPromptEndpoint:
    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_session_id(self, mock_get_working_memory, client):
        """Test the memory_prompt endpoint with only session_id provided"""
        # Mock the session memory
        mock_session_memory = WorkingMemoryResponse(
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there"),
            ],
            memories=[],
            session_id="test-session",
            context="Previous conversation context",
            tokens=150,
        )
        mock_get_working_memory.return_value = mock_session_memory

        # Call the endpoint
        query = "What's the weather like?"
        response = await client.post(
            "/v1/memory/prompt",
            json={
                "query": query,
                "session": {
                    "session_id": "test-session",
                    "namespace": "test-namespace",
                    "window_size": 10,
                    "model_name": "gpt-4o",
                    "context_window_max": 1000,
                },
            },
        )

        # Check status code
        assert response.status_code == 200

        # Check response data
        data = response.json()
        assert isinstance(data, dict)
        assert (
            len(data["messages"]) == 4
        )  # Context message + 2 session messages + query

        # Verify the messages content
        assert data["messages"][0]["role"] == "system"
        assert "Previous conversation context" in data["messages"][0]["content"]["text"]
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"]["text"] == "Hello"
        assert data["messages"][2]["role"] == "assistant"
        assert data["messages"][2]["content"]["text"] == "Hi there"
        assert data["messages"][3]["role"] == "user"
        assert data["messages"][3]["content"]["text"] == query

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_long_term_memory(self, mock_search, client):
        """Test the memory_prompt endpoint with only long_term_search_payload provided"""
        # Mock the long-term memory search
        mock_search.return_value = MemoryRecordResultsResponse(
            total=2,
            memories=[
                MemoryRecordResult(id="1", text="User likes coffee", dist=0.25),
                MemoryRecordResult(
                    id="2", text="User is allergic to peanuts", dist=0.35
                ),
            ],
            next_offset=None,
        )

        # Prepare the payload
        payload = {
            "query": "What should I eat?",
            "long_term_search": {
                "text": "food preferences allergies",
            },
        }

        # Call the endpoint
        response = await client.post("/v1/memory/prompt", json=payload)

        # Check status code
        assert response.status_code == 200

        # Check response data
        data = response.json()
        assert isinstance(data, dict)
        assert len(data["messages"]) == 2  # Long-term memory message + query

        # Verify the messages content
        assert data["messages"][0]["role"] == "system"
        assert "Long term memories" in data["messages"][0]["content"]["text"]
        assert "User likes coffee" in data["messages"][0]["content"]["text"]
        assert "User is allergic to peanuts" in data["messages"][0]["content"]["text"]
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"]["text"] == "What should I eat?"

    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_both_sources(
        self, mock_search, mock_get_working_memory, client
    ):
        """Test the memory_prompt endpoint with both session_id and long_term_search_payload"""
        # Mock session memory
        mock_session_memory = WorkingMemoryResponse(
            messages=[
                MemoryMessage(role="user", content="How do you make pasta?"),
                MemoryMessage(
                    role="assistant",
                    content="Boil water, add pasta, cook until al dente.",
                ),
            ],
            memories=[],
            session_id="test-session",
            context="Cooking conversation",
            tokens=200,
        )
        mock_get_working_memory.return_value = mock_session_memory

        # Mock the long-term memory search
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="1", text="User prefers gluten-free pasta", dist=0.3
                ),
            ],
            next_offset=None,
        )

        # Prepare the payload
        payload = {
            "query": "What pasta should I buy?",
            "session": {
                "session_id": "test-session",
                "namespace": "test-namespace",
            },
            "long_term_search": {
                "text": "pasta preferences",
            },
        }

        # Call the endpoint
        response = await client.post("/v1/memory/prompt", json=payload)

        # Check status code
        assert response.status_code == 200

        # Check response data
        data = response.json()
        assert isinstance(data, dict)
        assert (
            len(data["messages"]) == 5
        )  # Context + 2 session messages + long-term memory + query

        # Verify the messages content (order matters)
        assert data["messages"][0]["role"] == "system"
        assert "Cooking conversation" in data["messages"][0]["content"]["text"]
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"]["text"] == "How do you make pasta?"
        assert data["messages"][2]["role"] == "assistant"
        assert (
            data["messages"][2]["content"]["text"]
            == "Boil water, add pasta, cook until al dente."
        )
        assert data["messages"][3]["role"] == "system"
        assert "Long term memories" in data["messages"][3]["content"]["text"]
        assert (
            "User prefers gluten-free pasta" in data["messages"][3]["content"]["text"]
        )
        assert data["messages"][4]["role"] == "user"
        assert data["messages"][4]["content"]["text"] == "What pasta should I buy?"

    @pytest.mark.asyncio
    async def test_memory_prompt_without_required_params(self, client):
        """Test the memory_prompt endpoint without required parameters"""
        # Call the endpoint without session or long_term_search
        response = await client.post("/v1/memory/prompt", json={"query": "test"})

        # Check status code (should be 400 Bad Request)
        assert response.status_code == 400

        # Check error message
        data = response.json()
        assert "detail" in data
        assert "Either session or long_term_search must be provided" in data["detail"]

    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @pytest.mark.asyncio
    async def test_memory_prompt_session_not_found(
        self, mock_get_working_memory, client
    ):
        """Test the memory_prompt endpoint when session is not found"""
        # Mock the session memory to return None (session not found)
        mock_get_working_memory.return_value = None

        # Call the endpoint
        query = "What's the weather like?"
        response = await client.post(
            "/v1/memory/prompt",
            json={
                "query": query,
                "session": {
                    "session_id": "nonexistent-session",
                    "namespace": "test-namespace",
                },
            },
        )

        # Check status code (should be successful)
        assert response.status_code == 200

        # Check response data (should only contain the query)
        data = response.json()
        assert isinstance(data, dict)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"]["text"] == query

    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @patch("agent_memory_server.api.get_model_config")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_model_name(
        self, mock_get_model_config, mock_get_working_memory, client
    ):
        """Test the memory_prompt endpoint with model_name parameter"""
        # Mock the model config
        model_config = MagicMock()
        model_config.max_tokens = 4000
        mock_get_model_config.return_value = model_config

        # Mock the session memory
        mock_session_memory = WorkingMemoryResponse(
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there"),
            ],
            memories=[],
            session_id="test-session",
            context="Previous context",
            tokens=150,
        )
        mock_get_working_memory.return_value = mock_session_memory

        # Call the endpoint with model_name
        query = "What's the weather like?"
        response = await client.post(
            "/v1/memory/prompt",
            json={
                "query": query,
                "session": {
                    "session_id": "test-session",
                    "model_name": "gpt-4o",
                },
            },
        )

        # Check the model config was used
        mock_get_model_config.assert_called_once_with("gpt-4o")

        # Check status code
        assert response.status_code == 200

        # Verify the working memory function was called
        mock_get_working_memory.assert_called_once()


@pytest.mark.requires_api_keys
class TestLongTermMemoryEndpoint:
    @pytest.mark.asyncio
    async def test_create_long_term_memory_with_valid_id(self, client):
        """Test creating long-term memory with valid id"""
        payload = {
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "user_id": "user123",
                    "session_id": "session123",
                    "namespace": "test",
                    "memory_type": "semantic",
                    "id": "test-client-123",
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_create_long_term_memory_missing_id(self, client):
        """Test creating long-term memory without id should fail"""
        payload = {
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "user_id": "user123",
                    "session_id": "session123",
                    "namespace": "test",
                    "memory_type": "semantic",
                    # Missing id field
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "Field required" in str(data["detail"])

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_create_long_term_memory_persisted_at_ignored(self, client):
        """Test that client-provided persisted_at is ignored"""
        payload = {
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "id": "test-client-456",
                    "memory_type": "semantic",
                    "persisted_at": "2023-01-01T00:00:00Z",  # Use ISO string instead of datetime object
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.requires_api_keys
class TestUnifiedSearchEndpoint:
    @patch("agent_memory_server.api.long_term_memory.search_memories")
    @pytest.mark.asyncio
    async def test_unified_search(self, mock_search, client):
        """Test the unified search endpoint"""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=3,
            memories=[
                MemoryRecordResult(
                    id="working-1",
                    text="Working memory: User prefers dark mode",
                    dist=0.0,
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    persisted_at=None,  # Working memory
                ),
                MemoryRecordResult(
                    id="long-1",
                    text="Long-term: User likes coffee",
                    dist=0.25,
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    persisted_at=datetime(2023, 1, 1, 0, 0, 0),  # Long-term memory
                ),
                MemoryRecordResult(
                    id="long-2",
                    text="Long-term: User is allergic to peanuts",
                    dist=0.35,
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    persisted_at=datetime(2023, 1, 1, 1, 0, 0),  # Long-term memory
                ),
            ],
            next_offset=None,
        )

        # Create payload
        payload = {"text": "What are the user's preferences?"}

        # Call the unified search endpoint
        response = await client.post("/v1/memory/search", json=payload)

        # Check status code
        assert response.status_code == 200, response.text

        # Check response structure
        data = response.json()
        assert "memories" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["memories"]) == 3

        # Check that results include both working and long-term memory
        memories = data["memories"]

        # First result should be working memory (dist=0.0)
        assert memories[0]["id"] == "working-1"
        assert "Working memory" in memories[0]["text"]
        assert memories[0]["dist"] == 0.0
        assert memories[0]["persisted_at"] is None

        # Other results should be long-term memory
        assert memories[1]["id"] == "long-1"
        assert "Long-term" in memories[1]["text"]
        assert memories[1]["dist"] == 0.25
        assert memories[1]["persisted_at"] is not None

        assert memories[2]["id"] == "long-2"
        assert "Long-term" in memories[2]["text"]
        assert memories[2]["dist"] == 0.35
        assert memories[2]["persisted_at"] is not None

    @patch("agent_memory_server.api.long_term_memory.search_memories")
    @pytest.mark.asyncio
    async def test_unified_search_with_filters(self, mock_search, client):
        """Test the unified search endpoint with filters"""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="filtered-1",
                    text="User's semantic preference",
                    dist=0.1,
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    user_id="test-user",
                    session_id="test-session",
                ),
            ],
            next_offset=None,
        )

        # Create payload with filters
        payload = {
            "text": "preferences",
            "memory_type": {"eq": "semantic"},
            "user_id": {"eq": "test-user"},
            "session_id": {"eq": "test-session"},
            "limit": 5,
        }

        # Call the unified search endpoint
        response = await client.post("/v1/memory/search", json=payload)

        # Check status code
        assert response.status_code == 200

        # Verify the mock was called with correct parameters
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["text"] == "preferences"
        assert call_kwargs["limit"] == 5

        # Check response
        data = response.json()
        assert data["total"] == 1
        assert len(data["memories"]) == 1
        assert data["memories"][0]["memory_type"] == "semantic"
        assert data["memories"][0]["user_id"] == "test-user"
