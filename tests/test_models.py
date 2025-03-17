import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from models import (
    MemoryMessage,
    MemoryMessagesAndContext,
    MemoryResponse,
    ModelClientFactory,
    ModelProvider,
    OpenAIClientWrapper,
    RedisearchResult,
    SearchPayload,
    get_model_config,
)


class TestModels:
    def test_memory_message(self):
        """Test MemoryMessage model"""
        msg = MemoryMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

        # Test serialization
        data = msg.model_dump()
        assert data == {"role": "user", "content": "Hello, world!"}

    def test_memory_messages_and_context(self):
        """Test MemoryMessagesAndContext model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test without context
        payload = MemoryMessagesAndContext(messages=messages)
        assert payload.messages == messages
        assert payload.context is None

        # Test with context
        payload = MemoryMessagesAndContext(
            messages=messages, context="Previous conversation summary"
        )
        assert payload.messages == messages
        assert payload.context == "Previous conversation summary"

    def test_memory_response(self):
        """Test MemoryResponse model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test basic response
        response = MemoryResponse(messages=messages)
        assert response.messages == messages
        assert response.context is None
        assert response.tokens is None

        # Test with all fields
        response = MemoryResponse(
            messages=messages, context="Conversation summary", tokens=150
        )
        assert response.messages == messages
        assert response.context == "Conversation summary"
        assert response.tokens == 150

    def test_search_payload(self):
        """Test SearchPayload model"""
        payload = SearchPayload(text="What is the capital of France?")
        assert payload.text == "What is the capital of France?"

    def test_redisearch_result(self):
        """Test RedisearchResult model"""
        result = RedisearchResult(
            role="assistant", content="Paris is the capital of France", dist=0.75
        )
        assert result.role == "assistant"
        assert result.content == "Paris is the capital of France"
        assert result.dist == 0.75


@pytest.mark.asyncio()
class TestOpenAIClientWrapper:
    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
        },
    )
    @patch("models.AsyncOpenAI")
    async def test_init_regular_openai(self, mock_openai):
        """Test initializing with regular OpenAI"""
        # Set up the mock to return an AsyncMock
        mock_openai.return_value = AsyncMock()

        client = OpenAIClientWrapper()

        # Verify the client was created
        assert mock_openai.called

    @patch.object(OpenAIClientWrapper, "__init__", return_value=None)
    async def test_create_embedding(self, mock_init):
        """Test creating embeddings"""
        # Create a client with mocked init
        client = OpenAIClientWrapper()

        # Mock the embedding client and response
        mock_response = AsyncMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]

        client.embedding_client = AsyncMock()
        client.embedding_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        # Test creating embeddings
        query_vec = ["Hello, world!", "How are you?"]
        embeddings = await client.create_embedding(query_vec)

        # Verify embeddings were created correctly
        assert len(embeddings) == 2
        # Convert NumPy array to list or use np.array_equal for comparison
        assert np.array_equal(
            embeddings[0], np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        assert np.array_equal(
            embeddings[1], np.array([0.4, 0.5, 0.6], dtype=np.float32)
        )

        # Verify the client was called with correct parameters
        client.embedding_client.embeddings.create.assert_called_with(
            model="text-embedding-ada-002", input=query_vec
        )

    @patch.object(OpenAIClientWrapper, "__init__", return_value=None)
    async def test_create_chat_completion(self, mock_init):
        """Test creating chat completions"""
        # Create a client with mocked init
        client = OpenAIClientWrapper()

        # Mock the completion client and response
        # Create a response structure that matches our new ChatResponse format
        mock_response = AsyncMock()
        mock_response.choices = [{"message": {"content": "Test response"}}]
        mock_response.usage = {"total_tokens": 100}

        client.completion_client = AsyncMock()
        client.completion_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Test creating chat completion
        model = "gpt-3.5-turbo"
        prompt = "Hello, world!"
        response = await client.create_chat_completion(model, prompt)

        # Verify the response contains the expected structure
        assert response.choices[0]["message"]["content"] == "Test response"
        assert response.total_tokens == 100

        # Verify the client was called with correct parameters
        client.completion_client.chat.completions.create.assert_called_with(
            model=model, messages=[{"role": "user", "content": prompt}]
        )


@pytest.mark.parametrize(
    "model_name,expected_provider,expected_max_tokens",
    [
        ("gpt-4o", "openai", 128000),
        ("claude-3-sonnet-20240229", "anthropic", 200000),
        ("nonexistent-model", "openai", 128000),  # Should default to GPT-4o-mini
    ],
)
def test_get_model_config(model_name, expected_provider, expected_max_tokens):
    """Test the get_model_config function"""
    # Get the model config
    config = get_model_config(model_name)

    # Check the provider
    if expected_provider == "openai":
        assert config.provider == ModelProvider.OPENAI
    else:
        assert config.provider == ModelProvider.ANTHROPIC

    # Check the max tokens
    assert config.max_tokens == expected_max_tokens


@pytest.mark.asyncio()
async def test_model_client_factory():
    """Test the ModelClientFactory"""
    # Test with OpenAI model
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("models.OpenAIClientWrapper") as mock_openai:
            mock_openai.return_value = "openai-client"
            client = await ModelClientFactory.get_client("gpt-4")
            assert client == "openai-client"

    # Test with Anthropic model
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("models.AnthropicClientWrapper") as mock_anthropic:
            mock_anthropic.return_value = "anthropic-client"
            client = await ModelClientFactory.get_client("claude-3-sonnet-20240229")
            assert client == "anthropic-client"
