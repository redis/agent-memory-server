"""
Unit tests for the LLMClient facade.

These tests use a mock backend to verify the facade's behavior
without making actual API calls.
"""

import pytest

from agent_memory_server.llm import (
    ChatCompletionResponse,
    EmbeddingResponse,
    LLMClient,
)


class MockLLMBackend:
    """Mock backend for testing LLMClient."""

    def __init__(
        self,
        chat_response: ChatCompletionResponse | None = None,
        embedding_response: EmbeddingResponse | None = None,
    ):
        self.chat_response = chat_response or ChatCompletionResponse(
            content="Mock response",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="mock-model",
        )
        self.embedding_response = embedding_response or EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            total_tokens=5,
            model="mock-embedding-model",
        )
        self.chat_calls: list[dict] = []
        self.embedding_calls: list[dict] = []

    async def create_chat_completion(self, **kwargs) -> ChatCompletionResponse:
        self.chat_calls.append(kwargs)
        return self.chat_response

    async def create_embedding(self, **kwargs) -> EmbeddingResponse:
        self.embedding_calls.append(kwargs)
        return self.embedding_response


@pytest.fixture(autouse=True)
def reset_llm_client():
    """Reset LLMClient state before and after each test."""
    LLMClient.reset()
    yield
    LLMClient.reset()


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse Pydantic model."""

    def test_frozen_model(self):
        """ChatCompletionResponse should be immutable."""
        from pydantic import ValidationError

        response = ChatCompletionResponse(
            content="test",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="gpt-4o",
        )
        with pytest.raises(ValidationError, match="frozen"):
            response.content = "modified"

    def test_default_raw_response(self):
        """raw_response should default to None."""
        response = ChatCompletionResponse(
            content="test",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="gpt-4o",
        )
        assert response.raw_response is None


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse Pydantic model."""

    def test_frozen_model(self):
        """EmbeddingResponse should be immutable."""
        from pydantic import ValidationError

        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2]],
            total_tokens=5,
            model="text-embedding-3-small",
        )
        with pytest.raises(ValidationError, match="frozen"):
            response.model = "modified"


# NOTE: TestLLMClientModelNameMapping was removed because LiteLLM now handles
# provider detection automatically. The MODEL_NAME_MAP and _resolve_model_name()
# method were removed as part of the LiteLLM integration simplification.
# See dev_docs/litellm_redundancy_analysis.md for details.


# TODO: Gateway tests disabled pending integration test setup.
class TestLLMClientReset:
    """Tests for reset functionality."""

    def test_reset_clears_backend(self):
        """reset() should clear the backend."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)
        LLMClient.reset()
        assert LLMClient._backend is None


class TestLLMClientChatCompletion:
    """Tests for create_chat_completion method."""

    @pytest.mark.asyncio
    async def test_basic_chat_completion(self):
        """Basic chat completion should work with mock backend."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        response = await LLMClient.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.content == "Mock response"
        assert response.finish_reason == "stop"
        assert response.total_tokens == 15
        assert len(mock.chat_calls) == 1
        assert mock.chat_calls[0]["model"] == "gpt-4o"
        assert mock.chat_calls[0]["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_chat_completion_with_parameters(self):
        """Chat completion should pass all parameters to backend."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        await LLMClient.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
            response_format={"type": "json_object"},
            api_base="https://custom.api.com",
            api_key="custom-key",
        )

        call = mock.chat_calls[0]
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 100
        assert call["response_format"] == {"type": "json_object"}
        assert call["api_base"] == "https://custom.api.com"
        assert call["api_key"] == "custom-key"

    @pytest.mark.asyncio
    async def test_chat_completion_kwargs_passthrough(self):
        """Extra kwargs should be passed through to backend."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        await LLMClient.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )

        call = mock.chat_calls[0]
        assert call["tools"] == [{"type": "function", "function": {"name": "test"}}]
        assert call["tool_choice"] == "auto"


class TestLLMClientEmbedding:
    """Tests for create_embedding method."""

    @pytest.mark.asyncio
    async def test_basic_embedding(self):
        """Basic embedding should work with mock backend."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        response = await LLMClient.create_embedding(
            model="text-embedding-3-small",
            input_texts=["Hello world"],
        )

        assert response.embeddings == [[0.1, 0.2, 0.3]]
        assert response.total_tokens == 5
        assert len(mock.embedding_calls) == 1
        assert mock.embedding_calls[0]["model"] == "text-embedding-3-small"
        assert mock.embedding_calls[0]["input_texts"] == ["Hello world"]

    @pytest.mark.asyncio
    async def test_embedding_with_custom_endpoint(self):
        """Embedding should support custom API endpoints."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        await LLMClient.create_embedding(
            model="text-embedding-3-small",
            input_texts=["Hello"],
            api_base="https://custom.api.com",
            api_key="custom-key",
        )

        call = mock.embedding_calls[0]
        assert call["api_base"] == "https://custom.api.com"
        assert call["api_key"] == "custom-key"

    @pytest.mark.asyncio
    async def test_embedding_multiple_texts(self):
        """Embedding should handle multiple input texts."""
        mock = MockLLMBackend(
            embedding_response=EmbeddingResponse(
                embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                total_tokens=15,
                model="text-embedding-3-small",
            )
        )
        LLMClient.set_backend(mock)

        response = await LLMClient.create_embedding(
            model="text-embedding-3-small",
            input_texts=["Text 1", "Text 2", "Text 3"],
        )

        assert len(response.embeddings) == 3
        assert response.total_tokens == 15


class TestLLMClientBackendInjection:
    """Tests for backend injection (testing utilities)."""

    @pytest.mark.asyncio
    async def test_set_backend(self):
        """set_backend should inject custom backend."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        await LLMClient.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
        )

        assert len(mock.chat_calls) == 1

    @pytest.mark.asyncio
    async def test_backend_receives_all_params(self):
        """Backend should receive all parameters passed to LLMClient."""
        mock = MockLLMBackend()
        LLMClient.set_backend(mock)

        await LLMClient.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=50,
            custom_param="custom_value",
        )

        call = mock.chat_calls[0]
        assert call["model"] == "gpt-4o"
        assert call["temperature"] == 0.7
        assert call["max_tokens"] == 50
        assert call["custom_param"] == "custom_value"


# =============================================================================
# Integration Tests (require API keys)
# =============================================================================
# Run with: uv run pytest tests/test_llm_client.py --run-api-tests -v


@pytest.mark.requires_api_keys
class TestLLMClientIntegrationOpenAI:
    """Integration tests for LLMClient with real OpenAI API calls.

    These tests require OPENAI_API_KEY to be set in the environment.
    Run with: uv run pytest tests/test_llm_client.py --run-api-tests
    """

    @pytest.mark.asyncio
    async def test_openai_chat_completion(self):
        """Test real OpenAI chat completion."""
        response = await LLMClient.create_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            temperature=0.0,
            max_tokens=10,
        )

        assert response.content is not None
        assert len(response.content) > 0
        assert "hello" in response.content.lower()
        assert response.finish_reason in ("stop", "length")
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert (
            response.total_tokens == response.prompt_tokens + response.completion_tokens
        )
        assert "gpt-4o-mini" in response.model

    @pytest.mark.asyncio
    async def test_openai_chat_completion_json_mode(self):
        """Test OpenAI chat completion with JSON response format."""
        response = await LLMClient.create_chat_completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with a single key 'greeting' and value 'hello'.",
                }
            ],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"},
        )

        assert response.content is not None
        # Verify it's valid JSON
        import json

        parsed = json.loads(response.content)
        assert "greeting" in parsed
        assert parsed["greeting"].lower() == "hello"

    @pytest.mark.asyncio
    async def test_openai_embedding(self):
        """Test real OpenAI embedding creation."""
        response = await LLMClient.create_embedding(
            model="text-embedding-3-small",
            input_texts=["Hello, world!", "This is a test."],
        )

        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) > 0  # Has dimensions
        assert len(response.embeddings[1]) > 0
        assert response.total_tokens > 0
        assert "text-embedding-3-small" in response.model

    @pytest.mark.asyncio
    async def test_openai_embedding_single_text(self):
        """Test OpenAI embedding with single text input."""
        response = await LLMClient.create_embedding(
            model="text-embedding-3-small",
            input_texts=["Single text for embedding"],
        )

        assert len(response.embeddings) == 1
        # text-embedding-3-small has 1536 dimensions
        assert len(response.embeddings[0]) == 1536


@pytest.mark.requires_api_keys
class TestLLMClientIntegrationModelMapping:
    """Integration tests verifying model name mapping works with real APIs."""

    @pytest.mark.asyncio
    async def test_model_name_without_prefix(self):
        """Test that model names without provider prefix work correctly."""
        # Using "gpt-4o-mini" should be mapped to "openai/gpt-4o-mini"
        response = await LLMClient.create_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' only."}],
            temperature=0.0,
            max_tokens=5,
        )

        assert response.content is not None
        assert "gpt-4o-mini" in response.model

    @pytest.mark.asyncio
    async def test_model_name_with_prefix(self):
        """Test that model names with provider prefix work correctly."""
        # Using "openai/gpt-4o-mini" should pass through unchanged
        response = await LLMClient.create_chat_completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' only."}],
            temperature=0.0,
            max_tokens=5,
        )

        assert response.content is not None
        assert "gpt-4o-mini" in response.model


@pytest.mark.requires_api_keys
class TestLLMClientIntegrationErrorHandling:
    """Integration tests for error handling with real API calls."""

    @pytest.mark.asyncio
    async def test_invalid_model_raises_error(self):
        """Test that invalid model names raise appropriate errors."""
        import litellm

        with pytest.raises(litellm.exceptions.NotFoundError):
            await LLMClient.create_chat_completion(
                model="openai/nonexistent-model-xyz",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_empty_messages_raises_error(self):
        """Test that empty messages list raises an error."""
        import litellm

        with pytest.raises(litellm.exceptions.BadRequestError):
            await LLMClient.create_chat_completion(
                model="gpt-4o-mini",
                messages=[],
            )
