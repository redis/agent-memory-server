"""
Unit tests for the LLMClient facade.

These tests use standard mocking to verify the facade's behavior
without making actual API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.llm import (
    ChatCompletionResponse,
    EmbeddingResponse,
    LLMClient,
)


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


def _create_mock_litellm_chat_response(
    content: str = "Mock response",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "gpt-4o",
) -> MagicMock:
    """Create a mock LiteLLM chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content=content),
            finish_reason=finish_reason,
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    mock_response.model = model
    return mock_response


def _create_mock_litellm_embedding_response(
    embeddings: list[list[float]] | None = None,
    total_tokens: int = 5,
    model: str = "text-embedding-3-small",
) -> MagicMock:
    """Create a mock LiteLLM embedding response."""
    if embeddings is None:
        embeddings = [[0.1, 0.2, 0.3]]
    mock_response = MagicMock()
    # LiteLLM returns data as list of dicts with "embedding" key
    mock_response.data = [{"embedding": emb} for emb in embeddings]
    mock_response.usage = MagicMock(total_tokens=total_tokens)
    mock_response.model = model
    return mock_response


class TestLLMClientChatCompletion:
    """Tests for create_chat_completion method."""

    @pytest.mark.asyncio
    async def test_basic_chat_completion(self):
        """Basic chat completion should work with mocked LiteLLM."""
        mock_response = _create_mock_litellm_chat_response()

        with patch(
            "agent_memory_server.llm.client.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acompletion:
            response = await LLMClient.create_chat_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.content == "Mock response"
            assert response.finish_reason == "stop"
            assert response.total_tokens == 15
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_chat_completion_with_parameters(self):
        """Chat completion should pass all parameters to LiteLLM."""
        mock_response = _create_mock_litellm_chat_response()

        with patch(
            "agent_memory_server.llm.client.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acompletion:
            await LLMClient.create_chat_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.5,
                max_tokens=100,
                response_format={"type": "json_object"},
                api_base="https://custom.api.com",
                api_key="custom-key",
            )

            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["response_format"] == {"type": "json_object"}
            assert call_kwargs["api_base"] == "https://custom.api.com"
            assert call_kwargs["api_key"] == "custom-key"

    @pytest.mark.asyncio
    async def test_chat_completion_kwargs_passthrough(self):
        """Extra kwargs should be passed through to LiteLLM."""
        mock_response = _create_mock_litellm_chat_response()

        with patch(
            "agent_memory_server.llm.client.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acompletion:
            await LLMClient.create_chat_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                tools=[{"type": "function", "function": {"name": "test"}}],
                tool_choice="auto",
            )

            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["tools"] == [
                {"type": "function", "function": {"name": "test"}}
            ]
            assert call_kwargs["tool_choice"] == "auto"


class TestLLMClientEmbedding:
    """Tests for create_embedding method."""

    @pytest.mark.asyncio
    async def test_basic_embedding(self):
        """Basic embedding should work with mocked LiteLLM."""
        mock_response = _create_mock_litellm_embedding_response()

        with patch(
            "agent_memory_server.llm.client.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aembedding:
            response = await LLMClient.create_embedding(
                model="text-embedding-3-small",
                input_texts=["Hello world"],
            )

            assert response.embeddings == [[0.1, 0.2, 0.3]]
            assert response.total_tokens == 5
            mock_aembedding.assert_called_once()
            call_kwargs = mock_aembedding.call_args.kwargs
            assert call_kwargs["model"] == "text-embedding-3-small"
            assert call_kwargs["input"] == ["Hello world"]

    @pytest.mark.asyncio
    async def test_embedding_with_custom_endpoint(self):
        """Embedding should support custom API endpoints."""
        mock_response = _create_mock_litellm_embedding_response()

        with patch(
            "agent_memory_server.llm.client.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aembedding:
            await LLMClient.create_embedding(
                model="text-embedding-3-small",
                input_texts=["Hello"],
                api_base="https://custom.api.com",
                api_key="custom-key",
            )

            call_kwargs = mock_aembedding.call_args.kwargs
            assert call_kwargs["api_base"] == "https://custom.api.com"
            assert call_kwargs["api_key"] == "custom-key"

    @pytest.mark.asyncio
    async def test_embedding_multiple_texts(self):
        """Embedding should handle multiple input texts."""
        mock_response = _create_mock_litellm_embedding_response(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            total_tokens=15,
        )

        with patch(
            "agent_memory_server.llm.client.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await LLMClient.create_embedding(
                model="text-embedding-3-small",
                input_texts=["Text 1", "Text 2", "Text 3"],
            )

            assert len(response.embeddings) == 3
            assert response.total_tokens == 15


# =============================================================================
# Tests for create_embeddings and _map_provider
# =============================================================================


class TestCreateEmbeddings:
    """Tests for LLMClient.create_embeddings() factory method."""

    def test_unknown_embedding_model_raises_error(self):
        """create_embeddings should raise ModelValidationError for unknown models."""
        from unittest.mock import patch

        from agent_memory_server.llm.exceptions import ModelValidationError

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model = "unknown-embedding-model"
            mock_settings.embedding_model_config = None  # Unknown model returns None

            with pytest.raises(ModelValidationError, match="Unknown embedding model"):
                LLMClient.create_embeddings()

    def test_anthropic_embedding_raises_error(self):
        """create_embeddings should raise ModelValidationError for Anthropic models."""
        from unittest.mock import MagicMock, patch

        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.exceptions import ModelValidationError

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_config = MagicMock()
            mock_config.provider = ModelProvider.ANTHROPIC
            mock_settings.embedding_model = "claude-3-embedding"
            mock_settings.embedding_model_config = mock_config

            with pytest.raises(
                ModelValidationError,
                match="Anthropic does not provide embedding models",
            ):
                LLMClient.create_embeddings()

    def test_openai_embedding_returns_embeddings_instance(self):
        """create_embeddings should return OpenAIEmbeddings for OpenAI provider."""
        from unittest.mock import MagicMock, patch

        from agent_memory_server.config import ModelProvider

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_config = MagicMock()
            mock_config.provider = ModelProvider.OPENAI
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.embedding_model_config = mock_config
            mock_settings.openai_api_key = "test-key"

            embeddings = LLMClient.create_embeddings()

            # Should return a LangChain Embeddings instance
            from langchain_core.embeddings import Embeddings

            assert isinstance(embeddings, Embeddings)


class TestMapProvider:
    """Tests for LLMClient._map_provider() method."""

    def test_map_provider_openai(self):
        """_map_provider should map 'openai' to ModelProvider.OPENAI."""
        from agent_memory_server.config import ModelProvider

        result = LLMClient._map_provider("openai")
        assert result == ModelProvider.OPENAI

    def test_map_provider_anthropic(self):
        """_map_provider should map 'anthropic' to ModelProvider.ANTHROPIC."""
        from agent_memory_server.config import ModelProvider

        result = LLMClient._map_provider("anthropic")
        assert result == ModelProvider.ANTHROPIC

    def test_map_provider_bedrock(self):
        """_map_provider should map 'bedrock' to ModelProvider.AWS_BEDROCK."""
        from agent_memory_server.config import ModelProvider

        result = LLMClient._map_provider("bedrock")
        assert result == ModelProvider.AWS_BEDROCK

    def test_map_provider_azure_maps_to_openai(self):
        """_map_provider should map 'azure' to ModelProvider.OPENAI."""
        from agent_memory_server.config import ModelProvider

        result = LLMClient._map_provider("azure")
        assert result == ModelProvider.OPENAI

    def test_map_provider_unsupported_raises_error(self):
        """_map_provider should raise ModelValidationError for unsupported providers."""
        from agent_memory_server.llm.exceptions import ModelValidationError

        with pytest.raises(ModelValidationError, match="Unsupported LiteLLM provider"):
            LLMClient._map_provider("unsupported_provider")

    def test_map_provider_error_lists_supported_providers(self):
        """_map_provider error message should list supported providers."""
        from agent_memory_server.llm.exceptions import ModelValidationError

        with pytest.raises(ModelValidationError) as exc_info:
            LLMClient._map_provider("vertex_ai")

        error_message = str(exc_info.value)
        assert "openai" in error_message
        assert "anthropic" in error_message
        assert "bedrock" in error_message
        assert "azure" in error_message


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
