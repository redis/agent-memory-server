from unittest.mock import AsyncMock, patch

import pytest

from agent_memory_server.config import ModelProvider
from agent_memory_server.llm import (
    ChatCompletionResponse,
    LLMClient,
    get_model_config,
    optimize_query_for_vector_search,
)


@pytest.mark.parametrize(
    ("model_name", "expected_provider", "expected_max_tokens"),
    [
        ("gpt-4o", "openai", 128000),
        ("claude-3-sonnet-20240229", "anthropic", 200000),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "aws-bedrock", 200000),
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
    elif expected_provider == "anthropic":
        assert config.provider == ModelProvider.ANTHROPIC
    elif expected_provider == "aws-bedrock":
        assert config.provider == ModelProvider.AWS_BEDROCK

    # Check the max tokens
    assert config.max_tokens == expected_max_tokens


def test_get_model_config_via_llmclient():
    """Test that LLMClient.get_model_config works correctly."""
    config = LLMClient.get_model_config("gpt-4o")
    assert config.provider == ModelProvider.OPENAI
    assert config.max_tokens == 128000

    # Test fallback for unknown model
    config = LLMClient.get_model_config("unknown-model")
    assert config.provider == ModelProvider.OPENAI  # Defaults to gpt-4o-mini


@pytest.mark.asyncio
class TestQueryOptimization:
    """Test query optimization functionality."""

    async def test_optimize_query_success(self):
        """Test successful query optimization."""
        mock_response = ChatCompletionResponse(
            content="user interface preferences dark mode",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await optimize_query_for_vector_search(
                "Can you tell me about my UI preferences for dark mode?"
            )

            assert result == "user interface preferences dark mode"
            mock_create.assert_called_once()

    async def test_optimize_query_with_custom_model(self):
        """Test query optimization with custom model."""
        mock_response = ChatCompletionResponse(
            content="optimized query",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="custom-model",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await optimize_query_for_vector_search(
                "original query", model_name="custom-model"
            )

            assert result == "optimized query"
            mock_create.assert_called_once()
            # Verify the model name was passed to create_chat_completion
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "custom-model"

    @patch("agent_memory_server.config.settings")
    async def test_optimize_query_uses_fast_model_default(self, mock_settings):
        """Test that optimization uses fast_model by default."""
        mock_settings.fast_model = "gpt-4o-mini"
        mock_settings.query_optimization_prompt_template = "Optimize: {query}"
        mock_settings.min_optimized_query_length = 3

        mock_response = ChatCompletionResponse(
            content="optimized",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            await optimize_query_for_vector_search("test query")

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"

    async def test_optimize_query_empty_input(self):
        """Test optimization with empty or None input."""
        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
        ) as mock_create:
            # Test empty string
            result = await optimize_query_for_vector_search("")
            assert result == ""
            mock_create.assert_not_called()

            # Test whitespace only
            result = await optimize_query_for_vector_search("   ")
            assert result == "   "
            mock_create.assert_not_called()

    async def test_optimize_query_client_error_fallback(self):
        """Test fallback to original query when client fails."""
        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            side_effect=Exception("Model client error"),
        ) as mock_create:
            original_query = "What are my preferences?"
            result = await optimize_query_for_vector_search(original_query)

            assert result == original_query
            mock_create.assert_called_once()

    async def test_optimize_query_empty_response_fallback(self):
        """Test fallback when model returns empty response."""
        mock_response = ChatCompletionResponse(
            content="",  # Empty response
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=0,
            total_tokens=50,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            original_query = "What are my preferences?"
            result = await optimize_query_for_vector_search(original_query)

            assert result == original_query

    async def test_optimize_query_short_response_fallback(self):
        """Test fallback when model returns very short response."""
        mock_response = ChatCompletionResponse(
            content="a",  # Too short
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=1,
            total_tokens=51,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            original_query = "What are my preferences?"
            result = await optimize_query_for_vector_search(original_query)

            assert result == original_query

    async def test_optimize_query_none_content_fallback(self):
        """Test fallback when model response has None content."""
        mock_response = ChatCompletionResponse(
            content=None,  # None content
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=0,
            total_tokens=50,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            original_query = "What are my preferences?"
            result = await optimize_query_for_vector_search(original_query)

            assert result == original_query

    async def test_optimize_query_strips_whitespace(self):
        """Test that optimization strips whitespace from response."""
        mock_response = ChatCompletionResponse(
            content="  optimized query  \n",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await optimize_query_for_vector_search("test query")
            assert result == "optimized query"

    async def test_optimize_query_prompt_format(self):
        """Test that the optimization prompt is correctly formatted."""
        mock_response = ChatCompletionResponse(
            content="optimized",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm.client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            test_query = "Can you tell me about user preferences?"
            await optimize_query_for_vector_search(test_query)

            # Check that the prompt contains our test query
            call_kwargs = mock_create.call_args[1]
            prompt = call_kwargs["messages"][0]["content"]
            assert test_query in prompt
            assert "semantic search" in prompt
            assert "Guidelines:" in prompt
            assert "Optimized query:" in prompt
