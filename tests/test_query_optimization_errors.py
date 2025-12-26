"""
Test error handling and edge cases for query optimization feature.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agent_memory_server.llm_client import (
    ChatCompletionResponse,
    optimize_query_for_vector_search,
)
from agent_memory_server.long_term_memory import search_long_term_memories
from agent_memory_server.models import MemoryRecordResults


@pytest.mark.asyncio
class TestQueryOptimizationErrorHandling:
    """Test error handling scenarios for query optimization."""

    VERY_LONG_QUERY_REPEAT_COUNT = 1000

    async def test_optimization_with_network_timeout(self):
        """Test graceful fallback when model API times out."""
        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Request timed out"),
        ) as mock_create:
            original_query = "Can you tell me about my settings?"
            result = await optimize_query_for_vector_search(original_query)

            # Should fall back to original query
            assert result == original_query
            mock_create.assert_called_once()

    async def test_optimization_with_invalid_api_key(self):
        """Test fallback when API key is invalid."""
        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            side_effect=Exception("Invalid API key"),
        ):
            original_query = "What are my preferences?"
            result = await optimize_query_for_vector_search(original_query)

            # Should fall back to original query
            assert result == original_query

    async def test_optimization_with_none_content_response(self):
        """Test handling when model returns None content."""
        mock_response = ChatCompletionResponse(
            content=None,
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=0,
            total_tokens=50,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            original_query = "Find my user settings"
            result = await optimize_query_for_vector_search(original_query)

            # Should fall back to original query
            assert result == original_query

    async def test_optimization_with_empty_response(self):
        """Test handling when model returns empty content."""
        mock_response = ChatCompletionResponse(
            content="",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=0,
            total_tokens=50,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            original_query = "Show my preferences"
            result = await optimize_query_for_vector_search(original_query)

            # Should fall back to original query
            assert result == original_query

    async def test_optimization_with_unicode_query(self):
        """Test optimization with unicode and special characters."""
        mock_response = ChatCompletionResponse(
            content="préférences utilisateur émojis 🎉",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            unicode_query = "Mes préférences avec émojis 🎉 et caractères spéciaux"
            result = await optimize_query_for_vector_search(unicode_query)

            assert result == "préférences utilisateur émojis 🎉"
            mock_create.assert_called_once()

    async def test_optimization_with_very_long_query(self):
        """Test optimization with extremely long queries."""
        mock_response = ChatCompletionResponse(
            content="long query optimized",
            finish_reason="stop",
            prompt_tokens=500,
            completion_tokens=10,
            total_tokens=510,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            # Create a very long query (10,000 characters)
            long_query = (
                "Tell me about "
                + "preferences " * self.VERY_LONG_QUERY_REPEAT_COUNT
                + "settings"
            )
            result = await optimize_query_for_vector_search(long_query)

            assert result == "long query optimized"
            mock_create.assert_called_once()

    async def test_optimization_preserves_query_intent(self):
        """Test that optimization preserves the core intent of queries."""
        mock_response = ChatCompletionResponse(
            content="user interface dark mode settings",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            original_query = (
                "Can you please tell me about my dark mode settings for the UI?"
            )
            result = await optimize_query_for_vector_search(original_query)

            assert result == "user interface dark mode settings"
            # Verify the prompt includes the original query
            call_kwargs = mock_create.call_args[1]
            prompt = call_kwargs["messages"][0]["content"]
            assert original_query in prompt

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_continues_when_optimization_fails(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that search continues even if optimization completely fails."""
        # Mock optimization to return original query (simulating internal error handling)
        mock_optimize.return_value = (
            "test query"  # The function handles errors internally
        )

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=0, memories=[]
        )
        mock_get_adapter.return_value = mock_adapter

        # This should not raise an exception
        await search_long_term_memories(
            text="test query", optimize_query=True, limit=10
        )

        # Verify optimization was attempted
        mock_optimize.assert_called_once()
        # Verify search still proceeded
        mock_adapter.search_memories.assert_called_once()

    async def test_optimization_handles_special_characters_in_response(self):
        """Test handling of special characters and formatting in model responses."""
        mock_response = ChatCompletionResponse(
            content="\n\n  **user preferences settings**  \n\n",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await optimize_query_for_vector_search("What are my settings?")

            # Should strip whitespace but preserve the content
            assert result == "**user preferences settings**"

    async def test_optimization_with_model_rate_limit(self):
        """Test fallback when model API is rate limited."""
        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            side_effect=Exception("Rate limit exceeded"),
        ):
            original_query = "Find my account settings"
            result = await optimize_query_for_vector_search(original_query)

            # Should fall back to original query
            assert result == original_query

    @patch("agent_memory_server.config.settings")
    async def test_optimization_with_invalid_model_name(self, mock_settings):
        """Test handling of invalid/unavailable model names."""
        # Set an invalid model name
        mock_settings.fast_model = "invalid-model-name"
        mock_settings.query_optimization_prompt_template = "Optimize: {query}"
        mock_settings.min_optimized_query_length = 3

        with patch(
            "agent_memory_server.llm_client.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            side_effect=Exception("Model not found"),
        ) as mock_create:
            original_query = "Show user preferences"
            result = await optimize_query_for_vector_search(original_query)

            # Should fall back to original query
            assert result == original_query
            mock_create.assert_called_once()
            # Verify the model name was passed
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "invalid-model-name"
