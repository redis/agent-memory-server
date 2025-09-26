"""
Unit tests for context percentage calculation functions.
Includes regression tests for division by zero and edge cases.
"""

from agent_memory_server.api import _calculate_context_usage_percentages
from agent_memory_server.models import MemoryMessage


class TestContextPercentageCalculation:
    """Test context percentage calculation in various scenarios"""

    def test_context_percentages_with_context_window_max(self):
        """Test that context percentages are calculated when context_window_max is provided"""
        messages = [
            MemoryMessage(role="user", content="Hello, how are you today?"),
            MemoryMessage(
                role="assistant", content="I'm doing well, thank you for asking!"
            ),
            MemoryMessage(
                role="user",
                content="That's great to hear. Can you help me with something?",
            ),
        ]

        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=500
            )
        )

        assert (
            total_percentage is not None
        ), "total_percentage should not be None when context_window_max is provided"
        assert (
            until_summarization_percentage is not None
        ), "until_summarization_percentage should not be None when context_window_max is provided"
        assert isinstance(total_percentage, float), "total_percentage should be a float"
        assert isinstance(
            until_summarization_percentage, float
        ), "until_summarization_percentage should be a float"
        assert (
            0 <= total_percentage <= 100
        ), "total_percentage should be between 0 and 100"
        assert (
            0 <= until_summarization_percentage <= 100
        ), "until_summarization_percentage should be between 0 and 100"

    def test_context_percentages_with_model_name(self):
        """Test that context percentages are calculated when model_name is provided"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name="gpt-4o-mini", context_window_max=None
            )
        )

        assert (
            total_percentage is not None
        ), "total_percentage should not be None when model_name is provided"
        assert (
            until_summarization_percentage is not None
        ), "until_summarization_percentage should not be None when model_name is provided"
        assert isinstance(total_percentage, float), "total_percentage should be a float"
        assert isinstance(
            until_summarization_percentage, float
        ), "until_summarization_percentage should be a float"

    def test_context_percentages_without_model_info(self):
        """Test that context percentages return None when no model info is provided"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=None
            )
        )

        assert (
            total_percentage is None
        ), "total_percentage should be None when no model info is provided"
        assert (
            until_summarization_percentage is None
        ), "until_summarization_percentage should be None when no model info is provided"

    def test_context_percentages_with_empty_messages(self):
        """Test context percentages with empty messages list but model info provided"""
        messages = []

        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=500
            )
        )

        # CORRECTED: Should return 0.0 when model info is provided, even with empty messages
        assert (
            total_percentage == 0.0
        ), "total_percentage should be 0.0 for empty messages when model info provided"
        assert (
            until_summarization_percentage == 0.0
        ), "until_summarization_percentage should be 0.0 for empty messages when model info provided"

    def test_context_percentages_precedence(self):
        """Test that context_window_max takes precedence over model_name"""
        messages = [
            MemoryMessage(role="user", content="Hello world"),
        ]

        # Test with both provided - context_window_max should take precedence
        total_percentage_both, until_summarization_percentage_both = (
            _calculate_context_usage_percentages(
                messages=messages,
                model_name="gpt-4o-mini",  # This has a large context window
                context_window_max=100,  # This is much smaller
            )
        )

        # Test with only context_window_max
        total_percentage_max_only, until_summarization_percentage_max_only = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=100
            )
        )

        # Results should be the same, proving context_window_max takes precedence
        assert (
            total_percentage_both == total_percentage_max_only
        ), "context_window_max should take precedence over model_name"
        assert (
            until_summarization_percentage_both
            == until_summarization_percentage_max_only
        ), "context_window_max should take precedence over model_name"

    def test_context_percentages_high_token_usage(self):
        """Test context percentages when token usage is high"""
        # Create many messages to exceed typical limits
        messages = []
        for i in range(50):
            messages.append(
                MemoryMessage(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"This is message number {i} with substantial content that will use many tokens. "
                    * 10,
                )
            )

        # Test with small context window to force high percentages
        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=500
            )
        )

        assert total_percentage is not None
        assert until_summarization_percentage is not None
        # Should be capped at 100%
        assert total_percentage <= 100.0, "total_percentage should be capped at 100%"
        assert (
            until_summarization_percentage <= 100.0
        ), "until_summarization_percentage should be capped at 100%"

    def test_context_percentages_zero_context_window_regression(self):
        """
        Regression test for division by zero when context_window_max is 0 or very small.

        Bug: When max_tokens <= 0 or token_threshold <= 0, division by zero occurred.
        Fix: Added checks to return None for invalid context windows.
        """
        messages = [MemoryMessage(role="user", content="Hello")]

        # Test with zero context window
        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=0
            )
        )

        # Should return None for invalid context window
        assert total_percentage is None, "Should return None for zero context window"
        assert (
            until_summarization_percentage is None
        ), "Should return None for zero context window"

        # Test with negative context window
        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages, model_name=None, context_window_max=-1
            )
        )

        # Should return None for invalid context window
        assert (
            total_percentage is None
        ), "Should return None for negative context window"
        assert (
            until_summarization_percentage is None
        ), "Should return None for negative context window"

    def test_context_percentages_very_small_context_window_regression(self):
        """
        Regression test for division by zero when token_threshold becomes 0.

        Bug: When context_window_max is very small (e.g., 1) and summarization_threshold is 0.7,
        token_threshold = int(1 * 0.7) = 0, causing division by zero.
        Fix: Added check for token_threshold <= 0.
        """
        messages = [MemoryMessage(role="user", content="Hello world")]

        # Test with very small context window that would cause token_threshold = 0
        total_percentage, until_summarization_percentage = (
            _calculate_context_usage_percentages(
                messages=messages,
                model_name=None,
                context_window_max=1,  # With summarization_threshold=0.7, token_threshold = int(1 * 0.7) = 0
            )
        )

        # Should handle this gracefully without division by zero
        assert (
            total_percentage is not None
        ), "Should handle small context window without error"
        assert (
            until_summarization_percentage is not None
        ), "Should handle small context window without error"
        assert isinstance(total_percentage, float), "Should return valid float"
        assert isinstance(
            until_summarization_percentage, float
        ), "Should return valid float"
        # until_summarization_percentage should be 100% when threshold is 0
        assert (
            until_summarization_percentage == 100.0
        ), "Should return 100% when token threshold is 0"
