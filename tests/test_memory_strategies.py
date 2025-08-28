"""Tests for memory strategies functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_memory_server.memory_strategies import (
    MEMORY_STRATEGIES,
    BaseMemoryStrategy,
    CustomMemoryStrategy,
    DiscreteMemoryStrategy,
    SummaryMemoryStrategy,
    UserPreferencesMemoryStrategy,
    get_memory_strategy,
)


class TestBaseMemoryStrategy:
    """Test base memory strategy interface."""

    def test_base_strategy_methods_exist(self):
        """Test base strategy has required abstract methods."""
        # Check that the abstract methods exist
        assert hasattr(BaseMemoryStrategy, "extract_memories")
        assert hasattr(BaseMemoryStrategy, "get_extraction_description")
        assert hasattr(BaseMemoryStrategy, "get_strategy_name")

        # Test concrete strategy instantiation to verify base functionality
        strategy = DiscreteMemoryStrategy(test_param="test_value")
        assert strategy.config == {"test_param": "test_value"}
        assert strategy.get_strategy_name() == "DiscreteMemoryStrategy"


class TestMemoryStrategyFactory:
    """Test memory strategy factory function."""

    def test_get_strategy_discrete(self):
        """Test getting discrete memory strategy."""
        strategy = get_memory_strategy("discrete")
        assert isinstance(strategy, DiscreteMemoryStrategy)
        assert strategy.get_strategy_name() == "DiscreteMemoryStrategy"

    def test_get_strategy_summary(self):
        """Test getting summary memory strategy."""
        strategy = get_memory_strategy("summary")
        assert isinstance(strategy, SummaryMemoryStrategy)
        assert strategy.get_strategy_name() == "SummaryMemoryStrategy"

    def test_get_strategy_preferences(self):
        """Test getting preferences memory strategy."""
        strategy = get_memory_strategy("preferences")
        assert isinstance(strategy, UserPreferencesMemoryStrategy)
        assert strategy.get_strategy_name() == "UserPreferencesMemoryStrategy"

    def test_get_strategy_custom(self):
        """Test getting custom memory strategy with prompt."""
        custom_prompt = "Extract custom information: {message}"
        strategy = get_memory_strategy("custom", custom_prompt=custom_prompt)
        assert isinstance(strategy, CustomMemoryStrategy)
        assert strategy.custom_prompt == custom_prompt

    def test_get_strategy_custom_missing_prompt(self):
        """Test custom strategy raises error without prompt."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            get_memory_strategy("custom")

    def test_get_strategy_unknown(self):
        """Test getting unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown memory strategy 'unknown'"):
            get_memory_strategy("unknown")

    def test_get_strategy_with_config(self):
        """Test getting strategy with additional configuration."""
        strategy = get_memory_strategy("summary", max_summary_length=300)
        assert isinstance(strategy, SummaryMemoryStrategy)
        assert strategy.max_summary_length == 300


class TestDiscreteMemoryStrategy:
    """Test discrete memory strategy."""

    def test_extraction_description(self):
        """Test discrete strategy description."""
        strategy = DiscreteMemoryStrategy()
        description = strategy.get_extraction_description()
        assert "discrete semantic" in description.lower()
        assert "episodic" in description.lower()
        assert "factual" in description.lower()

    @pytest.mark.asyncio
    async def test_extract_memories(self):
        """Test discrete memory extraction."""
        strategy = DiscreteMemoryStrategy()

        # Mock the LLM response

        with patch(
            "agent_memory_server.memory_strategies.get_model_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.choices = [AsyncMock()]
            mock_response_obj.choices[
                0
            ].message.content = '{"memories": [{"type": "semantic", "text": "User prefers coffee", "topics": ["preferences"], "entities": ["User", "coffee"]}]}'
            mock_client.create_chat_completion.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await strategy.extract_memories("I love coffee!")

            assert isinstance(result, list)
            mock_client.create_chat_completion.assert_called_once()


class TestSummaryMemoryStrategy:
    """Test summary memory strategy."""

    def test_default_config(self):
        """Test summary strategy default configuration."""
        strategy = SummaryMemoryStrategy()
        assert strategy.max_summary_length == 500

    def test_custom_config(self):
        """Test summary strategy custom configuration."""
        strategy = SummaryMemoryStrategy(max_summary_length=300)
        assert strategy.max_summary_length == 300

    def test_extraction_description(self):
        """Test summary strategy description."""
        strategy = SummaryMemoryStrategy(max_summary_length=400)
        description = strategy.get_extraction_description()
        assert "summaries" in description.lower()
        assert "400 words" in description

    @pytest.mark.asyncio
    async def test_extract_memories(self):
        """Test summary memory extraction."""
        strategy = SummaryMemoryStrategy(max_summary_length=100)

        with patch(
            "agent_memory_server.memory_strategies.get_model_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.choices = [AsyncMock()]
            mock_response_obj.choices[
                0
            ].message.content = '{"memories": [{"type": "semantic", "text": "Discussion about project requirements", "topics": ["project"], "entities": ["requirements"]}]}'
            mock_client.create_chat_completion.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await strategy.extract_memories(
                "Long conversation about project..."
            )

            assert isinstance(result, list)
            # Check that prompt includes the max_summary_length
            call_args = mock_client.create_chat_completion.call_args
            assert "100" in call_args[1]["prompt"]


class TestUserPreferencesMemoryStrategy:
    """Test user preferences memory strategy."""

    def test_extraction_description(self):
        """Test preferences strategy description."""
        strategy = UserPreferencesMemoryStrategy()
        description = strategy.get_extraction_description()
        assert "preferences" in description.lower()
        assert "settings" in description.lower()
        assert "actionable" in description.lower()

    @pytest.mark.asyncio
    async def test_extract_memories(self):
        """Test preferences memory extraction."""
        strategy = UserPreferencesMemoryStrategy()

        with patch(
            "agent_memory_server.memory_strategies.get_model_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.choices = [AsyncMock()]
            mock_response_obj.choices[
                0
            ].message.content = '{"memories": [{"type": "semantic", "text": "User prefers dark mode", "topics": ["preferences"], "entities": ["User"]}]}'
            mock_client.create_chat_completion.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await strategy.extract_memories("I always use dark mode")

            assert isinstance(result, list)
            mock_client.create_chat_completion.assert_called_once()


class TestCustomMemoryStrategy:
    """Test custom memory strategy."""

    def test_custom_prompt_required(self):
        """Test custom strategy requires a prompt."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            CustomMemoryStrategy()

    def test_custom_prompt_initialization(self):
        """Test custom strategy initialization with prompt."""
        prompt = "Extract key points: {message}"
        strategy = CustomMemoryStrategy(custom_prompt=prompt)
        assert strategy.custom_prompt == prompt

    def test_extraction_description(self):
        """Test custom strategy description."""
        prompt = "Custom extraction"
        strategy = CustomMemoryStrategy(custom_prompt=prompt)
        description = strategy.get_extraction_description()
        assert "custom extraction prompt" in description.lower()
        assert "prompt template" in description.lower()

    @pytest.mark.asyncio
    async def test_extract_memories(self):
        """Test custom memory extraction."""
        custom_prompt = "Extract key information: {message} at {current_datetime}"
        strategy = CustomMemoryStrategy(custom_prompt=custom_prompt)

        with patch(
            "agent_memory_server.memory_strategies.get_model_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.choices = [AsyncMock()]
            mock_response_obj.choices[
                0
            ].message.content = '{"memories": [{"type": "semantic", "text": "Custom extracted info", "topics": ["custom"], "entities": ["info"]}]}'
            mock_client.create_chat_completion.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await strategy.extract_memories(
                "Test message", context={"extra": "data"}
            )

            assert isinstance(result, list)
            # Check that the custom prompt was used
            call_args = mock_client.create_chat_completion.call_args
            assert "Extract key information:" in call_args[1]["prompt"]
            assert "Test message" in call_args[1]["prompt"]


class TestMemoryStrategiesRegistry:
    """Test memory strategies registry."""

    def test_registry_completeness(self):
        """Test that all expected strategies are in the registry."""
        expected_strategies = {
            "discrete": DiscreteMemoryStrategy,
            "summary": SummaryMemoryStrategy,
            "preferences": UserPreferencesMemoryStrategy,
            "custom": CustomMemoryStrategy,
        }

        assert expected_strategies == MEMORY_STRATEGIES

    def test_all_strategies_inherit_from_base(self):
        """Test that all registered strategies inherit from base class."""
        for strategy_class in MEMORY_STRATEGIES.values():
            assert issubclass(strategy_class, BaseMemoryStrategy)
