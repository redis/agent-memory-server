"""Regression tests for GitHub issue #236."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agent_memory_server.extraction import extract_entities_llm, extract_topics_llm
from agent_memory_server.llm import ChatCompletionResponse
from agent_memory_server.memory_strategies import (
    CustomMemoryStrategy,
    DiscreteMemoryStrategy,
    SummaryMemoryStrategy,
    UserPreferencesMemoryStrategy,
)
from agent_memory_server.utils.llm_json import parse_llm_json


class TestIssue236LlmJsonParsing:
    """Verify JSON parsing tolerates markdown fences and wrapper prose."""

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (
                '{"entities": ["Redis", "Snowflake"]}',
                {"entities": ["Redis", "Snowflake"]},
            ),
            (
                '```json\n{"entities": ["Redis", "Snowflake"]}\n```',
                {"entities": ["Redis", "Snowflake"]},
            ),
            (
                'Here are the extracted topics:\n```json\n{"topics": ["data engineering", "recommendation engines"]}\n```\nI found these topics in the text.',
                {"topics": ["data engineering", "recommendation engines"]},
            ),
        ],
    )
    def test_parse_llm_json_handles_wrapped_content(self, content, expected):
        """The helper should recover valid JSON from common LLM wrappers."""
        assert parse_llm_json(content) == expected

    def test_parse_llm_json_raises_for_invalid_content(self):
        """Invalid non-JSON content should still fail fast."""
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("This response contains no JSON payload at all.")

    @pytest.mark.asyncio
    @patch("agent_memory_server.extraction.LLMClient.create_chat_completion")
    async def test_extract_entities_llm_parses_fenced_json(self, mock_llm):
        """Entity extraction should work when the model wraps JSON in fences."""
        mock_llm.return_value = Mock(
            content='```json\n{"entities": ["Redis", "Snowflake"]}\n```'
        )

        entities = await extract_entities_llm("Redis works with Snowflake.")

        assert set(entities) == {"Redis", "Snowflake"}
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent_memory_server.extraction.LLMClient.create_chat_completion")
    async def test_extract_topics_llm_parses_prose_wrapped_json(self, mock_llm):
        """Topic extraction should work when commentary surrounds the JSON."""
        mock_llm.return_value = Mock(
            content='Here are the extracted topics:\n```json\n{"topics": ["data engineering", "recommendation engines", "streaming"]}\n```\nI found these topics in the text.'
        )

        topics = await extract_topics_llm(
            "Kafka pipelines support recommendations.", num_topics=2
        )

        assert topics == ["data engineering", "recommendation engines"]
        mock_llm.assert_called_once()


@pytest.mark.asyncio
class TestIssue236MemoryStrategies:
    """Verify memory extraction strategies accept wrapped JSON responses."""

    @pytest.mark.parametrize(
        ("strategy_builder", "response_content"),
        [
            (
                lambda: DiscreteMemoryStrategy(),
                '```json\n{"memories": [{"type": "semantic", "text": "User prefers Redis", "topics": ["preferences"], "entities": ["User", "Redis"], "event_date": null}]}\n```',
            ),
            (
                lambda: SummaryMemoryStrategy(max_summary_length=100),
                'Summary generated below.\n```json\n{"memories": [{"type": "semantic", "text": "User discussed Redis adoption", "topics": ["redis"], "entities": ["User", "Redis"]}]}\n```\nDone.',
            ),
            (
                lambda: UserPreferencesMemoryStrategy(),
                '```json\n{"memories": [{"type": "semantic", "text": "User prefers dark mode", "topics": ["preferences"], "entities": ["User"]}]}\n```',
            ),
            (
                lambda: CustomMemoryStrategy(
                    custom_prompt="Extract memories from: {message}"
                ),
                'Custom extraction result:\n```json\n{"memories": [{"type": "semantic", "text": "User prefers async updates", "topics": ["communication"], "entities": ["User"]}]}\n```',
            ),
        ],
    )
    async def test_strategies_parse_wrapped_json(
        self, strategy_builder, response_content
    ):
        """All strategy variants should parse wrapped JSON without retry failure."""
        strategy = strategy_builder()
        response = ChatCompletionResponse(
            content=response_content,
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_create:
            memories = await strategy.extract_memories("Store this memory.")

        assert len(memories) == 1
        assert memories[0]["type"] == "semantic"
        assert memories[0]["text"].startswith("User")
        mock_create.assert_called_once()
