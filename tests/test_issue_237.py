"""Tests for GitHub issue #237: safe token counting when tiktoken is unavailable."""

from unittest.mock import patch

import pytest

from agent_memory_server.api import (
    _calculate_messages_token_count,
    _truncate_text_to_token_budget,
)
from agent_memory_server.models import MemoryMessage


class TestIssue237TiktokenFallback:
    def test_calculate_messages_token_count_falls_back_when_tiktoken_unavailable(
        self,
    ):
        """Token counting should degrade gracefully when the encoding cannot load."""
        messages = [MemoryMessage(role="user", content="Hello world")]

        with (
            patch("agent_memory_server.api._TIKTOKEN_ENCODING_CACHE", None),
            patch("agent_memory_server.api._TIKTOKEN_ENCODING_LOAD_ATTEMPTED", False),
            patch(
                "agent_memory_server.api.tiktoken.get_encoding",
                side_effect=Exception("Could not download encoding data"),
            ),
        ):
            token_count = _calculate_messages_token_count(messages)

        assert token_count > 0

    @pytest.mark.asyncio
    async def test_get_working_memory_uses_fallback_when_tiktoken_unavailable(
        self, client
    ):
        """GET should return session data instead of a 500 when tokenization fails."""
        if client is None:
            pytest.skip("Client not available")

        session_id = "issue-237-api"

        put_response = await client.put(
            f"/v1/working-memory/{session_id}",
            json={
                "messages": [{"role": "user", "content": "Hello from issue 237"}],
                "user_id": "alice",
                "namespace": "demo",
            },
        )
        assert put_response.status_code == 200

        with (
            patch("agent_memory_server.api._TIKTOKEN_ENCODING_CACHE", None),
            patch("agent_memory_server.api._TIKTOKEN_ENCODING_LOAD_ATTEMPTED", False),
            patch(
                "agent_memory_server.api.tiktoken.get_encoding",
                side_effect=Exception("Could not download encoding data"),
            ),
        ):
            get_response = await client.get(
                f"/v1/working-memory/{session_id}?model_name=gpt-4o"
            )

        assert get_response.status_code == 200, get_response.text
        data = get_response.json()
        assert data["session_id"] == session_id
        assert len(data["messages"]) == 1

    def test_truncate_text_to_token_budget_respects_actual_token_limit(self):
        """Oversized messages should be trimmed until they fit the target budget."""

        class FakeEncoding:
            def encode(self, text: str) -> list[int]:
                # Treat every character as a token so naive char*4 truncation would fail.
                return [0] * len(text)

        long_text = "user: " + ("x" * 100)

        with (
            patch("agent_memory_server.api._TIKTOKEN_ENCODING_CACHE", FakeEncoding()),
            patch("agent_memory_server.api._TIKTOKEN_ENCODING_LOAD_ATTEMPTED", True),
        ):
            truncated = _truncate_text_to_token_budget(long_text, 10)

        assert len(truncated) == 10
