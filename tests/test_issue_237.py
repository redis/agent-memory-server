"""Tests for GitHub issue #237: safe token counting when tiktoken is unavailable."""

from unittest.mock import Mock, patch

import pytest

from agent_memory_server.api import (
    _calculate_messages_token_count,
    _get_tiktoken_encoding,
)
from agent_memory_server.models import MemoryMessage


class TestIssue237TiktokenFallback:
    def test_calculate_messages_token_count_falls_back_when_tiktoken_unavailable(
        self,
    ):
        """Token counting should degrade gracefully when the encoding cannot load."""
        messages = [MemoryMessage(role="user", content="Hello world")]

        with (
            patch("agent_memory_server.api._tiktoken_encoding", None),
            patch("agent_memory_server.api._tiktoken_encoding_last_failed_at", None),
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
            patch("agent_memory_server.api._tiktoken_encoding", None),
            patch("agent_memory_server.api._tiktoken_encoding_last_failed_at", None),
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

    def test_get_tiktoken_encoding_skips_retries_within_backoff_window(self):
        """Repeated calls should not re-attempt loading within the retry interval."""
        mock_get_encoding = Mock(side_effect=Exception("Could not download encoding"))

        with (
            patch("agent_memory_server.api._tiktoken_encoding", None),
            patch("agent_memory_server.api._tiktoken_encoding_last_failed_at", None),
            patch("agent_memory_server.api.time.monotonic", side_effect=[100.0, 101.0]),
            patch("agent_memory_server.api.tiktoken.get_encoding", mock_get_encoding),
        ):
            assert _get_tiktoken_encoding() is None
            assert _get_tiktoken_encoding() is None

        assert mock_get_encoding.call_count == 1

    def test_get_tiktoken_encoding_retries_after_backoff_window(self):
        """A later call should retry loading once the backoff window has passed."""

        class FakeEncoding:
            def encode(self, text: str) -> list[int]:
                return [1] * len(text)

        mock_get_encoding = Mock(
            side_effect=[Exception("temporary failure"), FakeEncoding()]
        )

        with (
            patch("agent_memory_server.api._tiktoken_encoding", None),
            patch("agent_memory_server.api._tiktoken_encoding_last_failed_at", None),
            patch("agent_memory_server.api.time.monotonic", side_effect=[100.0, 401.0]),
            patch("agent_memory_server.api.tiktoken.get_encoding", mock_get_encoding),
        ):
            assert _get_tiktoken_encoding() is None
            assert _get_tiktoken_encoding() is not None

        assert mock_get_encoding.call_count == 2
