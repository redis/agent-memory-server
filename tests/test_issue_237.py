"""Tests for GitHub issue #237: safe token counting when tiktoken is unavailable."""

from unittest.mock import patch

import pytest

from agent_memory_server.api import _calculate_messages_token_count
from agent_memory_server.models import MemoryMessage


class TestIssue237TiktokenFallback:
    def test_calculate_messages_token_count_falls_back_when_tiktoken_unavailable(
        self,
    ):
        """Token counting should degrade gracefully when the encoding cannot load."""
        messages = [MemoryMessage(role="user", content="Hello world")]

        with (
            patch("agent_memory_server.api._tiktoken_encoding", None),
            patch("agent_memory_server.api._tiktoken_encoding_load_attempted", False),
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
            patch("agent_memory_server.api._tiktoken_encoding", None),
            patch("agent_memory_server.api._tiktoken_encoding_load_attempted", False),
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
