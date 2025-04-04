import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redis_memory_server.summarization import (
    _incremental_summary,
    summarize_session,
)
from redis_memory_server.utils import Keys


@pytest.mark.asyncio
class TestIncrementalSummarization:
    async def test_incremental_summarization_no_context(self, mock_openai_client):
        """Test incremental summarization without previous context"""
        model = "gpt-3.5-turbo"
        context = None
        messages = [
            json.dumps({"role": "user", "content": "Hello, world!"}),
            json.dumps({"role": "assistant", "content": "How are you?"}),
        ]

        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "This is a summary"}}]
        mock_response.total_tokens = 150

        mock_openai_client.create_chat_completion.return_value = mock_response

        summary, tokens_used = await _incremental_summary(
            model, mock_openai_client, context, messages
        )

        assert summary == "This is a summary"
        assert tokens_used == 150

        mock_openai_client.create_chat_completion.assert_called_once()
        args = mock_openai_client.create_chat_completion.call_args[0]

        assert args[0] == model
        assert "How are you?" in args[1]
        assert "Hello, world!" in args[1]

    async def test_incremental_summarization_with_context(self, mock_openai_client):
        """Test incremental summarization with previous context"""
        model = "gpt-3.5-turbo"
        context = "Previous summary"
        messages = [
            json.dumps({"role": "user", "content": "Hello, world!"}),
            json.dumps({"role": "assistant", "content": "How are you?"}),
        ]

        # Create a response that matches our new ChatResponse format
        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "Updated summary"}}]
        mock_response.total_tokens = 200

        mock_openai_client.create_chat_completion.return_value = mock_response

        summary, tokens_used = await _incremental_summary(
            model, mock_openai_client, context, messages
        )

        assert summary == "Updated summary"
        assert tokens_used == 200

        mock_openai_client.create_chat_completion.assert_called_once()
        args = mock_openai_client.create_chat_completion.call_args[0]

        assert args[0] == model
        assert "Previous summary" in args[1]
        assert "How are you?" in args[1]
        assert "Hello, world!" in args[1]


class TestHandleCompaction:
    @pytest.mark.asyncio
    @patch("redis_memory_server.models.summarization._incremental_summary")
    async def test_handle_compaction(
        self, mock_summarization, mock_openai_client, mock_async_redis_client
    ):
        """Test handle_compaction with mocked summarization"""
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        window_size = 12

        pipeline_mock = MagicMock()  # pipeline is not a coroutine
        mock_async_redis_client.pipeline = MagicMock(return_value=pipeline_mock)

        pipeline_mock.lrange = MagicMock(return_value=pipeline_mock)
        pipeline_mock.get = MagicMock(return_value=pipeline_mock)
        pipeline_mock.execute = AsyncMock(
            return_value=[
                [
                    json.dumps({"role": "user", "content": "Message 1"}),
                    json.dumps({"role": "assistant", "content": "Message 2"}),
                    json.dumps({"role": "user", "content": "Message 3"}),
                    json.dumps({"role": "assistant", "content": "Message 4"}),
                    json.dumps({"role": "user", "content": "Message 5"}),
                    json.dumps({"role": "assistant", "content": "Message 6"}),
                ],
                "Previous summary",
            ]
        )

        mock_summarization.return_value = ("New summary", 300)

        mock_async_redis_client.set = AsyncMock(return_value=True)
        mock_async_redis_client.lpop = AsyncMock(return_value=True)

        await summarize_session(
            session_id,
            model,
            window_size,
            mock_openai_client,
            mock_async_redis_client,
        )

        assert pipeline_mock.lrange.call_count == 1
        assert pipeline_mock.lrange.call_args[0][0] == Keys.messages_key(session_id)
        assert pipeline_mock.lrange.call_args[0][1] == 6
        assert pipeline_mock.lrange.call_args[0][2] == window_size

        mock_summarization.assert_called_once()
        assert mock_summarization.call_args[0][0] == model
        assert mock_summarization.call_args[0][1] == mock_openai_client
        assert mock_summarization.call_args[0][2] == "Previous summary"
        assert len(mock_summarization.call_args[0][3]) == 6

    @pytest.mark.asyncio
    @patch("redis_memory_server.models.summarization._incremental_summary")
    async def test_handle_compaction_no_messages(
        self, mock_summarization, mock_openai_client, mock_async_redis_client
    ):
        """Test handle_compaction when no messages need summarization"""
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        window_size = 12

        pipeline_mock = MagicMock()  # pipeline is not a coroutine
        mock_async_redis_client.pipeline = MagicMock(return_value=pipeline_mock)

        pipeline_mock.lrange = MagicMock(return_value=pipeline_mock)
        pipeline_mock.get = MagicMock(return_value=pipeline_mock)

        pipeline_mock.execute = AsyncMock(
            return_value=[
                [],
                "Previous summary",
            ]
        )

        mock_async_redis_client.set = AsyncMock(return_value=True)
        mock_async_redis_client.lpop = AsyncMock(return_value=True)

        await summarize_session(
            session_id,
            model,
            window_size,
            mock_openai_client,
            mock_async_redis_client,
        )

        assert mock_summarization.call_count == 0
