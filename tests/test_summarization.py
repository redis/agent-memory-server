import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.summarization import (
    _incremental_summary,
    summarize_session,
)
from agent_memory_server.utils.keys import Keys


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
        mock_choices = MagicMock()
        mock_choices.message = MagicMock()
        mock_choices.message.content = "This is a summary"
        mock_response.choices = [mock_choices]
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
        mock_choices = MagicMock()
        mock_choices.message = MagicMock()
        mock_choices.message.content = "Updated summary"
        mock_response.choices = [mock_choices]
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


class TestSummarizeSession:
    @pytest.mark.asyncio
    @patch("agent_memory_server.summarization._incremental_summary")
    async def test_summarize_session(
        self, mock_summarization, mock_openai_client, mock_async_redis_client
    ):
        """Test summarize_session with mocked summarization"""
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        max_context_tokens = 1000

        pipeline_mock = MagicMock()  # pipeline is not a coroutine
        pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
        pipeline_mock.watch = AsyncMock()
        mock_async_redis_client.pipeline = MagicMock(return_value=pipeline_mock)

        # Create messages that exceed the token limit
        long_content = (
            "This is a very long message that will exceed our token limit " * 50
        )
        messages_raw = [
            json.dumps({"role": "user", "content": long_content}),
            json.dumps({"role": "assistant", "content": long_content}),
            json.dumps({"role": "user", "content": long_content}),
            json.dumps({"role": "assistant", "content": "Short recent message"}),
        ]

        pipeline_mock.lrange = AsyncMock(return_value=messages_raw)
        pipeline_mock.hgetall = AsyncMock(
            return_value={
                "context": "Previous summary",
                "tokens": "100",
            }
        )
        pipeline_mock.hmset = MagicMock(return_value=True)
        pipeline_mock.ltrim = MagicMock(return_value=True)
        pipeline_mock.execute = AsyncMock(return_value=True)
        pipeline_mock.llen = AsyncMock(return_value=4)

        mock_summarization.return_value = ("New summary", 300)

        with (
            patch(
                "agent_memory_server.summarization.get_model_client"
            ) as mock_get_model_client,
            patch(
                "agent_memory_server.summarization.get_redis_conn",
                return_value=mock_async_redis_client,
            ),
        ):
            mock_get_model_client.return_value = mock_openai_client

            await summarize_session(
                session_id,
                model,
                max_context_tokens,
            )

        assert pipeline_mock.lrange.call_count == 1
        assert pipeline_mock.lrange.call_args[0][0] == Keys.messages_key(session_id)
        assert pipeline_mock.lrange.call_args[0][1] == 0
        assert pipeline_mock.lrange.call_args[0][2] == -1  # Get all messages

        assert pipeline_mock.hgetall.call_count == 1
        assert pipeline_mock.hgetall.call_args[0][0] == Keys.metadata_key(session_id)

        assert pipeline_mock.hmset.call_count == 1
        assert pipeline_mock.hmset.call_args[0][0] == Keys.metadata_key(session_id)
        # Verify that hmset was called with the new summary
        hmset_mapping = pipeline_mock.hmset.call_args.kwargs["mapping"]
        assert hmset_mapping["context"] == "New summary"
        # Token count will vary based on the actual messages passed for summarization
        assert "tokens" in hmset_mapping
        assert (
            int(hmset_mapping["tokens"]) > 300
        )  # Should include summarization tokens plus message tokens

        assert pipeline_mock.ltrim.call_count == 1
        assert pipeline_mock.ltrim.call_args[0][0] == Keys.messages_key(session_id)
        # New token-based approach keeps recent messages
        assert pipeline_mock.ltrim.call_args[0][1] == -1  # Keep last message
        assert pipeline_mock.ltrim.call_args[0][2] == -1

        assert pipeline_mock.execute.call_count == 1

        mock_summarization.assert_called_once()
        assert mock_summarization.call_args[0][0] == model
        assert mock_summarization.call_args[0][1] == mock_openai_client
        assert mock_summarization.call_args[0][2] == "Previous summary"
        # Verify that some messages were passed for summarization
        assert len(mock_summarization.call_args[0][3]) > 0

    @pytest.mark.asyncio
    @patch("agent_memory_server.summarization._incremental_summary")
    async def test_handle_summarization_no_messages(
        self, mock_summarization, mock_openai_client, mock_async_redis_client
    ):
        """Test summarize_session when no messages need summarization"""
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        max_context_tokens = 10000  # High limit so no summarization needed

        pipeline_mock = MagicMock()  # pipeline is not a coroutine
        pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
        pipeline_mock.watch = AsyncMock()
        mock_async_redis_client.pipeline = MagicMock(return_value=pipeline_mock)

        # Set up short messages that won't exceed token limit
        short_messages = [
            json.dumps({"role": "user", "content": "Short message 1"}),
            json.dumps({"role": "assistant", "content": "Short response 1"}),
        ]

        pipeline_mock.llen = AsyncMock(return_value=2)
        pipeline_mock.lrange = AsyncMock(return_value=short_messages)
        pipeline_mock.hgetall = AsyncMock(return_value={})
        pipeline_mock.hmset = AsyncMock(return_value=True)
        pipeline_mock.ltrim = AsyncMock(return_value=True)
        pipeline_mock.execute = AsyncMock(return_value=True)

        with patch(
            "agent_memory_server.summarization.get_redis_conn",
            return_value=mock_async_redis_client,
        ):
            await summarize_session(
                session_id,
                model,
                max_context_tokens,
            )

        # Should not summarize because messages are under token limit
        assert mock_summarization.call_count == 0
        # But should still check messages and metadata
        assert pipeline_mock.lrange.call_count == 1
        assert pipeline_mock.hgetall.call_count == 1
        # Should not update anything since no summarization needed
        assert pipeline_mock.hmset.call_count == 0
        assert pipeline_mock.ltrim.call_count == 0
        assert pipeline_mock.execute.call_count == 0
