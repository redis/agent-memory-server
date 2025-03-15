import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tiktoken

from reducer import incremental_summarization, handle_compaction


@pytest.mark.asyncio
class TestReducer:
    async def test_incremental_summarization_no_context(self, mock_openai_client):
        """Test incremental summarization without previous context"""
        # Setup
        model = "gpt-3.5-turbo"
        context = None
        messages = ["Hello, world!", "How are you?"]

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a summary"
        mock_response.usage.total_tokens = 150

        mock_openai_client.create_chat_completion.return_value = mock_response

        # Call function
        summary, tokens_used = await incremental_summarization(
            model, mock_openai_client, context, messages
        )

        # Verify results
        assert summary == "This is a summary"
        assert tokens_used == 150

        # Verify mock calls
        mock_openai_client.create_chat_completion.assert_called_once()
        args = mock_openai_client.create_chat_completion.call_args[0]

        # Check that the model is correct
        assert args[0] == model

        # Check that the prompt contains the messages
        assert "How are you?" in args[1]
        assert "Hello, world!" in args[1]

    async def test_incremental_summarization_with_context(self, mock_openai_client):
        """Test incremental summarization with previous context"""
        # Setup
        model = "gpt-3.5-turbo"
        context = "Previous summary"
        messages = ["Hello, world!", "How are you?"]

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Updated summary"
        mock_response.usage.total_tokens = 200

        mock_openai_client.create_chat_completion.return_value = mock_response

        # Call function
        summary, tokens_used = await incremental_summarization(
            model, mock_openai_client, context, messages
        )

        # Verify results
        assert summary == "Updated summary"
        assert tokens_used == 200

        # Verify mock calls
        mock_openai_client.create_chat_completion.assert_called_once()
        args = mock_openai_client.create_chat_completion.call_args[0]

        # Check that the model is correct
        assert args[0] == model

        # Check that the prompt contains both the context and messages
        assert "Previous summary" in args[1]
        assert "How are you?" in args[1]
        assert "Hello, world!" in args[1]

    @pytest.mark.asyncio
    @patch("reducer.incremental_summarization")
    async def test_handle_compaction(
        self, mock_summarization, mock_openai_client, mock_redis
    ):
        """Test handle_compaction with mocked summarization"""
        # Setup
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        window_size = 12

        # Configure mock Redis properly - the key is to NOT make pipeline a coroutine
        pipeline_mock = MagicMock()  # Use MagicMock, not AsyncMock for the pipeline
        mock_redis.pipeline = MagicMock(return_value=pipeline_mock)

        # Set up the pipeline mock methods
        pipeline_mock.lrange = MagicMock(return_value=pipeline_mock)
        pipeline_mock.get = MagicMock(return_value=pipeline_mock)

        # Set up the pipeline execute result - this needs to be an awaitable
        pipeline_mock.execute = AsyncMock(
            return_value=[
                # Messages (would be JSON strings in real life)
                [
                    "Message 1",
                    "Message 2",
                    "Message 3",
                    "Message 4",
                    "Message 5",
                    "Message 6",
                ],
                # Context
                "Previous summary",
            ]
        )

        # Mock summarization
        mock_summarization.return_value = ("New summary", 300)

        # Set up other redis methods that might be called
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.lpop = AsyncMock(return_value=True)

        # Mock tiktoken
        with patch("reducer.tiktoken", autospec=True) as mock_tiktoken:
            # Setup mock encoder
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens per message
            mock_tiktoken.get_encoding.return_value = mock_encoding

            # Call function
            await handle_compaction(
                session_id, model, window_size, mock_openai_client, mock_redis
            )

            # Verify summarization was called with correct parameters
            mock_summarization.assert_called_once()
            assert mock_summarization.call_args[0][0] == model
            assert mock_summarization.call_args[0][1] == mock_openai_client
            assert mock_summarization.call_args[0][2] == "Previous summary"

    @pytest.mark.asyncio
    async def test_handle_compaction_no_messages(self, mock_openai_client, mock_redis):
        """Test handle_compaction when no messages need summarization"""
        # Setup
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        window_size = 12

        # Configure mock Redis properly - the key is to NOT make pipeline a coroutine
        pipeline_mock = MagicMock()  # Use MagicMock, not AsyncMock for the pipeline
        mock_redis.pipeline = MagicMock(return_value=pipeline_mock)

        # Set up the pipeline mock methods
        pipeline_mock.lrange = MagicMock(return_value=pipeline_mock)
        pipeline_mock.get = MagicMock(return_value=pipeline_mock)

        # Set up the pipeline execute result - this needs to be an awaitable
        pipeline_mock.execute = AsyncMock(
            return_value=[
                # Empty messages list
                [],
                # Context
                "Previous summary",
            ]
        )

        # Set up other redis methods that might be called
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.lpop = AsyncMock(return_value=True)

        # Call function
        with patch("reducer.tiktoken", autospec=True):
            await handle_compaction(
                session_id, model, window_size, mock_openai_client, mock_redis
            )
