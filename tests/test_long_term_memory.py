from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from redis.commands.search.document import Document

from long_term_memory import index_messages, search_messages
from models import OpenAIClientWrapper, RedisearchResult
from utils import REDIS_INDEX_NAME


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_index_messages(
        self, memory_messages, mock_openai_client, mock_async_redis_client
    ):
        """Test indexing messages"""
        # Set up the mock embedding response
        mock_openai_client.create_embedding.return_value = np.array(
            [[0.1, 0.2, 0.3, 0.4] for _ in memory_messages]
        )

        # Call index_messages
        session_id = "test-session"
        mock_async_redis_client.hset = AsyncMock()

        await index_messages(
            memory_messages, session_id, mock_openai_client, mock_async_redis_client
        )

        # Check that create_embedding was called with the right arguments
        contents = [msg.content for msg in memory_messages]
        mock_openai_client.create_embedding.assert_called_with(contents)

        # Verify one of the calls to make sure the data is correct
        for call in mock_async_redis_client.hset.call_args_list:
            args, kwargs = call

            # Check that the key starts with the memory key prefix
            assert args[0].startswith("memory:")

            # Check that the mapping contains the right keys
            mapping = kwargs["mapping"]
            assert "session" in mapping
            assert "vector" in mapping
            assert "content" in mapping
            assert "role" in mapping

            # Check that the session ID is correct
            assert mapping["session"] == session_id

            # Check that the vector is bytes
            assert isinstance(mapping["vector"], bytes)

    @pytest.mark.asyncio
    async def test_search_messages(self, mock_openai_client, mock_async_redis_client):
        """Test searching messages"""
        # Set up the mock embedding response
        mock_openai_client.create_embedding.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        )

        class MockResult:
            def __init__(self, docs):
                self.total = len(docs)
                self.docs = docs

        # Create a proper mock structure for Redis ft().search()
        mock_search = AsyncMock()
        mock_search.return_value = MockResult(
            [
                Document(
                    id=b"doc1",
                    role=b"user",
                    content=b"Hello, world!",
                    dist=0.25,
                ),
                Document(
                    id=b"doc2",
                    role=b"assistant",
                    content=b"Hi there!",
                    dist=0.75,
                ),
            ]
        )

        # Create a mock FT object that has a search method
        mock_ft = MagicMock()
        mock_ft.search = mock_search

        # Setup the ft method to return our mock_ft object
        mock_async_redis_client.ft = MagicMock(return_value=mock_ft)

        # Call search_messages
        query = "What is the meaning of life?"
        session_id = "test-session"
        results = await search_messages(
            query, session_id, mock_openai_client, mock_async_redis_client
        )

        # Check that create_embedding was called with the right arguments
        mock_openai_client.create_embedding.assert_called_with([query])

        # Check that the index name is correct
        assert mock_async_redis_client.ft.call_count == 1
        assert mock_async_redis_client.ft.call_args[0][0] == REDIS_INDEX_NAME

        # Check that search was called with the right arguments
        assert mock_ft.search.call_count == 1
        args = mock_ft.search.call_args[0]
        assert (
            args[0]._query_string
            == "@session:{test-session}=>[KNN 10 @vector $vec AS dist]"
        )

        # Check that the results are parsed correctly
        assert len(results.docs) == 2
        assert isinstance(results.docs[0], RedisearchResult)
        assert results.docs[0].role == "user"
        assert results.docs[0].content == "Hello, world!"
        assert results.docs[0].dist == 0.25
        assert results.docs[1].role == "assistant"
        assert results.docs[1].content == "Hi there!"
        assert results.docs[1].dist == 0.75


@pytest.mark.requires_api_keys
class TestLongTermMemoryIntegration:
    @pytest.mark.asyncio
    async def test_search_messages(self, memory_messages, async_redis_client):
        """Test searching messages"""

        await index_messages(
            memory_messages, "123", OpenAIClientWrapper(), async_redis_client
        )

        results = await search_messages(
            "What is the capital of France?",
            "123",
            OpenAIClientWrapper(),
            async_redis_client,
            limit=1,
        )

        assert results.total == 1
        assert results.docs[0].role == "user"
        assert results.docs[0].content == "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_search_messages_with_distance_threshold(
        self, memory_messages, async_redis_client
    ):
        """Test searching messages with a distance threshold"""

        await index_messages(
            memory_messages, "123", OpenAIClientWrapper(), async_redis_client
        )

        results = await search_messages(
            "What is the capital of France?",
            "123",
            OpenAIClientWrapper(),
            async_redis_client,
            distance_threshold=1.5,
            limit=2,
        )

        assert results.total == 4
        assert len(results.docs) == 2

        assert results.docs[0].role == "user"
        assert results.docs[0].content == "What is the capital of France?"
        assert results.docs[1].role == "assistant"
        assert results.docs[1].content == "The capital of France is Paris."
