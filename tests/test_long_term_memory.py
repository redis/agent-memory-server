import time
from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from redis.commands.search.document import Document
from ulid import ULID

from agent_memory_server.filters import SessionId
from agent_memory_server.long_term_memory import (
    deduplicate_by_id,
    index_long_term_memories,
    promote_working_memory_to_long_term,
    search_long_term_memories,
    search_memories,
)
from agent_memory_server.models import MemoryRecord, MemoryRecordResult
from agent_memory_server.utils.redis import ensure_search_index_exists


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_index_memories(
        self, mock_openai_client, mock_async_redis_client, session
    ):
        """Test indexing messages"""
        long_term_memories = [
            MemoryRecord(text="Paris is the capital of France", session_id=session),
            MemoryRecord(text="France is a country in Europe", session_id=session),
        ]

        # Create two separate embedding vectors
        mock_vectors = [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
            np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32).tobytes(),
        ]

        mock_vectorizer = MagicMock()
        mock_vectorizer.aembed_many = AsyncMock(return_value=mock_vectors)

        mock_async_redis_client.hset = AsyncMock()

        with mock.patch(
            "agent_memory_server.long_term_memory.OpenAITextVectorizer",
            return_value=mock_vectorizer,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=mock_async_redis_client,
            )

        # Check that create_embedding was called with the right arguments
        contents = [memory.text for memory in long_term_memories]
        mock_vectorizer.aembed_many.assert_called_with(
            contents,
            batch_size=20,
            as_buffer=True,
        )

        # Verify one of the calls to make sure the data is correct
        for i, call in enumerate(mock_async_redis_client.hset.call_args_list):
            args, kwargs = call

            # Check that the key starts with the memory key prefix
            assert args[0].startswith("memory:")

            # Check that the mapping contains the right keys
            mapping = kwargs["mapping"]
            assert mapping == {
                "text": long_term_memories[i].text,
                "id_": long_term_memories[i].id_,
                "session_id": long_term_memories[i].session_id,
                "user_id": long_term_memories[i].user_id,
                "last_accessed": long_term_memories[i].last_accessed,
                "created_at": long_term_memories[i].created_at,
                "vector": mock_vectors[i],
            }

    @pytest.mark.asyncio
    async def test_search_memories(self, mock_openai_client, mock_async_redis_client):
        """Test searching memories"""
        # Set up the mock embedding response
        mock_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mock_vectorizer = MagicMock()
        mock_vectorizer.aembed = AsyncMock(return_value=mock_vector)

        class MockResult:
            def __init__(self, docs):
                self.total = len(docs)
                self.docs = docs

        mock_now = time.time()

        mock_query = AsyncMock()
        # Return a list of documents directly instead of a MockResult object
        mock_query.return_value = [
            Document(
                id=b"doc1",
                id_=str(ULID()),
                text=b"Hello, world!",
                vector_distance=0.25,
                created_at=mock_now,
                last_accessed=mock_now,
                user_id=None,
                session_id=None,
                namespace=None,
                topics=None,
                entities=None,
            ),
            Document(
                id=b"doc2",
                id_=str(ULID()),
                text=b"Hi there!",
                vector_distance=0.75,
                created_at=mock_now,
                last_accessed=mock_now,
                user_id=None,
                session_id=None,
                namespace=None,
                topics=None,
                entities=None,
            ),
        ]

        mock_index = MagicMock()
        mock_index.query = mock_query

        query = "What is the meaning of life?"
        session_id = SessionId(eq="test-session")

        with (
            mock.patch(
                "agent_memory_server.long_term_memory.OpenAITextVectorizer",
                return_value=mock_vectorizer,
            ),
            mock.patch(
                "agent_memory_server.long_term_memory.get_search_index",
                return_value=mock_index,
            ),
        ):
            results = await search_long_term_memories(
                query,
                mock_async_redis_client,
                session_id=session_id,
            )

        # Check that create_embedding was called with the right arguments
        mock_vectorizer.aembed.assert_called_with(query)

        assert mock_index.query.call_count == 1

        assert len(results.memories) == 1
        assert isinstance(results.memories[0], MemoryRecordResult)
        assert results.memories[0].text == "Hello, world!"
        assert results.memories[0].dist == 0.25
        assert results.memories[0].memory_type == "message"

    @pytest.mark.asyncio
    async def test_search_memories_unified_search(self, mock_async_redis_client):
        """Test unified search across working memory and long-term memory"""

        from agent_memory_server.models import (
            MemoryRecordResults,
            WorkingMemory,
        )

        # Mock search_long_term_memories to return some long-term results
        mock_long_term_results = MemoryRecordResults(
            total=1,
            memories=[
                MemoryRecordResult(
                    id_="long-term-1",
                    text="Long-term: User likes coffee",
                    dist=0.3,
                    memory_type="semantic",
                    created_at=datetime.fromtimestamp(1000),
                    updated_at=datetime.fromtimestamp(1000),
                    last_accessed=datetime.fromtimestamp(1000),
                )
            ],
        )

        # Mock working memory with matching content
        test_working_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[
                MemoryRecord(
                    text="Working memory: coffee preferences",
                    id="working-1",
                    id_="working-1",  # Set both id and id_ for consistency
                    memory_type="semantic",
                    persisted_at=None,  # Not persisted yet
                )
            ],
        )

        with (
            patch(
                "agent_memory_server.long_term_memory.search_long_term_memories"
            ) as mock_search_lt,
            patch("agent_memory_server.messages.list_sessions") as mock_list_sessions,
            patch(
                "agent_memory_server.working_memory.get_working_memory"
            ) as mock_get_wm,
            patch("agent_memory_server.long_term_memory.settings") as mock_settings,
        ):
            # Setup mocks
            mock_settings.long_term_memory = True
            mock_search_lt.return_value = mock_long_term_results
            mock_list_sessions.return_value = (1, ["test-session"])
            mock_get_wm.return_value = test_working_memory

            # Call search_memories
            results = await search_memories(
                text="coffee",
                redis=mock_async_redis_client,
                include_working_memory=True,
                include_long_term_memory=True,
                limit=10,
                offset=0,
            )

            # Verify both search functions were called
            mock_search_lt.assert_called_once()
            mock_list_sessions.assert_called_once()
            mock_get_wm.assert_called_once()

            # Verify results contain both working and long-term memory
            assert results.total == 2  # 1 from long-term + 1 from working
            assert len(results.memories) == 2

            # Working memory should come first (dist=0.0)
            working_result = results.memories[0]
            assert working_result.id_ == "working-1"
            assert working_result.text == "Working memory: coffee preferences"
            assert working_result.dist == 0.0

            # Long-term memory should come second
            long_term_result = results.memories[1]
            assert long_term_result.id_ == "long-term-1"
            assert long_term_result.text == "Long-term: User likes coffee"
            assert long_term_result.dist == 0.3

    @pytest.mark.asyncio
    async def test_deduplicate_by_id(self, mock_async_redis_client):
        """Test id-based deduplication"""
        # Create a memory with an id
        memory = MemoryRecord(
            text="Test memory",
            id="test-client-123",
            namespace="test",
        )

        # Mock Redis search to return no existing memory with this id
        mock_async_redis_client.execute_command.return_value = [0]

        result_memory, was_overwrite = await deduplicate_by_id(
            memory=memory,
            redis_client=mock_async_redis_client,
        )

        assert was_overwrite is False

        # Mock Redis search to return an existing memory with the same id
        mock_async_redis_client.execute_command.return_value = [
            1,
            "memory:existing-key",
            "1234567890",
        ]
        # Mock the delete method as an AsyncMock
        mock_async_redis_client.delete = AsyncMock()

        result_memory, was_overwrite = await deduplicate_by_id(
            memory=memory,
            redis_client=mock_async_redis_client,
        )
        assert was_overwrite is True
        assert result_memory is not None
        assert result_memory.id == memory.id

        # Verify delete was called
        mock_async_redis_client.delete.assert_called_once_with("memory:existing-key")

    @pytest.mark.asyncio
    async def test_promote_working_memory_to_long_term(self, mock_async_redis_client):
        """Test promotion of working memory to long-term storage"""
        from datetime import datetime
        from unittest.mock import patch

        from agent_memory_server.models import MemoryRecord, WorkingMemory

        # Create test memories - some persisted, some not
        persisted_memory = MemoryRecord(
            text="Already persisted memory",
            id="persisted-id",
            namespace="test",
            memory_type="semantic",
            persisted_at=datetime.now(UTC),
        )

        unpersisted_memory1 = MemoryRecord(
            text="Unpersisted memory 1",
            id="unpersisted-1",
            namespace="test",
            memory_type="semantic",
            persisted_at=None,
        )

        unpersisted_memory2 = MemoryRecord(
            text="Unpersisted memory 2",
            id="unpersisted-2",
            namespace="test",
            memory_type="episodic",
            persisted_at=None,
        )

        test_working_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[persisted_memory, unpersisted_memory1, unpersisted_memory2],
        )

        # Mock working_memory functions
        with (
            patch("agent_memory_server.working_memory.get_working_memory") as mock_get,
            patch("agent_memory_server.working_memory.set_working_memory") as mock_set,
            patch(
                "agent_memory_server.long_term_memory.deduplicate_by_id"
            ) as mock_dedup,
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index,
        ):
            # Setup mocks
            mock_get.return_value = test_working_memory
            mock_set.return_value = None
            mock_dedup.side_effect = [
                (unpersisted_memory1, False),  # First call - no overwrite
                (unpersisted_memory2, False),  # Second call - no overwrite
            ]
            mock_index.return_value = None

            # Call the promotion function
            promoted_count = await promote_working_memory_to_long_term(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Verify results
            assert promoted_count == 2

            # Verify working memory was retrieved
            mock_get.assert_called_once_with(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Verify deduplication was called for unpersisted memories
            assert mock_dedup.call_count == 2

            # Verify indexing was called for unpersisted memories
            assert mock_index.call_count == 2

            # Verify working memory was updated with new timestamps
            mock_set.assert_called_once()
            updated_memory = mock_set.call_args[1]["working_memory"]

            # Check that the unpersisted memories now have persisted_at set
            unpersisted_memories_updated = [
                mem
                for mem in updated_memory.memories
                if mem.id in ["unpersisted-1", "unpersisted-2"]
            ]
            assert len(unpersisted_memories_updated) == 2
            for mem in unpersisted_memories_updated:
                assert mem.persisted_at is not None
                assert isinstance(mem.persisted_at, datetime)

            # Check that already persisted memory was unchanged
            persisted_memories = [
                mem for mem in updated_memory.memories if mem.id == "persisted-id"
            ]
            assert len(persisted_memories) == 1
            assert persisted_memories[0].persisted_at == persisted_memory.persisted_at

    @pytest.mark.asyncio
    async def test_sync_and_conflict_safety(self, mock_async_redis_client):
        """Test that client state resubmission is safe and converges properly."""
        from datetime import datetime
        from unittest.mock import patch

        from agent_memory_server.models import MemoryRecord, WorkingMemory

        # Create test memories - some persisted, some not
        persisted_memory = MemoryRecord(
            text="Already persisted memory",
            id="persisted-id",
            namespace="test",
            memory_type="semantic",
            persisted_at=datetime.now(UTC),
        )

        unpersisted_memory1 = MemoryRecord(
            text="Unpersisted memory 1",
            id="unpersisted-1",
            namespace="test",
            memory_type="semantic",
            persisted_at=None,
        )

        unpersisted_memory2 = MemoryRecord(
            text="Unpersisted memory 2",
            id="unpersisted-2",
            namespace="test",
            memory_type="episodic",
            persisted_at=None,
        )

        test_working_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[persisted_memory, unpersisted_memory1, unpersisted_memory2],
        )

        # Mock working_memory functions
        with (
            patch("agent_memory_server.working_memory.get_working_memory") as mock_get,
            patch("agent_memory_server.working_memory.set_working_memory") as mock_set,
            patch(
                "agent_memory_server.long_term_memory.deduplicate_by_id"
            ) as mock_dedup,
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index,
        ):
            # Setup mocks
            mock_get.return_value = test_working_memory
            mock_set.return_value = None
            mock_dedup.side_effect = [
                (unpersisted_memory1, False),  # First call - no overwrite
                (unpersisted_memory2, False),  # Second call - no overwrite
            ]
            mock_index.return_value = None

            # Call the promotion function
            promoted_count = await promote_working_memory_to_long_term(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Verify results
            assert promoted_count == 2

            # Verify working memory was retrieved
            mock_get.assert_called_once_with(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Verify deduplication was called for unpersisted memories
            assert mock_dedup.call_count == 2

            # Verify indexing was called for unpersisted memories
            assert mock_index.call_count == 2

            # Verify working memory was updated with new timestamps
            mock_set.assert_called_once()
            updated_memory = mock_set.call_args[1]["working_memory"]

            # Check that the unpersisted memories now have persisted_at set
            unpersisted_memories_updated = [
                mem
                for mem in updated_memory.memories
                if mem.id in ["unpersisted-1", "unpersisted-2"]
            ]
            assert len(unpersisted_memories_updated) == 2
            for mem in unpersisted_memories_updated:
                assert mem.persisted_at is not None
                assert isinstance(mem.persisted_at, datetime)

            # Check that already persisted memory was unchanged
            persisted_memories = [
                mem for mem in updated_memory.memories if mem.id == "persisted-id"
            ]
            assert len(persisted_memories) == 1
            assert persisted_memories[0].persisted_at == persisted_memory.persisted_at

        # Now test client resubmission scenario
        # Simulate client resubmitting stale state with new memory
        resubmitted_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[
                # Existing memory resubmitted without persisted_at (client doesn't track this)
                MemoryRecord(
                    text="Unpersisted memory 1",
                    id="unpersisted-1",  # Same id as before
                    namespace="test",
                    memory_type="semantic",
                    persisted_at=None,  # Client doesn't know about server timestamps
                ),
                # New memory from client
                MemoryRecord(
                    text="New memory from client",
                    id="new-memory-3",
                    namespace="test",
                    memory_type="semantic",
                    persisted_at=None,
                ),
            ],
        )

        with (
            patch("agent_memory_server.working_memory.get_working_memory") as mock_get2,
            patch("agent_memory_server.working_memory.set_working_memory") as mock_set2,
            patch(
                "agent_memory_server.long_term_memory.deduplicate_by_id"
            ) as mock_dedup2,
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index2,
        ):
            # Setup mocks for resubmission scenario
            mock_get2.return_value = resubmitted_memory
            mock_set2.return_value = None
            # First call: existing memory found (overwrite)
            # Second call: new memory, no existing (no overwrite)
            mock_dedup2.side_effect = [
                (resubmitted_memory.memories[0], True),  # Overwrite existing
                (resubmitted_memory.memories[1], False),  # New memory
            ]
            mock_index2.return_value = None

            # Call promotion again
            promoted_count_2 = await promote_working_memory_to_long_term(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Both memories should be promoted (one overwrite, one new)
            assert promoted_count_2 == 2

            # Verify final working memory state
            mock_set2.assert_called_once()
            final_memory = mock_set2.call_args[1]["working_memory"]

            # Both memories should have persisted_at set
            for mem in final_memory.memories:
                assert mem.persisted_at is not None

            # This demonstrates that:
            # 1. Client can safely resubmit stale state
            # 2. Server handles id-based overwrites correctly
            # 3. Working memory converges to consistent state with proper timestamps


@pytest.mark.requires_api_keys
class TestLongTermMemoryIntegration:
    """Integration tests for long-term memory"""

    @pytest.mark.asyncio
    async def test_search_messages(self, async_redis_client):
        """Test searching messages"""
        await ensure_search_index_exists(async_redis_client)

        long_term_memories = [
            MemoryRecord(text="Paris is the capital of France", session_id="123"),
            MemoryRecord(text="France is a country in Europe", session_id="123"),
        ]

        with mock.patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            return_value=async_redis_client,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=async_redis_client,
            )

        results = await search_long_term_memories(
            "What is the capital of France?",
            async_redis_client,
            session_id=SessionId(eq="123"),
            limit=1,
        )

        assert results.total == 1
        assert len(results.memories) == 1
        assert results.memories[0].text == "Paris is the capital of France"
        assert results.memories[0].session_id == "123"
        assert results.memories[0].memory_type == "message"

    @pytest.mark.asyncio
    async def test_search_messages_with_distance_threshold(self, async_redis_client):
        """Test searching messages with a distance threshold"""
        await ensure_search_index_exists(async_redis_client)

        long_term_memories = [
            MemoryRecord(text="Paris is the capital of France", session_id="123"),
            MemoryRecord(text="France is a country in Europe", session_id="123"),
        ]

        with mock.patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            return_value=async_redis_client,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=async_redis_client,
            )

        results = await search_long_term_memories(
            "What is the capital of France?",
            async_redis_client,
            session_id=SessionId(eq="123"),
            distance_threshold=0.1,
            limit=2,
        )

        assert results.total == 1
        assert len(results.memories) == 1
        assert results.memories[0].text == "Paris is the capital of France"
        assert results.memories[0].session_id == "123"
        assert results.memories[0].memory_type == "message"
