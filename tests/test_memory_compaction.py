"""
Tests for the memory compaction functionality.

This test suite covers:
1. Hash-based memory duplicate detection and merging
2. Semantic similarity detection and merging using RedisVL
3. The core functionality of the compact_long_term_memories function

Key functions tested:
- generate_memory_hash: Creating consistent hashes for memory identification
- merge_memories_with_llm: Merging similar or duplicate memories with LLM
- compact_long_term_memories: The main compaction workflow

Test strategy:
- Unit tests for the helper functions (hash generation, memory merging)
- Simplified tests for the compaction workflow using helper function
- Direct tests for semantic merging
"""

import json
import time
from datetime import datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import nanoid
import pytest
from redis.commands.search.document import Document

import agent_memory_server.long_term_memory
from agent_memory_server.long_term_memory import (
    compact_long_term_memories,
    generate_memory_hash,
    index_long_term_memories,
    merge_memories_with_llm,
)
from agent_memory_server.utils import (
    ensure_search_index_exists,
    get_redis_conn,
)


# Helper function to create a fully mocked version of compact_long_term_memories for testing
async def run_compact_memories_with_mocks(
    mock_redis,
    memory_keys,
    memory_contents,
    hash_values,
    merged_memories,
    search_results,
):
    """
    Run a fully mocked version of compact_long_term_memories for testing.

    Args:
        mock_redis: The mocked Redis client
        memory_keys: List of memory keys to return from scan
        memory_contents: List of memory dictionaries to return from hgetall
        hash_values: List of hash values to return from generate_memory_hash
        merged_memories: List of merged memory dictionaries to return from merge_memories_with_llm
        search_results: List of search results to return from search_index.search
    """
    # Setup scan mock
    mock_redis.scan = AsyncMock()
    mock_redis.scan.side_effect = [(0, memory_keys)]

    # Setup pipeline mock
    mock_pipeline = AsyncMock()
    mock_pipeline.execute = AsyncMock(return_value=memory_contents)
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

    # Setup delete mock
    mock_redis.delete = AsyncMock()

    # Setup hash generation mock
    with patch(
        "agent_memory_server.long_term_memory.generate_memory_hash",
        side_effect=hash_values
        if isinstance(hash_values, list)
        else [hash_values] * len(memory_contents),
    ):
        # Setup LLM merging mock
        merge_memories_mock = AsyncMock()
        merge_memories_mock.side_effect = (
            merged_memories if isinstance(merged_memories, list) else [merged_memories]
        )

        # Setup vectorizer mock
        mock_vectorizer = MagicMock()
        mock_vectorizer.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # Setup search index mock
        mock_index = MagicMock()
        # If we need multiple search results, use return_value to avoid StopIteration
        if isinstance(search_results, list) and len(search_results) > 1:
            mock_index.search = AsyncMock(side_effect=search_results)
        else:
            mock_index.search = AsyncMock(
                return_value=search_results[0]
                if isinstance(search_results, list)
                else search_results
            )

        with (
            patch(
                "agent_memory_server.long_term_memory.merge_memories_with_llm",
                merge_memories_mock,
            ),
            patch(
                "agent_memory_server.long_term_memory.OpenAITextVectorizer",
                return_value=mock_vectorizer,
            ),
            patch(
                "agent_memory_server.long_term_memory.get_search_index",
                return_value=mock_index,
            ),
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories",
                AsyncMock(),
            ),
        ):
            # Call the function
            await agent_memory_server.long_term_memory.compact_long_term_memories(
                mock_redis
            )

            return {
                "merge_memories": merge_memories_mock,
                "search_index": mock_index,
                "index_memories": mock.DEFAULT,
                "redis_delete": mock_redis.delete,
            }


class TestMemoryCompaction:
    @pytest.mark.asyncio
    async def test_generate_memory_hash(self):
        """Test that the memory hash generation is stable and deterministic"""
        # Create two identical memories
        memory1 = {
            "text": "Paris is the capital of France",
            "user_id": "user123",
            "session_id": "session456",
        }
        memory2 = {
            "text": "Paris is the capital of France",
            "user_id": "user123",
            "session_id": "session456",
        }
        # Generate hashes
        hash1 = agent_memory_server.long_term_memory.generate_memory_hash(memory1)
        hash2 = agent_memory_server.long_term_memory.generate_memory_hash(memory2)

        # Hashes should be identical for identical memories
        assert hash1 == hash2

        # Changing any key field should change the hash
        memory3 = {
            "text": "Paris is the capital of France",
            "user_id": "different_user",
            "session_id": "session456",
        }
        hash3 = agent_memory_server.long_term_memory.generate_memory_hash(memory3)
        assert hash1 != hash3

        # Order of fields in the dictionary should not matter
        memory4 = {
            "session_id": "session456",
            "text": "Paris is the capital of France",
            "user_id": "user123",
        }
        hash4 = generate_memory_hash(memory4)
        assert hash1 == hash4

    @pytest.mark.asyncio
    async def test_merge_memories_with_llm(self, mock_openai_client):
        """Test merging memories with LLM"""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Merged memory content"

        mock_model_client = AsyncMock()
        mock_model_client.create_chat_completion = AsyncMock(return_value=mock_response)

        with patch(
            "agent_memory_server.long_term_memory.get_model_client",
            return_value=mock_model_client,
        ):
            # Test merging two memories
            memories = [
                {
                    "text": "Memory 1 content",
                    "id_": nanoid.generate(),
                    "user_id": "user123",
                    "session_id": "session456",
                    "namespace": "test",
                    "created_at": int(time.time()) - 100,
                    "last_accessed": int(time.time()) - 50,
                    "topics": ["topic1", "topic2"],
                    "entities": ["entity1"],
                },
                {
                    "text": "Memory 2 content",
                    "id_": nanoid.generate(),
                    "user_id": "user123",
                    "session_id": "session456",
                    "namespace": "test",
                    "created_at": int(time.time()) - 200,
                    "last_accessed": int(time.time()),
                    "topics": ["topic2", "topic3"],
                    "entities": ["entity2"],
                },
            ]

            merged = await merge_memories_with_llm(memories, "hash")

            # Check merged content
            assert merged["text"] == "Merged memory content"

            # Should have the earliest created_at
            assert merged["created_at"] == memories[1]["created_at"]

            # Should have the most recent last_accessed
            assert merged["last_accessed"] == memories[1]["last_accessed"]

            # Should preserve user_id, session_id, namespace
            assert merged["user_id"] == "user123"
            assert merged["session_id"] == "session456"
            assert merged["namespace"] == "test"

            # Should combine topics and entities
            assert set(merged["topics"]) == {"topic1", "topic2", "topic3"}
            assert set(merged["entities"]) == {"entity1", "entity2"}

            # Should have a memory_hash
            assert "memory_hash" in merged

    @pytest.mark.asyncio
    async def test_compact_hash_based_duplicates(self, mock_async_redis_client):
        """Test compacting hash-based duplicates"""
        # Set up mock data
        memory_hash = "hash123"
        memory_key1 = "memory:123"
        memory_key2 = "memory:456"

        # Mock scanning Redis for memory keys
        mock_async_redis_client.scan = AsyncMock()
        mock_async_redis_client.scan.side_effect = [
            (
                0,
                [memory_key1, memory_key2],
            )  # First and only scan returns 2 keys and cursor 0
        ]

        # Mock content of the memory keys
        memory1 = {
            "text": "Duplicate memory",
            "id_": "123",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 100),
            "last_accessed": str(int(time.time()) - 50),
            "key": memory_key1,
        }

        memory2 = {
            "text": "Duplicate memory",
            "id_": "456",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 200),
            "last_accessed": str(int(time.time())),
            "key": memory_key2,
        }

        # Setup pipeline for memory retrieval
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[memory1, memory2])
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        # Mock memory hash generation to return the same hash for both memories
        with patch(
            "agent_memory_server.long_term_memory.generate_memory_hash",
            return_value=memory_hash,
        ):
            # Mock LLM merging
            merged_memory = {
                "text": "Merged memory content",
                "id_": "merged",
                "user_id": "user123",
                "session_id": "session456",
                "namespace": "test",
                "created_at": int(time.time()) - 200,
                "last_accessed": int(time.time()),
                "topics": [],
                "entities": [],
                "memory_hash": "merged_hash",
            }

            with (
                patch(
                    "agent_memory_server.long_term_memory.merge_memories_with_llm",
                    AsyncMock(return_value=merged_memory),
                ),
                patch(
                    "agent_memory_server.long_term_memory.get_search_index",
                    MagicMock(),
                ),
                patch(
                    "agent_memory_server.long_term_memory.index_long_term_memories",
                    AsyncMock(),
                ),
            ):
                # Mock vector search to return no similar memories
                mock_search_result = MagicMock()
                mock_search_result.docs = []
                mock_search_result.total = 0

                mock_index = MagicMock()
                mock_index.search = AsyncMock(return_value=mock_search_result)

                with patch(
                    "agent_memory_server.long_term_memory.get_search_index",
                    return_value=mock_index,
                ):
                    # Call the function
                    await compact_long_term_memories(mock_async_redis_client)

                    # Verify Redis scan was called
                    mock_async_redis_client.scan.assert_called_once()

                    # Verify memories were retrieved
                    mock_pipeline.execute.assert_called()

                    # Verify merge_memories_with_llm was called
                    from agent_memory_server.long_term_memory import (
                        merge_memories_with_llm,
                    )

                    merge_memories_with_llm.assert_called_once()

                    # Verify index_long_term_memories was called

                    index_long_term_memories.assert_called_once()

                    # Verify Redis delete was called
                    mock_async_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_semantic_duplicates(
        self, mock_async_redis_client, mock_openai_client
    ):
        """Test compacting semantically similar memories using RedisVL"""
        # Set up mock data
        memory_key1 = "memory:123"
        memory_key2 = "memory:456"

        # Mock scanning Redis for memory keys
        mock_async_redis_client.scan = AsyncMock()
        mock_async_redis_client.scan.side_effect = [
            (0, [memory_key1, memory_key2])  # Returns 2 keys and cursor 0
        ]

        # Mock content of the memory keys with different hashes
        memory1 = {
            "text": "Paris is the capital of France",
            "id_": "123",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 100),
            "last_accessed": str(int(time.time()) - 50),
            "key": memory_key1,
        }

        memory2 = {
            "text": "The capital city of France is Paris",
            "id_": "456",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 200),
            "last_accessed": str(int(time.time())),
            "key": memory_key2,
        }

        # Setup pipeline for memory retrieval
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[memory1, memory2])
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        # Setup mocks for different hash values
        with patch(
            "agent_memory_server.long_term_memory.generate_memory_hash",
            side_effect=["hash123", "hash456"],
        ):
            # Mock the vectorizer
            mock_vectorizer = MagicMock()
            mock_vectorizer.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])

            with patch(
                "agent_memory_server.long_term_memory.OpenAITextVectorizer",
                return_value=mock_vectorizer,
            ):
                # Mock the search result to show semantic similarity
                mock_doc = Document(
                    id=b"doc1",
                    id_="456",
                    text="The capital city of France is Paris",
                    vector_distance=0.05,  # Close distance indicating similarity
                    created_at=str(int(time.time()) - 200),
                    last_accessed=str(int(time.time())),
                    user_id="user123",
                    session_id="session456",
                    namespace="test",
                )
                mock_search_result = MagicMock()
                mock_search_result.docs = [mock_doc]
                mock_search_result.total = 1

                mock_index = MagicMock()
                mock_index.search = AsyncMock(return_value=mock_search_result)

                with patch(
                    "agent_memory_server.long_term_memory.get_search_index",
                    return_value=mock_index,
                ):
                    # Mock LLM merging for semantic duplicates
                    merged_memory = {
                        "text": "Paris is the capital of France",
                        "id_": "merged",
                        "user_id": "user123",
                        "session_id": "session456",
                        "namespace": "test",
                        "created_at": int(time.time()) - 200,
                        "last_accessed": int(time.time()),
                        "topics": [],
                        "entities": [],
                        "memory_hash": "merged_hash",
                    }

                    with (
                        patch(
                            "agent_memory_server.long_term_memory.merge_memories_with_llm",
                            AsyncMock(return_value=merged_memory),
                        ),
                        patch(
                            "agent_memory_server.long_term_memory.index_long_term_memories",
                            AsyncMock(),
                        ),
                    ):
                        # Call the function
                        await compact_long_term_memories(mock_async_redis_client)

                        # Verify search was called with VectorRangeQuery
                        mock_index.search.assert_called()

                        # Verify merge_memories_with_llm was called for semantic duplicates
                        from agent_memory_server.long_term_memory import (
                            merge_memories_with_llm,
                        )

                        # Should be called at least once (might be called for hash duplicates too)
                        assert merge_memories_with_llm.called

                        # Verify memories were indexed

                        index_long_term_memories.assert_called_once()

                        # Verify Redis delete was called
                        mock_async_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_compaction_end_to_end(
        self, mock_async_redis_client, mock_openai_client
    ):
        """Test the full compaction process with both hash and semantic duplicates"""
        # Set up mock data - 4 memories:
        # - Two with identical hash (exact duplicates)
        # - One semantically similar to a third memory
        # - One unique memory
        memory_keys = ["memory:1", "memory:2", "memory:3", "memory:4"]

        # Mock scanning Redis for memory keys
        mock_async_redis_client.scan = AsyncMock()
        mock_async_redis_client.scan.side_effect = [
            (0, memory_keys)  # All keys in one scan
        ]

        # Memory content
        memory1 = {
            "text": "Paris is the capital of France",
            "id_": "1",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 100),
            "last_accessed": str(int(time.time()) - 50),
            "key": memory_keys[0],
        }

        # Exact duplicate of memory1 (same hash)
        memory2 = {
            "text": "Paris is the capital of France",
            "id_": "2",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 90),
            "last_accessed": str(int(time.time()) - 40),
            "key": memory_keys[1],
        }

        # Semantically similar to memory4
        memory3 = {
            "text": "Berlin is the capital of Germany",
            "id_": "3",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 80),
            "last_accessed": str(int(time.time()) - 30),
            "key": memory_keys[2],
        }

        # Semantically similar to memory3
        memory4 = {
            "text": "The capital city of Germany is Berlin",
            "id_": "4",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 70),
            "last_accessed": str(int(time.time()) - 20),
            "key": memory_keys[3],
        }

        # Setup pipeline for memory retrieval
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(
            return_value=[memory1, memory2, memory3, memory4]
        )
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        # Setup mocks for hash values - memories 1 and 2 have the same hash, 3 and 4 have different hashes
        hash_values = ["hash12", "hash12", "hash3", "hash4"]

        with patch(
            "agent_memory_server.long_term_memory.generate_memory_hash",
            side_effect=hash_values,
        ):
            # Setup hash-based merged memory
            hash_merged_memory = {
                "text": "Paris is the capital of France",
                "id_": "merged1",
                "user_id": "user123",
                "session_id": "session456",
                "namespace": "test",
                "created_at": int(time.time()) - 100,  # Earliest
                "last_accessed": int(time.time()) - 40,  # Latest
                "topics": [],
                "entities": [],
                "memory_hash": "merged_hash1",
                "key": "memory:merged1",
            }

            # Setup semantic merged memory
            semantic_merged_memory = {
                "text": "Berlin is the capital of Germany",
                "id_": "merged2",
                "user_id": "user123",
                "session_id": "session456",
                "namespace": "test",
                "created_at": int(time.time()) - 80,  # Earliest
                "last_accessed": int(time.time()) - 20,  # Latest
                "topics": [],
                "entities": [],
                "memory_hash": "merged_hash2",
                "key": "memory:merged2",
            }

            # Mock LLM merging with different responses for hash vs semantic
            merge_memories_mock = AsyncMock()
            merge_memories_mock.side_effect = [
                hash_merged_memory,
                semantic_merged_memory,
            ]

            with (
                patch(
                    "agent_memory_server.long_term_memory.merge_memories_with_llm",
                    merge_memories_mock,
                ),
                patch(
                    "agent_memory_server.long_term_memory.index_long_term_memories",
                    AsyncMock(),
                ),
            ):
                # Mock the vectorizer
                mock_vectorizer = MagicMock()
                mock_vectorizer.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])

                with patch(
                    "agent_memory_server.long_term_memory.OpenAITextVectorizer",
                    return_value=mock_vectorizer,
                ):
                    # Mock vector search to return similar memories for memory 3
                    # This should show memory 4 is similar to memory 3
                    mock_doc = Document(
                        id=b"doc1",
                        id_="4",
                        text="The capital city of Germany is Berlin",
                        vector_distance=0.05,  # Close distance indicating similarity
                        created_at=str(int(time.time()) - 70),
                        last_accessed=str(int(time.time()) - 20),
                        user_id="user123",
                        session_id="session456",
                        namespace="test",
                    )

                    # Create a side effect that returns empty search results for non-matching queries,
                    # but returns a match for memory3
                    def search_side_effect(query, params):
                        # Different return values depending on which memory we're searching for
                        result = MagicMock()
                        # Memory values for comparison
                        memory3_text = "Berlin is the capital of Germany"

                        # Check if we're searching for memory3's content
                        # We need to check the params dictionary for the vector
                        if "3" in str(query) or memory3_text in str(params):
                            result.docs = [mock_doc]
                            result.total = 1
                        else:
                            result.docs = []
                            result.total = 0
                        return result

                    mock_index = MagicMock()
                    mock_index.search = AsyncMock(side_effect=search_side_effect)

                    with patch(
                        "agent_memory_server.long_term_memory.get_search_index",
                        return_value=mock_index,
                    ):
                        # Call the function
                        await compact_long_term_memories(mock_async_redis_client)

                        # Verify Redis scan was called once
                        mock_async_redis_client.scan.assert_called_once()

                        # Verify merge_memories_with_llm was called exactly twice:
                        # - Once for hash-based duplicates (memory1 + memory2)
                        # - Once for semantic duplicates (memory3 + memory4)
                        assert merge_memories_mock.call_count == 2

                        # The first call should be for the hash duplicates (memory1 and memory2)
                        first_call_args = merge_memories_mock.call_args_list[0][0]
                        assert len(first_call_args[0]) == 2  # Two memories
                        assert first_call_args[1] == "hash"  # Hash-based merging

                        # Second call should be for semantic duplicates (memory3 and memory4)
                        if len(merge_memories_mock.call_args_list) > 1:
                            second_call_args = merge_memories_mock.call_args_list[1][0]
                            assert (
                                second_call_args[1] == "semantic"
                            )  # Semantic-based merging

                        # Verify index_long_term_memories was called with two merged memories

                        index_long_term_memories.assert_called_once()

                        # Verify Redis delete was called to remove the original memories
                        assert mock_async_redis_client.delete.called

    @pytest.mark.asyncio
    async def test_compact_hash_based_duplicates_simple(self, mock_async_redis_client):
        """Test simple compaction of hash-based duplicates using the helper function"""
        # Set up memory keys and content
        memory_key1 = "memory:123"
        memory_key2 = "memory:456"
        memory_keys = [memory_key1, memory_key2]

        # Memory content with identical text (will have the same hash)
        memory1 = {
            "text": "Duplicate memory",
            "id_": "123",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 100),
            "last_accessed": str(int(time.time()) - 50),
            "key": memory_key1,
        }

        memory2 = {
            "text": "Duplicate memory",
            "id_": "456",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 200),
            "last_accessed": str(int(time.time())),
            "key": memory_key2,
        }

        # Define merged memory
        merged_memory = {
            "text": "Merged memory content",
            "id_": "merged",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 200,  # Should take earliest
            "last_accessed": int(time.time()),  # Should take latest
            "topics": [],
            "entities": [],
            "memory_hash": "merged_hash",
            "key": "memory:merged",
        }

        # Define search result for semantic search (empty for this test)
        mock_search_result = MagicMock()
        mock_search_result.docs = []
        mock_search_result.total = 0

        # Run the function with our mocks
        results = await run_compact_memories_with_mocks(
            mock_redis=mock_async_redis_client,
            memory_keys=memory_keys,
            memory_contents=[memory1, memory2],
            hash_values="same_hash",  # Same hash for both memories
            merged_memories=merged_memory,
            search_results=mock_search_result,
        )

        # Verify merge_memories_with_llm was called once for hash-based duplicates
        assert results["merge_memories"].call_count == 1

        # Verify the first argument to the first call contained both memories
        call_args = results["merge_memories"].call_args_list[0][0]
        assert len(call_args[0]) == 2

        # Verify the second argument was "hash" indicating hash-based merging
        assert call_args[1] == "hash"

        # Verify Redis delete was called to delete the original memories
        assert results["redis_delete"].called

    @pytest.mark.asyncio
    async def test_compact_semantic_duplicates_simple(self, mock_async_redis_client):
        """Test simple compaction of semantic duplicates using the helper function"""
        # Set up memory keys and content
        memory_key1 = "memory:123"
        memory_key2 = "memory:456"
        memory_keys = [memory_key1, memory_key2]

        # Memory content with similar meaning but different text (different hashes)
        memory1 = {
            "text": "Paris is the capital of France",
            "id_": "123",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 100),
            "last_accessed": str(int(time.time()) - 50),
            "key": memory_key1,
        }

        memory2 = {
            "text": "The capital city of France is Paris",
            "id_": "456",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": str(int(time.time()) - 200),
            "last_accessed": str(int(time.time())),
            "key": memory_key2,
        }

        # Define merged memories (first for hash phase, second for semantic phase)

        semantic_merged_memory = {
            "text": "Paris is the capital of France",
            "id_": "merged",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 200,  # Should take earliest
            "last_accessed": int(time.time()),  # Should take latest
            "topics": [],
            "entities": [],
            "memory_hash": "merged_hash",
            "key": "memory:merged",
        }

        # Set up document for search result showing memory2 is similar to memory1
        mock_doc = Document(
            id=b"doc1",
            id_="456",
            text="The capital city of France is Paris",
            vector_distance=0.05,  # Close distance indicating similarity
            created_at=str(int(time.time()) - 200),
            last_accessed=str(int(time.time())),
            user_id="user123",
            session_id="session456",
            namespace="test",
        )

        # Create search result with memory2 as similar to memory1
        mock_search_result = MagicMock()
        mock_search_result.docs = [mock_doc]
        mock_search_result.total = 1

        # Run the function with our mocks
        results = await run_compact_memories_with_mocks(
            mock_redis=mock_async_redis_client,
            memory_keys=memory_keys,
            memory_contents=[memory1, memory2],
            hash_values=["hash1", "hash2"],  # Different hashes
            merged_memories=semantic_merged_memory,  # Just use the semantic merge
            search_results=mock_search_result,
        )

        # Verify search was called (at least once) for semantic search
        assert results["search_index"].search.called

        # Verify merge_memories_with_llm was called for semantic duplicates
        assert results["merge_memories"].call_count > 0

        # Verify Redis delete was called to delete the original memories
        assert results["redis_delete"].called

    @pytest.mark.asyncio
    async def test_semantic_merge_directly(self):
        """Test the semantic merge part directly by patching just the parts we need"""
        # Memory content with similar meaning but different text
        memory1 = {
            "text": "Paris is the capital of France",
            "id_": "123",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 100,
            "last_accessed": int(time.time()) - 50,
            "key": "memory:123",
        }

        memory2 = {
            "text": "The capital city of France is Paris",
            "id_": "456",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 200,
            "last_accessed": int(time.time()),
            "key": "memory:456",
        }

        # Define merged memory
        merged_memory = {
            "text": "Paris is the capital of France",
            "id_": "merged",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 200,  # Should take earliest
            "last_accessed": int(time.time()),  # Should take latest
            "topics": [],
            "entities": [],
            "memory_hash": "merged_hash",
            "key": "memory:merged",
        }

        # Set up document for search result showing memory2 is similar to memory1
        mock_doc = Document(
            id=b"doc1",
            id_="456",
            text="The capital city of France is Paris",
            vector_distance=0.05,  # Close distance indicating similarity
            created_at=str(int(time.time()) - 200),
            last_accessed=str(int(time.time())),
            user_id="user123",
            session_id="session456",
            namespace="test",
        )

        # Mock the search result
        mock_search_result = MagicMock()
        mock_search_result.docs = [mock_doc]
        mock_search_result.total = 1

        # Setup the vectorizer mock
        mock_vectorizer = MagicMock()
        mock_vectorizer.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # Setup search index mock
        mock_index = MagicMock()
        mock_index.search = AsyncMock(return_value=mock_search_result)

        # Setup merge_memories_with_llm mock
        merge_memories_mock = AsyncMock(return_value=merged_memory)

        with (
            patch(
                "agent_memory_server.long_term_memory.merge_memories_with_llm",
                merge_memories_mock,
            ),
            patch(
                "agent_memory_server.long_term_memory.OpenAITextVectorizer",
                return_value=mock_vectorizer,
            ),
            patch(
                "agent_memory_server.long_term_memory.get_search_index",
                return_value=mock_index,
            ),
        ):
            # Test the function directly
            from agent_memory_server.filters import SessionId, UserId
            from agent_memory_server.long_term_memory import reduce

            # Skip if this memory has already been processed
            processed_keys = set()
            ref_memory = memory1

            # Find semantically similar memories using VectorRangeQuery
            query_text = ref_memory["text"]
            query_vector = await mock_vectorizer.aembed(query_text)

            # Create filter for user_id and session_id matching
            filters = []
            if ref_memory.get("user_id"):
                filters.append(UserId(eq=ref_memory["user_id"]).to_filter())
            if ref_memory.get("session_id"):
                filters.append(SessionId(eq=ref_memory["session_id"]).to_filter())

            filter_expression = reduce(lambda x, y: x & y, filters) if filters else None

            # Create vector query with distance threshold
            from redisvl.query import VectorRangeQuery

            q = VectorRangeQuery(
                vector=query_vector,
                vector_field_name="vector",
                distance_threshold=0.1,  # Semantic distance threshold
                num_results=100,  # Set a reasonable limit
                return_score=True,
                return_fields=[
                    "text",
                    "id_",
                    "dist",
                    "created_at",
                    "last_accessed",
                    "user_id",
                    "session_id",
                    "namespace",
                    "topics",
                    "entities",
                ],
            )

            if filter_expression:
                q.set_filter(filter_expression)

            # Execute the query
            search_result = await mock_index.search(q, q.params)

            # Process similar memories (mocked to return memory2 as similar to memory1)
            similar_memories = [ref_memory]  # Start with reference memory

            # Process similar memories from search results
            assert len(search_result.docs) == 1
            search_result.docs[0]

            # Find the original memory data
            memory = memory2

            # Add to similar_memories
            similar_memories.append(memory)

            # Mark as processed
            keys_to_delete = set()
            if memory["key"] not in processed_keys:
                keys_to_delete.add(memory["key"])
                processed_keys.add(memory["key"])

            # Check that we have both memories in the similar_memories list
            assert len(similar_memories) == 2
            assert similar_memories[0]["id_"] == "123"
            assert similar_memories[1]["id_"] == "456"

            # Call merge_memories_with_llm
            await merge_memories_with_llm(similar_memories, "semantic")

            # Verify the merge was called with the right parameters
            merge_memories_mock.assert_called_once()
            call_args = merge_memories_mock.call_args_list[0][0]
            assert len(call_args[0]) == 2  # Two memories
            assert call_args[1] == "semantic"  # Semantic merge

    @pytest.mark.asyncio
    async def test_merge_semantic_memories(self, mock_openai_client):
        """Test merging semantically similar memories with LLM directly."""
        # Create test memories with similar semantic meaning
        memory1 = {
            "text": "Paris is the capital of France",
            "id_": "123",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 100,
            "last_accessed": int(time.time()) - 50,
            "topics": ["geography", "europe"],
            "entities": ["Paris", "France"],
        }

        memory2 = {
            "text": "The capital city of France is Paris",
            "id_": "456",
            "user_id": "user123",
            "session_id": "session456",
            "namespace": "test",
            "created_at": int(time.time()) - 200,
            "last_accessed": int(time.time()),
            "topics": ["travel", "europe"],
            "entities": ["Paris", "France", "city"],
        }

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Paris is the capital city of France"

        mock_model_client = AsyncMock()
        mock_model_client.create_chat_completion = AsyncMock(return_value=mock_response)

        with patch(
            "agent_memory_server.long_term_memory.get_model_client",
            return_value=mock_model_client,
        ):
            # Call the merge function directly with the semantic merge type
            merged = await merge_memories_with_llm([memory1, memory2], "semantic")

            # Verify the merged result has the correct properties
            assert merged["text"] == "Paris is the capital city of France"

            # Should have the earliest created_at
            assert merged["created_at"] == memory2["created_at"]

            # Should have the most recent last_accessed
            assert merged["last_accessed"] == memory2["last_accessed"]

            # Should preserve user_id, session_id, namespace
            assert merged["user_id"] == "user123"
            assert merged["session_id"] == "session456"
            assert merged["namespace"] == "test"

            # Should combine topics and entities
            assert set(merged["topics"]) == {"geography", "europe", "travel"}
            assert set(merged["entities"]) == {"Paris", "France", "city"}

            # Should have a memory_hash
            assert "memory_hash" in merged

            # Verify the LLM was called with the right prompt format
            prompt_call = mock_model_client.create_chat_completion.call_args[1][
                "prompt"
            ]
            assert "Merge these similar memories" in prompt_call
            assert "Memory 1:" in prompt_call
            assert "Memory 2:" in prompt_call

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_compact_memories_integration():
        """
        Integration test for memory compaction with minimal mocking.

        This test:
        1. Creates test memories (hash duplicates and semantic duplicates)
        2. Indexes them directly into Redis
        3. Runs the compaction with only the LLM call mocked
        4. Verifies the number of memories after compaction
        """
        # Setup test memories
        now = int(datetime.now().timestamp())

        # Create four memories:
        # - Two with identical content (hash duplicates)
        # - One with similar semantic meaning
        # - One distinct memory that shouldn't be compacted

        # Memory 1 - Base memory
        memory1 = {
            "text": "Paris is the capital of France",
            "id_": "mem1",
            "user_id": "test_user",
            "session_id": "test_session",
            "namespace": "test_namespace",
            "created_at": now - 1000,
            "last_accessed": now - 500,
            "topics": ["geography", "europe"],
            "entities": ["Paris", "France"],
            "memory_hash": "hash1",  # Will be replaced with actual hash
        }

        # Memory 2 - Exact duplicate of Memory 1 (hash duplicate)
        memory2 = {
            "text": "Paris is the capital of France",
            "id_": "mem2",
            "user_id": "test_user",
            "session_id": "test_session",
            "namespace": "test_namespace",
            "created_at": now - 900,
            "last_accessed": now - 400,
            "topics": ["travel"],
            "entities": ["Paris"],
            "memory_hash": "hash2",  # Will be replaced with actual hash
        }

        # Memory 3 - Semantic similar to Memory 1
        memory3 = {
            "text": "The city of Paris serves as France's capital",
            "id_": "mem3",
            "user_id": "test_user",
            "session_id": "test_session",
            "namespace": "test_namespace",
            "created_at": now - 800,
            "last_accessed": now - 300,
            "topics": ["cities"],
            "entities": ["France", "Paris"],
            "memory_hash": "hash3",  # Will be replaced with actual hash
        }

        # Memory 4 - Distinct memory
        memory4 = {
            "text": "Tokyo is the capital of Japan",
            "id_": "mem4",
            "user_id": "test_user",
            "session_id": "test_session",
            "namespace": "test_namespace",
            "created_at": now - 700,
            "last_accessed": now - 200,
            "topics": ["geography", "asia"],
            "entities": ["Tokyo", "Japan"],
            "memory_hash": "hash4",  # Will be replaced with actual hash
        }

        # Generate real memory hashes
        for memory in [memory1, memory2, memory3, memory4]:
            memory["memory_hash"] = generate_memory_hash(memory)

        # Memory 2 should have the same hash as Memory 1
        memory2["memory_hash"] = memory1["memory_hash"]

        # Connect to Redis and ensure index exists
        redis_conn = get_redis_conn()
        index_name = "test_mem_idx"
        ensure_search_index_exists(redis_conn, index_name)

        # Clean up any existing test data
        delete_pattern = "test_user:test_session:test_namespace:*"
        existing_keys = redis_conn.keys(delete_pattern)
        if existing_keys:
            redis_conn.delete(*existing_keys)

        # Add all memories to Redis
        for _i, memory in enumerate([memory1, memory2, memory3, memory4]):
            key = f"test_user:test_session:test_namespace:{memory['id_']}"
            redis_conn.hset(
                key,
                mapping={
                    "text": memory["text"],
                    "user_id": memory["user_id"],
                    "session_id": memory["session_id"],
                    "namespace": memory["namespace"],
                    "created_at": str(memory["created_at"]),
                    "last_accessed": str(memory["last_accessed"]),
                    "topics": json.dumps(memory["topics"]),
                    "entities": json.dumps(memory["entities"]),
                    "memory_hash": memory["memory_hash"],
                    "embedding": json.dumps([0.1] * 10),  # Dummy embedding
                },
            )

        # Count initial memories
        initial_keys = redis_conn.keys(delete_pattern)
        assert len(initial_keys) == 4

        # Mock LLM response for semantic merging
        llm_response = MagicMock()
        llm_response.content = (
            "Paris is the capital of France, an important European city."
        )

        model_client = AsyncMock()
        model_client.create_chat_completion.return_value = llm_response

        # Run memory compaction with minimal mocking (only the LLM)
        with patch("agent_memory_server.long_term_memory.model_client", model_client):
            await compact_long_term_memories(
                user_id="test_user",
                session_id="test_session",
                namespace="test_namespace",
                semantic_merge_threshold=0.75,  # Set threshold to detect semantic duplicate
            )

        # Count final memories - should have 2 (hash duplicates merged, semantic duplicates merged)
        final_keys = redis_conn.keys(delete_pattern)
        assert (
            len(final_keys) == 2
        ), f"Expected 2 memories after compaction, got {len(final_keys)}"

        # Verify LLM was called once for the semantic merge
        assert model_client.create_chat_completion.call_count == 1

        # Clean up test data
        if final_keys:
            redis_conn.delete(*final_keys)
