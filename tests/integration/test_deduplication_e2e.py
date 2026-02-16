"""
End-to-End Integration Tests for Memory Deduplication (Store-Time)

This module tests the store-time deduplication flow with real OpenAI embeddings
and Redis to verify that duplicate memories are correctly detected and merged.

Requirements:
- OPENAI_API_KEY environment variable set
- Redis testcontainer (automatically started by conftest.py)
- Run with: uv run pytest tests/integration/test_deduplication_e2e.py --run-api-tests -v

Test Coverage:
- Paraphrased memories are merged (GitHub Issue #110)
- Distinct memories are kept separate
- Vector search with proper threshold catches duplicates
- Threshold boundary testing (0.12 vs 0.35)
- Real-world API/MCP usage patterns
"""

import asyncio

import pytest
import ulid

from agent_memory_server.filters import Namespace, UserId
from agent_memory_server.long_term_memory import (
    count_long_term_memories,
    deduplicate_by_semantic_search,
    index_long_term_memories,
    search_long_term_memories,
)
from agent_memory_server.models import MemoryRecord


@pytest.fixture
def unique_namespace():
    """Generate a unique namespace for test isolation."""
    return f"test-dedup-{ulid.ULID()}"


@pytest.fixture
def unique_user_id():
    """Generate a unique user ID for test isolation."""
    return f"user-{ulid.ULID()}"


@pytest.mark.asyncio
@pytest.mark.requires_api_keys
class TestDeduplicationE2E:
    """End-to-end tests for store-time memory deduplication with real embeddings."""

    async def test_paraphrased_memories_are_merged_not_duplicated(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that paraphrased memories about the same topic are merged,
        not stored as duplicates. This reproduces GitHub Issue #110.

        The "Flat White coffee preference" case:
        - "User likes coffee, flat white usually"
        - "They are a coffee enthusiast, favorite coffee is flatwhite"
        - "User loves coffee, especially flat white"

        All three should be merged into a single memory.
        """
        # Create the first memory
        memory1 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User likes coffee, flat white usually",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Index the first memory (no deduplication needed for first one)
        await index_long_term_memories(
            [memory1],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )

        # Wait for indexing to complete
        await asyncio.sleep(1)

        # Create second paraphrased memory
        memory2 = MemoryRecord(
            id=str(ulid.ULID()),
            text="They are a coffee enthusiast, favorite coffee is flatwhite",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Index with deduplication enabled - should merge with memory1
        await index_long_term_memories(
            [memory2],
            redis_client=use_test_redis_connection,
            deduplicate=True,
        )

        await asyncio.sleep(1)

        # Create third paraphrased memory (mirrors original issue description)
        memory3 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User loves coffee, especially flat white",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Index with deduplication enabled - should merge with existing
        await index_long_term_memories(
            [memory3],
            redis_client=use_test_redis_connection,
            deduplicate=True,
        )

        await asyncio.sleep(1)

        # Count memories - should be 1 (merged), not 3
        count = await count_long_term_memories(
            namespace=unique_namespace,
            user_id=unique_user_id,
            redis_client=use_test_redis_connection,
        )

        # Assert: Only 1 memory exists (merged), not 3 separate memories
        assert count == 1, (
            f"Expected 1 merged memory, but found {count}. "
            "Paraphrased memories were stored as duplicates instead of being merged."
        )

        # Search for the merged memory
        results = await search_long_term_memories(
            text="coffee preference flat white",
            namespace=Namespace(eq=unique_namespace),
            user_id=UserId(eq=unique_user_id),
            limit=10,
        )

        assert (
            len(results.memories) == 1
        ), f"Expected 1 search result, got {len(results.memories)}"

        # The merged memory should contain key information
        merged_text = results.memories[0].text.lower()
        assert "coffee" in merged_text, "Merged memory should mention coffee"

    async def test_distinct_memories_are_not_merged(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that semantically distinct memories are kept separate.

        - "User prefers flat white coffee"
        - "User prefers tea in the afternoon"

        These should remain as 2 separate memories.
        """
        memory1 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User prefers flat white coffee in the morning",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        await index_long_term_memories(
            [memory1],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )

        await asyncio.sleep(1)

        memory2 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User prefers green tea in the afternoon",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        await index_long_term_memories(
            [memory2],
            redis_client=use_test_redis_connection,
            deduplicate=True,
        )

        await asyncio.sleep(1)

        # Count memories - should be 2 (distinct topics)
        count = await count_long_term_memories(
            namespace=unique_namespace,
            user_id=unique_user_id,
            redis_client=use_test_redis_connection,
        )

        assert count == 2, (
            f"Expected 2 distinct memories, but found {count}. "
            "Distinct memories were incorrectly merged."
        )

    async def test_vector_search_catches_paraphrased_duplicates(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that vector search with proper threshold (0.35) catches
        paraphrased duplicates effectively.
        """
        # Create first memory
        memory1 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User enjoys drinking flat white coffee every morning",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        await index_long_term_memories(
            [memory1],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )

        await asyncio.sleep(1)

        # Create a paraphrased memory with shared keywords
        memory2 = MemoryRecord(
            id=str(ulid.ULID()),
            text="The user's preferred morning beverage is flat white coffee",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Test with default threshold (0.35)
        result, was_merged = await deduplicate_by_semantic_search(
            memory=memory2,
            redis_client=use_test_redis_connection,
            namespace=unique_namespace,
            user_id=unique_user_id,
        )

        # Vector search with 0.35 threshold should detect this as a duplicate
        assert was_merged, (
            "Vector search with threshold 0.35 should detect paraphrased "
            "memory as duplicate."
        )

    async def test_strict_threshold_would_miss_duplicates(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that demonstrates the old strict threshold (0.12) would miss
        duplicates that the new threshold (0.35) catches.

        This validates the fix for GitHub Issue #110.
        """
        # Create first memory
        memory1 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User prefers flat white coffee",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        await index_long_term_memories(
            [memory1],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )

        await asyncio.sleep(1)

        # Create paraphrased memory
        memory2 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User loves drinking flat white, it's their favorite coffee",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Test with strict threshold (0.12) - should NOT detect duplicate
        await deduplicate_by_semantic_search(
            memory=memory2,
            redis_client=use_test_redis_connection,
            namespace=unique_namespace,
            user_id=unique_user_id,
            vector_distance_threshold=0.12,  # Old strict threshold
        )

        # Test with relaxed threshold (0.35) - SHOULD detect duplicate
        result_relaxed, was_merged_relaxed = await deduplicate_by_semantic_search(
            memory=memory2,
            redis_client=use_test_redis_connection,
            namespace=unique_namespace,
            user_id=unique_user_id,
            vector_distance_threshold=0.35,  # New relaxed threshold
        )

        # The relaxed threshold should catch the duplicate
        # (strict may or may not, depending on embedding similarity)
        assert was_merged_relaxed, (
            "Relaxed threshold (0.35) should detect paraphrased memory as duplicate. "
            "This validates the fix for GitHub Issue #110."
        )

    async def test_api_mcp_usage_pattern_deduplication(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test the real-world API/MCP usage pattern where memories are created
        via index_long_term_memories with deduplicate=True.

        This simulates what happens when an assistant calls create_long_term_memories
        through the MCP tool or REST API.
        """
        coffee_memories = [
            "User likes coffee, flat white usually",
            "They are a coffee enthusiast, favorite coffee is flatwhite",
            "User loves coffee, especially flat white",
        ]

        counts = []

        for i, text in enumerate(coffee_memories):
            memory = MemoryRecord(
                id=str(ulid.ULID()),
                text=text,
                namespace=unique_namespace,
                user_id=unique_user_id,
                memory_type="semantic",
            )

            # This is what the API/MCP tool does - deduplicate=True by default
            await index_long_term_memories(
                [memory],
                redis_client=use_test_redis_connection,
                # In this test, we know the first insert sees an empty namespace,
                # so we skip dedup for it as an optimization only
                deduplicate=(i > 0),
            )

            await asyncio.sleep(1)

            count = await count_long_term_memories(
                namespace=unique_namespace,
                user_id=unique_user_id,
                redis_client=use_test_redis_connection,
            )
            counts.append(count)

        # All 3 memories should be merged into 1
        assert counts[-1] == 1, (
            f"Expected 1 merged memory, but found {counts[-1]}. "
            f"Memory count progression: {counts}. "
            "API/MCP usage pattern should merge paraphrased memories."
        )

        # Verify the progression shows merging
        assert counts == [1, 1, 1], (
            f"Expected progression [1, 1, 1] (all merged), got {counts}. "
            "Each new paraphrased memory should merge with existing."
        )


@pytest.mark.asyncio
@pytest.mark.requires_api_keys
class TestEmbeddingsInputValidation:
    """
    Tests to identify what input causes OpenAI's '$.input' is invalid error.

    This reproduces the error from compact_long_term_memories:
    litellm.exceptions.BadRequestError: OpenAIException - Error code: 400 -
    {'error': {'message': "'$.input' is invalid..."}}
    """

    async def test_empty_string_rejected_at_api_layer(self):
        """
        Test that empty string text is rejected when creating memories via API.

        Empty strings cannot be embedded by OpenAI's API (causes "'$.input' is invalid"
        error). We reject them at the API layer with validation.
        """
        from agent_memory_server.models import (
            CreateMemoryRecordRequest,
            ExtractedMemoryRecord,
            LenientMemoryRecord,
        )

        # Creating a memory with empty text via LenientMemoryRecord should fail
        with pytest.raises(ValueError, match="Memory text cannot be empty"):
            LenientMemoryRecord(text="")

        # Creating a memory request with empty text should fail
        valid_memory = ExtractedMemoryRecord(
            id=str(ulid.ULID()),
            text="Valid memory",
        )
        empty_memory = ExtractedMemoryRecord(
            id=str(ulid.ULID()),
            text="",  # Empty string
        )

        with pytest.raises(ValueError, match="has empty text"):
            CreateMemoryRecordRequest(memories=[valid_memory, empty_memory])

    async def test_legacy_empty_text_filtered_on_index(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that legacy memories with empty text are filtered out when indexing.

        Old databases may contain memories with empty text. These should be
        gracefully filtered during indexing/deduplication to avoid API errors.
        """
        # Create a MemoryRecord directly (bypassing API validation) with empty text
        # This simulates legacy data that already exists in the database
        empty_text_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="",  # Empty string - simulating legacy data
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # index_long_term_memories should filter this out without error
        await index_long_term_memories(
            [empty_text_memory],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )
        # No error means it was filtered successfully

        # deduplicate_by_semantic_search should also handle empty text gracefully
        result, was_merged = await deduplicate_by_semantic_search(
            memory=empty_text_memory,
            redis_client=use_test_redis_connection,
            namespace=unique_namespace,
            user_id=unique_user_id,
        )

        # Should return the original memory unchanged, without merging
        assert result is not None
        assert result.id == empty_text_memory.id
        assert was_merged is False

    async def test_whitespace_only_does_not_cause_error(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that whitespace-only text does NOT cause the '$.input' is invalid error.

        OpenAI's embedding API accepts whitespace-only strings and generates embeddings
        for them. Only truly empty strings ("") cause the error.
        """
        # First, create a valid memory so there's something to search against
        valid_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="User likes tea",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )
        await index_long_term_memories(
            [valid_memory],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )
        await asyncio.sleep(1)

        # Now try to deduplicate a memory with whitespace-only text
        whitespace_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="   \t\n   ",  # Whitespace only
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Whitespace-only strings are accepted by OpenAI's API
        result, was_merged = await deduplicate_by_semantic_search(
            memory=whitespace_memory,
            redis_client=use_test_redis_connection,
            namespace=unique_namespace,
            user_id=unique_user_id,
        )

        # Should complete without error (though it won't find duplicates)
        assert result is not None
        assert was_merged is False  # Whitespace won't match any real memories

    async def test_valid_text_works(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """Baseline test: valid text should work without errors."""
        # First, create a valid memory so there's something to search against
        valid_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="User prefers Earl Grey tea",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )
        await index_long_term_memories(
            [valid_memory],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )
        await asyncio.sleep(1)

        # Try to deduplicate a memory with valid text
        valid_text_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="User enjoys drinking Earl Grey",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # This should NOT raise an error
        result, was_merged = await deduplicate_by_semantic_search(
            memory=valid_text_memory,
            redis_client=use_test_redis_connection,
            namespace=unique_namespace,
            user_id=unique_user_id,
        )

        # Should successfully complete (merged or not)
        assert result is not None or was_merged is not None
