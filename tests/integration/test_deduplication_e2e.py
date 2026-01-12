"""
End-to-End Integration Tests for Memory Deduplication

This module tests the complete deduplication flow with real OpenAI embeddings
and Redis to verify that duplicate memories are correctly detected and merged.

Requirements:
- OPENAI_API_KEY environment variable set
- Redis testcontainer (automatically started by conftest.py)
- Run with: uv run pytest tests/integration/test_deduplication_e2e.py --run-api-tests -v

Test Coverage:
- Paraphrased memories are merged (GitHub Issue #110)
- Distinct memories are kept separate
- Hybrid search catches more duplicates than vector-only search
- Threshold boundary testing
"""

import asyncio
import os
from unittest.mock import patch

import pytest
import ulid

from agent_memory_server.config import settings
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
    """End-to-end tests for memory deduplication with real embeddings."""

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

        # Create third paraphrased memory
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

        assert len(results.memories) == 1, (
            f"Expected 1 search result, got {len(results.memories)}"
        )

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

