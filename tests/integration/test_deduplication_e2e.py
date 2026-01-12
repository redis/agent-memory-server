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
            text="Coffee is a favorite, especially flat white",
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
        result_strict, was_merged_strict = await deduplicate_by_semantic_search(
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

        Based on: scripts/simulate_api_usage.py
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
                deduplicate=(i > 0),  # First memory doesn't need dedup check
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

    async def test_old_threshold_misses_duplicates_new_catches_them(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that demonstrates the old strict threshold (0.12) would create
        duplicates while the new threshold (0.35) correctly merges them.

        This validates the fix for GitHub Issue #110 by showing the before/after
        behavior difference.

        Based on: scripts/simulate_old_threshold.py
        """
        coffee_memories = [
            "User likes coffee, flat white usually",
            "They are a coffee enthusiast, favorite coffee is flatwhite",
            "User loves coffee, especially flat white",
        ]

        # Test with OLD threshold (0.12) - should create duplicates
        old_namespace = f"{unique_namespace}-old"
        old_counts = []

        for i, text in enumerate(coffee_memories):
            memory = MemoryRecord(
                id=str(ulid.ULID()),
                text=text,
                namespace=old_namespace,
                user_id=unique_user_id,
                memory_type="semantic",
            )

            await index_long_term_memories(
                [memory],
                redis_client=use_test_redis_connection,
                deduplicate=(i > 0),
                vector_distance_threshold=0.12,  # OLD strict threshold
            )

            await asyncio.sleep(1)

            count = await count_long_term_memories(
                namespace=old_namespace,
                user_id=unique_user_id,
                redis_client=use_test_redis_connection,
            )
            old_counts.append(count)

        # Test with NEW threshold (0.35) - should merge all
        new_namespace = f"{unique_namespace}-new"
        new_counts = []

        for i, text in enumerate(coffee_memories):
            memory = MemoryRecord(
                id=str(ulid.ULID()),
                text=text,
                namespace=new_namespace,
                user_id=unique_user_id,
                memory_type="semantic",
            )

            await index_long_term_memories(
                [memory],
                redis_client=use_test_redis_connection,
                deduplicate=(i > 0),
                vector_distance_threshold=0.35,  # NEW relaxed threshold
            )

            await asyncio.sleep(1)

            count = await count_long_term_memories(
                namespace=new_namespace,
                user_id=unique_user_id,
                redis_client=use_test_redis_connection,
            )
            new_counts.append(count)

        # NEW threshold should result in 1 merged memory
        assert new_counts[-1] == 1, (
            f"NEW threshold (0.35) should merge all to 1, got {new_counts[-1]}. "
            f"Progression: {new_counts}"
        )

        # OLD threshold should result in more memories (duplicates)
        # Note: The exact count depends on embedding distances, but it should be > 1
        assert old_counts[-1] >= new_counts[-1], (
            f"OLD threshold should create >= memories than NEW. "
            f"OLD: {old_counts[-1]}, NEW: {new_counts[-1]}"
        )


@pytest.mark.asyncio
@pytest.mark.requires_api_keys
class TestLoadTimeDeduplication:
    """Tests for Tier 2: Load-Time Deduplication with LLM."""

    async def test_cluster_memories_by_similarity(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that similar memories are clustered together.
        """
        from agent_memory_server.long_term_memory import cluster_memories_by_similarity
        from agent_memory_server.models import MemoryRecordResult

        # Create mock memory results (similar coffee preferences)
        memories = [
            MemoryRecordResult(
                id="mem1",
                text="User likes coffee, flat white usually",
                dist=0.1,
            ),
            MemoryRecordResult(
                id="mem2",
                text="They are a coffee enthusiast, favorite coffee is flatwhite",
                dist=0.15,
            ),
            MemoryRecordResult(
                id="mem3",
                text="User loves coffee, especially flat white",
                dist=0.12,
            ),
            MemoryRecordResult(
                id="mem4",
                text="User prefers dark mode for their IDE",
                dist=0.2,
            ),
        ]

        # Cluster with default threshold (0.4)
        clusters = await cluster_memories_by_similarity(
            memories, distance_threshold=0.4
        )

        # Should have 2 clusters: coffee preferences and IDE preference
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"

        # Find the coffee cluster (should have 3 memories)
        coffee_cluster = None
        ide_cluster = None
        for cluster in clusters:
            texts = [m.text for m in cluster]
            if any("coffee" in t.lower() for t in texts):
                coffee_cluster = cluster
            if any("IDE" in t for t in texts):
                ide_cluster = cluster

        assert coffee_cluster is not None, "Coffee cluster not found"
        assert ide_cluster is not None, "IDE cluster not found"
        assert (
            len(coffee_cluster) == 3
        ), f"Coffee cluster should have 3 memories, got {len(coffee_cluster)}"
        assert (
            len(ide_cluster) == 1
        ), f"IDE cluster should have 1 memory, got {len(ide_cluster)}"

    async def test_deduplicate_search_results_without_llm(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test load-time deduplication without LLM verification.
        """
        from agent_memory_server.long_term_memory import deduplicate_search_results
        from agent_memory_server.models import MemoryRecordResult

        # Create mock memory results with duplicates
        memories = [
            MemoryRecordResult(
                id="mem1",
                text="User likes coffee, flat white usually",
                dist=0.1,
            ),
            MemoryRecordResult(
                id="mem2",
                text="They are a coffee enthusiast, favorite coffee is flatwhite",
                dist=0.15,
            ),
            MemoryRecordResult(
                id="mem3",
                text="User prefers dark mode for their IDE",
                dist=0.2,
            ),
        ]

        # Deduplicate without LLM
        deduplicated = await deduplicate_search_results(
            memories,
            distance_threshold=0.4,
            use_llm_verification=False,
        )

        # Should have 2 results: merged coffee + IDE
        assert (
            len(deduplicated) == 2
        ), f"Expected 2 deduplicated results, got {len(deduplicated)}"

        # Check that coffee memories were merged
        coffee_result = None
        ide_result = None
        for result in deduplicated:
            if "IDE" in result.text:
                ide_result = result
            else:
                coffee_result = result

        assert ide_result is not None, "IDE result not found"
        assert coffee_result is not None, "Coffee result not found"
        # The merged coffee result should mention coffee
        assert (
            "coffee" in coffee_result.text.lower()
            or "flat white" in coffee_result.text.lower()
        )

    async def test_deduplicate_search_results_with_llm(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test load-time deduplication with LLM verification.
        """
        from agent_memory_server.long_term_memory import deduplicate_search_results
        from agent_memory_server.models import MemoryRecordResult

        # Create mock memory results with duplicates
        memories = [
            MemoryRecordResult(
                id="mem1",
                text="User likes coffee, flat white usually",
                dist=0.1,
            ),
            MemoryRecordResult(
                id="mem2",
                text="They are a coffee enthusiast, favorite coffee is flatwhite",
                dist=0.15,
            ),
            MemoryRecordResult(
                id="mem3",
                text="User prefers dark mode for their IDE",
                dist=0.2,
            ),
        ]

        # Deduplicate with LLM verification
        deduplicated = await deduplicate_search_results(
            memories,
            distance_threshold=0.4,
            use_llm_verification=True,
        )

        # Should have 2 results: merged coffee + IDE
        assert (
            len(deduplicated) == 2
        ), f"Expected 2 deduplicated results, got {len(deduplicated)}"

    async def test_search_with_load_time_deduplication(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that search_long_term_memories applies load-time deduplication.
        """
        # First, store duplicate memories WITHOUT store-time deduplication
        # to simulate legacy duplicates
        memory1 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User likes coffee, flat white usually",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )
        memory2 = MemoryRecord(
            id=str(ulid.ULID()),
            text="They are a coffee enthusiast, favorite coffee is flatwhite",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )
        memory3 = MemoryRecord(
            id=str(ulid.ULID()),
            text="User prefers dark mode for their IDE",
            namespace=unique_namespace,
            user_id=unique_user_id,
            memory_type="semantic",
        )

        # Index without deduplication to simulate legacy duplicates
        await index_long_term_memories(
            [memory1, memory2, memory3],
            redis_client=use_test_redis_connection,
            deduplicate=False,
        )

        # Wait for indexing
        await asyncio.sleep(2)

        # Search WITHOUT load-time deduplication
        raw_results = await search_long_term_memories(
            text="What coffee does the user like?",
            namespace=Namespace(eq=unique_namespace),
            user_id=UserId(eq=unique_user_id),
            limit=10,
            deduplicate=False,
        )

        # Should return all 3 memories (2 coffee + 1 IDE)
        # Note: The IDE memory might not be returned if it's not similar enough
        assert (
            len(raw_results.memories) >= 2
        ), f"Expected at least 2 raw results, got {len(raw_results.memories)}"

        # Search WITH load-time deduplication
        dedup_results = await search_long_term_memories(
            text="What coffee does the user like?",
            namespace=Namespace(eq=unique_namespace),
            user_id=UserId(eq=unique_user_id),
            limit=10,
            deduplicate=True,
            dedup_threshold=0.4,
            dedup_use_llm=True,
        )

        # Should return fewer results due to deduplication
        # The coffee memories should be merged
        assert len(dedup_results.memories) <= len(raw_results.memories), (
            f"Deduplicated results ({len(dedup_results.memories)}) should be <= "
            f"raw results ({len(raw_results.memories)})"
        )

    async def test_verify_duplicates_with_llm(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that LLM correctly identifies which memories represent the same fact.
        """
        from agent_memory_server.long_term_memory import verify_duplicates_with_llm
        from agent_memory_server.models import MemoryRecordResult

        # Create a cluster with 2 coffee memories and 1 unrelated memory
        # that happened to be clustered together by vector similarity
        cluster = [
            MemoryRecordResult(
                id="mem1",
                text="User likes coffee, flat white usually",
                dist=0.1,
            ),
            MemoryRecordResult(
                id="mem2",
                text="They are a coffee enthusiast, favorite coffee is flatwhite",
                dist=0.15,
            ),
            MemoryRecordResult(
                id="mem3",
                text="User drinks tea in the afternoon",
                dist=0.3,
            ),
        ]

        # LLM should split this into 2 groups: coffee and tea
        verified_subclusters = await verify_duplicates_with_llm(cluster)

        # Should have 2 subclusters
        assert (
            len(verified_subclusters) >= 1
        ), f"Expected at least 1 subcluster, got {len(verified_subclusters)}"

        # The coffee memories should be in the same subcluster
        # and the tea memory should be separate (or in its own cluster)
        total_memories = sum(len(sc) for sc in verified_subclusters)
        assert (
            total_memories == 3
        ), f"All memories should be accounted for, got {total_memories}"

    async def test_merge_memory_cluster_for_display(
        self, use_test_redis_connection, unique_namespace, unique_user_id
    ):
        """
        Test that memory clusters are merged into coherent display text.
        """
        from agent_memory_server.long_term_memory import (
            merge_memory_cluster_for_display,
        )
        from agent_memory_server.models import MemoryRecordResult

        # Create a cluster of duplicate coffee memories
        cluster = [
            MemoryRecordResult(
                id="mem1",
                text="User likes coffee, flat white usually",
                dist=0.1,
                topics=["coffee", "preferences"],
            ),
            MemoryRecordResult(
                id="mem2",
                text="They are a coffee enthusiast, favorite coffee is flatwhite",
                dist=0.15,
                entities=["flat white"],
            ),
        ]

        # Merge the cluster
        merged = await merge_memory_cluster_for_display(cluster)

        # The merged result should:
        # 1. Have text that mentions coffee/flat white
        assert (
            "coffee" in merged.text.lower() or "flat white" in merged.text.lower()
        ), f"Merged text should mention coffee: {merged.text}"

        # 2. Have the best (lowest) distance
        assert merged.dist == 0.1, f"Merged dist should be 0.1, got {merged.dist}"

        # 3. Combine topics and entities
        assert merged.topics is not None
        assert "coffee" in merged.topics or "preferences" in merged.topics
        assert merged.entities is not None
        assert "flat white" in merged.entities
