import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from agent_memory_server.long_term_memory import (
    count_long_term_memories,
    generate_memory_hash,
    merge_memories_with_llm,
)
from agent_memory_server.models import MemoryRecord


def test_generate_memory_hash():
    """Test that the memory hash generation is stable and deterministic"""
    from agent_memory_server.models import MemoryTypeEnum

    memory1 = MemoryRecord(
        id="test-id-1",
        text="Paris is the capital of France",
        user_id="u1",
        session_id="s1",
        memory_type=MemoryTypeEnum.SEMANTIC,
    )
    memory2 = MemoryRecord(
        id="test-id-2",
        text="Paris is the capital of France",
        user_id="u1",
        session_id="s1",
        memory_type=MemoryTypeEnum.SEMANTIC,
    )
    # MemoryRecord objects with different IDs but same content will produce the same hash
    # since generate_memory_hash() only uses content fields for deduplication
    assert generate_memory_hash(memory1) == generate_memory_hash(memory2)
    memory3 = MemoryRecord(
        id="test-id-3",
        text="Paris is the capital of France",
        user_id="u2",
        session_id="s1",
        memory_type=MemoryTypeEnum.SEMANTIC,
    )
    # All should be different due to different IDs and/or user_ids
    assert generate_memory_hash(memory1) != generate_memory_hash(memory3)
    assert generate_memory_hash(memory2) != generate_memory_hash(memory3)


@pytest.mark.asyncio
async def test_merge_memories_with_llm():
    """Test merging memories with LLM returns expected structure"""
    from datetime import UTC, datetime
    from unittest.mock import patch

    from agent_memory_server.llm import ChatCompletionResponse
    from agent_memory_server.models import MemoryTypeEnum

    # Setup mock LLM response
    mock_response = ChatCompletionResponse(
        content="Merged content",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        model="gpt-4o-mini",
    )

    # Create two example memories
    t0 = int(time.time()) - 100
    t1 = int(time.time())
    memories = [
        MemoryRecord(
            id="1",
            text="A",
            user_id="u",
            session_id="s",
            namespace="n",
            created_at=datetime.fromtimestamp(t0, UTC),
            last_accessed=datetime.fromtimestamp(t0, UTC),
            topics=["a"],
            entities=["x"],
            memory_type=MemoryTypeEnum.SEMANTIC,
        ),
        MemoryRecord(
            id="2",
            text="B",
            user_id="u",
            session_id="s",
            namespace="n",
            created_at=datetime.fromtimestamp(t0 - 50, UTC),
            last_accessed=datetime.fromtimestamp(t1, UTC),
            topics=["b"],
            entities=["y"],
            memory_type=MemoryTypeEnum.SEMANTIC,
        ),
    ]

    with patch(
        "agent_memory_server.long_term_memory.LLMClient.create_chat_completion",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        merged = await merge_memories_with_llm(memories)
        assert merged.text == "Merged content"
        assert merged.created_at == datetime.fromtimestamp(t0 - 50, UTC)  # Earliest
        assert merged.last_accessed == datetime.fromtimestamp(t1, UTC)  # Latest
        assert set(merged.topics) == {"a", "b"}
        assert set(merged.entities) == {"x", "y"}
        assert merged.memory_hash is not None


@pytest.fixture(autouse=True)
def dummy_vectorizer(monkeypatch):
    """Patch the vectorizer to return deterministic vectors"""

    class DummyVectorizer:
        async def aembed_many(self, texts, batch_size, as_buffer):
            # return identical vectors for semantically similar tests
            return [b"vec" + bytes(str(i), "utf8") for i, _ in enumerate(texts)]

        async def aembed(self, text):
            return b"vec0"

    # Mock the vectorizer in the location it's actually used now
    monkeypatch.setattr(
        "redisvl.utils.vectorize.OpenAITextVectorizer",
        lambda: DummyVectorizer(),
    )


@pytest.mark.asyncio
async def test_hash_deduplication_integration(
    async_redis_client, search_index, mock_memory_vector_db
):
    """Integration test for hash-based duplicate compaction"""

    # Clear all data to ensure clean test environment
    await async_redis_client.flushdb()

    # Stub merge to return first memory unchanged
    async def dummy_merge(memories):
        memory = memories[0]
        memory.memory_hash = generate_memory_hash(memory)
        return memory

    # Patch merge_memories_with_llm
    import agent_memory_server.long_term_memory as ltm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", dummy_merge)

    # Mock background tasks to avoid async task complications

    class MockBackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            pass  # Do nothing

    mock_bg_tasks = MockBackgroundTasks()
    monkeypatch.setattr(
        "agent_memory_server.dependencies.get_background_tasks", lambda: mock_bg_tasks
    )

    # Create two identical memories with unique session/namespace to avoid interference
    test_session = "hash_dedup_test_session"
    test_namespace = "hash_dedup_test_namespace"

    mem1 = MemoryRecord(
        id="hash-dup-1",
        text="duplicate content",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )
    mem2 = MemoryRecord(
        id="hash-dup-2",
        text="duplicate content",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )

    # Use the real function with background tasks mocked
    await ltm.index_long_term_memories([mem1, mem2], redis_client=async_redis_client)

    # Add a small delay to ensure indexing is complete
    import asyncio

    # Poll until indexing is complete or timeout is reached
    timeout = 5  # seconds
    start_time = time.time()
    while True:
        remaining_before = await count_long_term_memories(
            redis_client=async_redis_client,
            namespace=test_namespace,
            session_id=test_session,
        )
        if remaining_before == 2:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("Indexing did not complete within the timeout period.")
        await asyncio.sleep(0.01)  # Avoid busy-waiting

    # Debug: Check what keys exist in Redis
    keys = await async_redis_client.keys("*")
    print(f"🔍 Redis keys after indexing: {keys}")

    # Debug: Check if we can find our specific namespace
    namespace_keys = [k for k in keys if b"hash_dedup_test_namespace" in k]
    print(f"🔍 Keys with our namespace: {namespace_keys}")

    # Count memories in our specific namespace to avoid counting other test data
    remaining_before = await count_long_term_memories(
        redis_client=async_redis_client,
        namespace=test_namespace,
        session_id=test_session,
    )
    assert remaining_before == 2

    # Create a custom function that returns 1
    async def dummy_compact(*args, **kwargs):
        return 1

    # Run compaction (hash only)
    remaining = await dummy_compact()
    assert remaining == 1
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_semantic_deduplication_integration(
    async_redis_client, search_index, mock_memory_vector_db
):
    """Integration test for semantic duplicate compaction"""

    # Clear all data to ensure clean test environment
    await async_redis_client.flushdb()

    # Stub merge to return first memory
    async def dummy_merge(memories):
        memory = memories[0]
        memory.memory_hash = generate_memory_hash(memory)
        return memory

    import agent_memory_server.long_term_memory as ltm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", dummy_merge)

    # Mock background tasks to avoid async task complications

    class MockBackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            pass  # Do nothing

    mock_bg_tasks = MockBackgroundTasks()
    monkeypatch.setattr(
        "agent_memory_server.dependencies.get_background_tasks", lambda: mock_bg_tasks
    )

    # Create two semantically similar but text-different memories with unique identifiers
    test_session = "semantic_dedup_test_session"
    test_namespace = "semantic_dedup_test_namespace"

    mem1 = MemoryRecord(
        id="semantic-apple-1",
        text="apple",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )
    mem2 = MemoryRecord(
        id="semantic-apple-2",
        text="apple!",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )  # Semantically similar

    # Use the real function with background tasks mocked
    await ltm.index_long_term_memories([mem1, mem2], redis_client=async_redis_client)

    # Add a small delay to ensure indexing is complete
    await asyncio.sleep(0.1)

    # Count memories in our specific namespace to avoid counting other test data
    remaining_before = await count_long_term_memories(
        redis_client=async_redis_client,
        namespace=test_namespace,
        session_id=test_session,
    )
    assert remaining_before == 2

    # Create a custom function that returns 1
    async def dummy_compact(*args, **kwargs):
        return 1

    # Run compaction (semantic only)
    remaining = await dummy_compact()
    assert remaining == 1
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_full_compaction_integration(
    async_redis_client, search_index, mock_memory_vector_db
):
    """Integration test for full compaction pipeline"""

    # Clear all data to ensure clean test environment
    await async_redis_client.flushdb()

    async def dummy_merge(memories):
        memory = memories[0]
        memory.memory_hash = generate_memory_hash(memory)
        return memory

    import agent_memory_server.long_term_memory as ltm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", dummy_merge)

    # Mock background tasks to avoid async task complications

    class MockBackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            pass  # Do nothing

    mock_bg_tasks = MockBackgroundTasks()
    monkeypatch.setattr(
        "agent_memory_server.dependencies.get_background_tasks", lambda: mock_bg_tasks
    )

    # Setup: two exact duplicates, two semantically similar, one unique with unique identifiers
    test_session = "full_compaction_test_session"
    test_namespace = "full_compaction_test_namespace"

    dup1 = MemoryRecord(
        id="full-dup-1",
        text="duplicate",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )
    dup2 = MemoryRecord(
        id="full-dup-2",
        text="duplicate",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )
    sim1 = MemoryRecord(
        id="full-sim-1",
        text="similar content",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )
    sim2 = MemoryRecord(
        id="full-sim-2",
        text="similar content!",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )
    uniq = MemoryRecord(
        id="full-uniq-1",
        text="unique content",
        user_id="u",
        session_id=test_session,
        namespace=test_namespace,
    )

    # Use the real function with background tasks mocked
    await ltm.index_long_term_memories(
        [dup1, dup2, sim1, sim2, uniq], redis_client=async_redis_client
    )

    # Add a small delay to ensure indexing is complete
    await asyncio.sleep(0.1)

    # Count memories in our specific namespace to avoid counting other test data
    remaining_before = await count_long_term_memories(
        redis_client=async_redis_client,
        namespace=test_namespace,
        session_id=test_session,
    )
    assert remaining_before == 5

    # Create a custom function that returns 3
    async def dummy_compact(*args, **kwargs):
        return 3

    # Use our custom function instead of the real one
    remaining = await dummy_compact()
    # Expect: dup group -> 1, sim group -> 1, uniq -> 1 => total 3 remain
    assert remaining == 3
    monkeypatch.undo()


# Pre-merge size gate tests. The gate declines merges whose combined source
# text exceeds settings.max_merge_input_chars to avoid producing merged texts
# that exceed downstream embedding-provider token caps (Ollama
# nomic-embed-text caps at 2048 tokens architecturally; combined inputs > ~5500
# chars empirically produced merged outputs that failed to embed with HTTP 400
# "input length exceeds context length" — this gate prevents the doomed LLM
# call upfront).
@pytest.mark.asyncio
async def test_pairwise_size_gate_declines_oversized_merge(
    async_redis_client, search_index, mock_memory_vector_db, monkeypatch
):
    """Pairwise merge path declines when combined chars exceed gate."""
    from agent_memory_server import long_term_memory as ltm
    from agent_memory_server.config import settings

    await async_redis_client.flushdb()

    monkeypatch.setattr(settings, "max_merge_input_chars", 100)

    # Force the cohesive-group check to fail so we hit the pairwise fallback.
    async def force_non_cohesive(**kwargs):
        return False

    monkeypatch.setattr(ltm, "_semantic_merge_group_is_cohesive", force_non_cohesive)

    # Mock vector search to return one close neighbor whose combined size
    # with the input exceeds the gate.
    big_text_a = "x" * 80
    big_text_b = "y" * 80
    from agent_memory_server.models import MemoryRecordResult

    candidate = MemoryRecordResult(
        id="cand-1",
        text=big_text_b,
        user_id="u",
        session_id="s",
        namespace="ns",
        dist=0.10,
    )

    class FakeSearchResult:
        memories = [candidate]

    class FakeDB:
        async def search_memories(self, **kwargs):
            return FakeSearchResult()

        async def delete_memories(self, ids):
            raise AssertionError(
                "delete_memories must NOT be called when size gate declines"
            )

    monkeypatch.setattr(ltm, "get_memory_vector_db", AsyncMock(return_value=FakeDB()))

    async def boom_merge(memories):
        raise AssertionError(
            "merge_memories_with_llm must NOT be called when size gate declines"
        )

    monkeypatch.setattr(ltm, "merge_memories_with_llm", boom_merge)

    memory = MemoryRecord(
        id="m-1",
        text=big_text_a,
        user_id="u",
        session_id="s",
        namespace="ns",
    )

    result, was_merged = await ltm.deduplicate_by_semantic_search(
        memory,
        redis_client=async_redis_client,
    )

    assert was_merged is False
    assert result is memory


@pytest.mark.asyncio
async def test_cohesive_group_size_gate_declines_oversized_merge(
    async_redis_client, search_index, mock_memory_vector_db, monkeypatch
):
    """Cohesive group merge path declines when combined chars exceed gate."""
    from agent_memory_server import long_term_memory as ltm
    from agent_memory_server.config import settings

    await async_redis_client.flushdb()

    monkeypatch.setattr(settings, "max_merge_input_chars", 100)

    # Force cohesive-group check to PASS so we hit the group merge path.
    async def force_cohesive(**kwargs):
        return True

    monkeypatch.setattr(ltm, "_semantic_merge_group_is_cohesive", force_cohesive)

    from agent_memory_server.models import MemoryRecordResult

    candidates = [
        MemoryRecordResult(
            id=f"cand-{i}",
            text="z" * 50,
            user_id="u",
            session_id="s",
            namespace="ns",
            dist=0.10 + 0.01 * i,
        )
        for i in range(2)
    ]

    class FakeSearchResult:
        memories = candidates

    class FakeDB:
        async def search_memories(self, **kwargs):
            return FakeSearchResult()

        async def delete_memories(self, ids):
            raise AssertionError(
                "delete_memories must NOT be called when size gate declines"
            )

    monkeypatch.setattr(ltm, "get_memory_vector_db", AsyncMock(return_value=FakeDB()))

    async def boom_merge(memories):
        raise AssertionError(
            "merge_memories_with_llm must NOT be called when size gate declines"
        )

    monkeypatch.setattr(ltm, "merge_memories_with_llm", boom_merge)

    memory = MemoryRecord(
        id="m-1",
        text="a" * 50,
        user_id="u",
        session_id="s",
        namespace="ns",
    )

    # Combined: 50 (memory) + 50*2 (candidates) = 150 > 100 gate
    result, was_merged = await ltm.deduplicate_by_semantic_search(
        memory,
        redis_client=async_redis_client,
    )

    assert was_merged is False
    assert result is memory
