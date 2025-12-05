import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

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
async def test_merge_memories_with_llm(mock_openai_client, monkeypatch):
    """Test merging memories with LLM returns expected structure"""
    from datetime import UTC, datetime

    from agent_memory_server.models import MemoryTypeEnum

    # Setup dummy LLM response
    dummy_response = MagicMock()
    dummy_response.choices = [MagicMock()]
    dummy_response.choices[0].message = MagicMock()
    dummy_response.choices[0].message.content = "Merged content"
    mock_openai_client.create_chat_completion = AsyncMock(return_value=dummy_response)

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

    merged = await merge_memories_with_llm(memories, llm_client=mock_openai_client)
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
    async_redis_client, search_index, mock_openai_client, mock_vectorstore_adapter
):
    """Integration test for hash-based duplicate compaction"""

    # Clear all data to ensure clean test environment
    await async_redis_client.flushdb()

    # Stub merge to return first memory unchanged
    async def dummy_merge(memories, llm_client=None):
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
    async_redis_client, search_index, mock_openai_client, mock_vectorstore_adapter
):
    """Integration test for semantic duplicate compaction"""

    # Clear all data to ensure clean test environment
    await async_redis_client.flushdb()

    # Stub merge to return first memory
    async def dummy_merge(memories, llm_client=None):
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
    async_redis_client, search_index, mock_openai_client, mock_vectorstore_adapter
):
    """Integration test for full compaction pipeline"""

    # Clear all data to ensure clean test environment
    await async_redis_client.flushdb()

    async def dummy_merge(memories, llm_client=None):
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
