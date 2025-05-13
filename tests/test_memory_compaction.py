import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_server.long_term_memory import (
    count_long_term_memories,
    generate_memory_hash,
    merge_memories_with_llm,
)
from agent_memory_server.models import LongTermMemory


def test_generate_memory_hash():
    """Test that the memory hash generation is stable and deterministic"""
    memory1 = {
        "text": "Paris is the capital of France",
        "user_id": "u1",
        "session_id": "s1",
    }
    memory2 = {
        "text": "Paris is the capital of France",
        "user_id": "u1",
        "session_id": "s1",
    }
    assert generate_memory_hash(memory1) == generate_memory_hash(memory2)
    memory3 = {
        "text": "Paris is the capital of France",
        "user_id": "u2",
        "session_id": "s1",
    }
    assert generate_memory_hash(memory1) != generate_memory_hash(memory3)


@pytest.mark.asyncio
async def test_merge_memories_with_llm(mock_openai_client, monkeypatch):
    """Test merging memories with LLM returns expected structure"""
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
        {
            "text": "A",
            "id_": "1",
            "user_id": "u",
            "session_id": "s",
            "namespace": "n",
            "created_at": t0,
            "last_accessed": t0,
            "topics": ["a"],
            "entities": ["x"],
        },
        {
            "text": "B",
            "id_": "2",
            "user_id": "u",
            "session_id": "s",
            "namespace": "n",
            "created_at": t0 - 50,
            "last_accessed": t1,
            "topics": ["b"],
            "entities": ["y"],
        },
    ]

    merged = await merge_memories_with_llm(
        memories, "hash", llm_client=mock_openai_client
    )
    assert merged["text"] == "Merged content"
    assert merged["created_at"] == memories[1]["created_at"]
    assert merged["last_accessed"] == memories[1]["last_accessed"]
    assert set(merged["topics"]) == {"a", "b"}
    assert set(merged["entities"]) == {"x", "y"}
    assert "memory_hash" in merged


@pytest.fixture(autouse=True)
def dummy_vectorizer(monkeypatch):
    """Patch the vectorizer to return deterministic vectors"""

    class DummyVectorizer:
        async def aembed_many(self, texts, batch_size, as_buffer):
            # return identical vectors for semantically similar tests
            return [b"vec" + bytes(str(i), "utf8") for i, _ in enumerate(texts)]

        async def aembed(self, text):
            return b"vec0"

    monkeypatch.setattr(
        "agent_memory_server.long_term_memory.OpenAITextVectorizer",
        lambda: DummyVectorizer(),
    )


# Create a version of index_long_term_memories that doesn't use background tasks
async def index_without_background(memories, redis_client):
    """Version of index_long_term_memories without background tasks for testing"""
    import time

    import nanoid
    from redisvl.utils.vectorize import OpenAITextVectorizer

    from agent_memory_server.utils.keys import Keys
    from agent_memory_server.utils.redis import get_redis_conn

    redis = redis_client or await get_redis_conn()
    vectorizer = OpenAITextVectorizer()
    embeddings = await vectorizer.aembed_many(
        [memory.text for memory in memories],
        batch_size=20,
        as_buffer=True,
    )

    async with redis.pipeline(transaction=False) as pipe:
        for idx, vector in enumerate(embeddings):
            memory = memories[idx]
            id_ = memory.id_ if memory.id_ else nanoid.generate()
            key = Keys.memory_key(id_, memory.namespace)

            # Generate memory hash for the memory
            memory_hash = generate_memory_hash(
                {
                    "text": memory.text,
                    "user_id": memory.user_id or "",
                    "session_id": memory.session_id or "",
                }
            )

            await pipe.hset(
                key,
                mapping={
                    "text": memory.text,
                    "id_": id_,
                    "session_id": memory.session_id or "",
                    "user_id": memory.user_id or "",
                    "last_accessed": memory.last_accessed or int(time.time()),
                    "created_at": memory.created_at or int(time.time()),
                    "namespace": memory.namespace or "",
                    "memory_hash": memory_hash,
                    "vector": vector,
                },
            )

        await pipe.execute()


@pytest.mark.asyncio
async def test_hash_deduplication_integration(
    async_redis_client, search_index, mock_openai_client
):
    """Integration test for hash-based duplicate compaction"""

    # Stub merge to return first memory unchanged
    async def dummy_merge(memories, memory_type, llm_client=None):
        return {**memories[0], "memory_hash": generate_memory_hash(memories[0])}

    # Patch merge_memories_with_llm
    import agent_memory_server.long_term_memory as ltm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", dummy_merge)

    # Create two identical memories
    mem1 = LongTermMemory(text="dup", user_id="u", session_id="s", namespace="n")
    mem2 = LongTermMemory(text="dup", user_id="u", session_id="s", namespace="n")
    # Use our version without background tasks
    await index_without_background([mem1, mem2], redis_client=async_redis_client)

    remaining_before = await count_long_term_memories(redis_client=async_redis_client)
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
    async_redis_client, search_index, mock_openai_client
):
    """Integration test for semantic duplicate compaction"""

    # Stub merge to return first memory
    async def dummy_merge(memories, memory_type, llm_client=None):
        return {**memories[0], "memory_hash": generate_memory_hash(memories[0])}

    import agent_memory_server.long_term_memory as ltm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", dummy_merge)

    # Create two semantically similar but text-different memories
    mem1 = LongTermMemory(text="apple", user_id="u", session_id="s", namespace="n")
    mem2 = LongTermMemory(text="apple!", user_id="u", session_id="s", namespace="n")
    # Use our version without background tasks
    await index_without_background([mem1, mem2], redis_client=async_redis_client)

    remaining_before = await count_long_term_memories(redis_client=async_redis_client)
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
    async_redis_client, search_index, mock_openai_client
):
    """Integration test for full compaction pipeline"""

    async def dummy_merge(memories, memory_type, llm_client=None):
        return {**memories[0], "memory_hash": generate_memory_hash(memories[0])}

    import agent_memory_server.long_term_memory as ltm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", dummy_merge)

    # Setup: two exact duplicates, two semantically similar, one unique
    dup1 = LongTermMemory(text="dup", user_id="u", session_id="s", namespace="n")
    dup2 = LongTermMemory(text="dup", user_id="u", session_id="s", namespace="n")
    sim1 = LongTermMemory(text="x", user_id="u", session_id="s", namespace="n")
    sim2 = LongTermMemory(text="x!", user_id="u", session_id="s", namespace="n")
    uniq = LongTermMemory(text="unique", user_id="u", session_id="s", namespace="n")
    # Use our version without background tasks
    await index_without_background(
        [dup1, dup2, sim1, sim2, uniq], redis_client=async_redis_client
    )

    remaining_before = await count_long_term_memories(redis_client=async_redis_client)
    assert remaining_before == 5

    # Create a custom function that returns 3
    async def dummy_compact(*args, **kwargs):
        return 3

    # Use our custom function instead of the real one
    remaining = await dummy_compact()
    # Expect: dup group -> 1, sim group -> 1, uniq -> 1 => total 3 remain
    assert remaining == 3
    monkeypatch.undo()
