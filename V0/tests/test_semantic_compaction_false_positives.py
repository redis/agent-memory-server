import math
from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest
import ulid
from redisvl.index import AsyncSearchIndex

import agent_memory_server.long_term_memory as ltm
import agent_memory_server.memory_vector_db_factory as memory_db_factory
from agent_memory_server.config import settings
from agent_memory_server.filters import Namespace, UserId
from agent_memory_server.long_term_memory import (
    compact_long_term_memories,
    count_long_term_memories,
    deduplicate_by_semantic_search,
    index_long_term_memories,
)
from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase
from agent_memory_server.models import MemoryRecord


class ControlledEmbeddings:
    """Deterministic embeddings for threshold-focused compaction tests."""

    def __init__(self, vectors_by_text: dict[str, Sequence[float]]):
        self._vectors_by_text = {
            text: [float(value) for value in vector]
            for text, vector in vectors_by_text.items()
        }
        self._dimensions = len(next(iter(self._vectors_by_text.values())))
        self.model = "controlled-test-embeddings"

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        try:
            return self._vectors_by_text[text]
        except KeyError as exc:
            raise AssertionError(
                f"No controlled embedding configured for: {text}"
            ) from exc


class DummyBackgroundTasks:
    def add_task(self, func, *args, **kwargs):  # noqa: ANN001
        return None


def install_controlled_memory_db(
    monkeypatch, vectors_by_text: dict[str, Sequence[float]]
):
    """Patch the long-term memory path to use a real RedisVL DB with fake vectors."""

    schema = {
        "index": {
            "name": settings.redisvl_index_name,
            "prefix": settings.redisvl_index_prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "session_id", "type": "tag"},
            {"name": "user_id", "type": "tag"},
            {"name": "namespace", "type": "tag"},
            {"name": "memory_type", "type": "tag"},
            {"name": "topics", "type": "tag"},
            {"name": "entities", "type": "tag"},
            {"name": "memory_hash", "type": "tag"},
            {"name": "discrete_memory_extracted", "type": "tag"},
            {"name": "pinned", "type": "tag"},
            {"name": "extracted_from", "type": "tag"},
            {"name": "id_", "type": "tag"},
            {"name": "access_count", "type": "numeric"},
            {"name": "created_at", "type": "numeric"},
            {"name": "last_accessed", "type": "numeric"},
            {"name": "updated_at", "type": "numeric"},
            {"name": "persisted_at", "type": "numeric"},
            {"name": "event_date", "type": "numeric"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": len(next(iter(vectors_by_text.values()))),
                    "distance_metric": settings.redisvl_distance_metric.lower(),
                    "algorithm": settings.redisvl_indexing_algorithm.lower(),
                    "datatype": "float32",
                },
            },
        ],
    }

    index = AsyncSearchIndex.from_dict(schema, redis_url=settings.redis_url)
    db = RedisVLMemoryVectorDatabase(index, ControlledEmbeddings(vectors_by_text))

    async def get_db():
        return db

    monkeypatch.setattr(ltm, "get_memory_vector_db", get_db)
    monkeypatch.setattr(memory_db_factory, "get_memory_vector_db", get_db)
    monkeypatch.setattr(ltm, "get_background_tasks", lambda: DummyBackgroundTasks())

    return db


@pytest.mark.asyncio
async def test_unrelated_diet_and_basketball_memories_do_not_semantically_compact(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    diet_text = "User eats a healthy diet with vegetables and lean protein."
    sports_text = "User played basketball in high school."

    db = install_controlled_memory_db(
        monkeypatch,
        {
            diet_text: [1.0, 0.0, 0.0],
            sports_text: [0.0, 1.0, 0.0],
        },
    )

    diet_memory = MemoryRecord(
        id="diet-memory",
        text=diet_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )
    sports_memory = MemoryRecord(
        id="sports-memory",
        text=sports_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )

    await index_long_term_memories(
        [diet_memory],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    search_result = await db.search_memories(
        query=sports_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=10,
    )

    assert search_result.total == 0

    returned_memory, was_merged = await deduplicate_by_semantic_search(
        memory=sports_memory,
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert not was_merged
    assert returned_memory is not None
    assert returned_memory.id == sports_memory.id


@pytest.mark.asyncio
async def test_bridge_memory_is_rejected_when_candidate_group_is_not_cohesive(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    diet_text = "User eats a healthy diet with vegetables and lean protein."
    sports_text = "User played basketball in high school."
    bridge_text = "User stayed healthy in high school by playing basketball and eating a healthy diet."

    bridge_component = 1 / math.sqrt(2)
    db = install_controlled_memory_db(
        monkeypatch,
        {
            diet_text: [1.0, 0.0, 0.0],
            sports_text: [0.0, 1.0, 0.0],
            bridge_text: [bridge_component, bridge_component, 0.0],
        },
    )

    diet_memory = MemoryRecord(
        id="diet-memory",
        text=diet_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )
    sports_memory = MemoryRecord(
        id="sports-memory",
        text=sports_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )
    bridge_memory = MemoryRecord(
        id="bridge-memory",
        text=bridge_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )

    await index_long_term_memories(
        [diet_memory, sports_memory],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    search_result = await db.search_memories(
        query=bridge_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=10,
    )

    assert {memory.id for memory in search_result.memories} == {
        diet_memory.id,
        sports_memory.id,
    }
    assert all(memory.dist < 0.35 for memory in search_result.memories)

    merge_mock = AsyncMock()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    returned_memory, was_merged = await deduplicate_by_semantic_search(
        memory=bridge_memory,
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert not was_merged
    assert returned_memory is not None
    assert returned_memory.id == bridge_memory.id
    merge_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_cohesive_paraphrase_cluster_still_merges(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    coffee_1 = "User likes flat white coffee."
    coffee_2 = "User's favorite coffee is a flat white."
    coffee_3 = "User usually orders a flat white coffee."

    db = install_controlled_memory_db(
        monkeypatch,
        {
            coffee_1: [1.0, 0.0, 0.0],
            coffee_2: [0.99, 0.01, 0.0],
            coffee_3: [0.98, 0.02, 0.0],
        },
    )

    await index_long_term_memories(
        [
            MemoryRecord(
                id="coffee-1",
                text=coffee_1,
                namespace=namespace,
                user_id=user_id,
                memory_type="semantic",
            ),
            MemoryRecord(
                id="coffee-2",
                text=coffee_2,
                namespace=namespace,
                user_id=user_id,
                memory_type="semantic",
            ),
        ],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    search_result = await db.search_memories(
        query=coffee_3,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=10,
    )

    assert {memory.id for memory in search_result.memories} == {"coffee-1", "coffee-2"}

    async def fake_merge(memories: list[MemoryRecord]) -> MemoryRecord:
        return memories[0]

    merge_mock = AsyncMock(side_effect=fake_merge)
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    _, was_merged = await deduplicate_by_semantic_search(
        memory=MemoryRecord(
            id="coffee-3",
            text=coffee_3,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert was_merged
    merge_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_range_query_threshold_controls_bridge_candidate_selection(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    diet_text = "User eats a healthy diet with vegetables and lean protein."
    sports_text = "User played basketball in high school."
    bridge_text = "User stayed healthy in high school by playing basketball and eating a healthy diet."

    bridge_component = 1 / math.sqrt(2)
    db = install_controlled_memory_db(
        monkeypatch,
        {
            diet_text: [1.0, 0.0, 0.0],
            sports_text: [0.0, 1.0, 0.0],
            bridge_text: [bridge_component, bridge_component, 0.0],
        },
    )

    await index_long_term_memories(
        [
            MemoryRecord(
                id="diet-memory",
                text=diet_text,
                namespace=namespace,
                user_id=user_id,
                memory_type="semantic",
            ),
            MemoryRecord(
                id="sports-memory",
                text=sports_text,
                namespace=namespace,
                user_id=user_id,
                memory_type="semantic",
            ),
        ],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    strict_result = await db.search_memories(
        query=bridge_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.29,
        limit=10,
    )
    relaxed_result = await db.search_memories(
        query=bridge_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=10,
    )

    assert strict_result.total == 0
    assert relaxed_result.total == 2
    assert all(
        memory.dist == pytest.approx(0.292893, abs=1e-5)
        for memory in relaxed_result.memories
    )


@pytest.mark.asyncio
async def test_compaction_does_not_snowball_through_a_bridge_memory(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    diet_text = "User eats a healthy diet with vegetables and lean protein."
    sports_text = "User played basketball in high school."
    bridge_text = "User stayed healthy in high school by playing basketball and eating a healthy diet."

    bridge_component = 1 / math.sqrt(2)
    install_controlled_memory_db(
        monkeypatch,
        {
            diet_text: [1.0, 0.0, 0.0],
            sports_text: [0.0, 1.0, 0.0],
            bridge_text: [bridge_component, bridge_component, 0.0],
        },
    )

    memories = [
        MemoryRecord(
            id="diet-memory",
            text=diet_text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        MemoryRecord(
            id="sports-memory",
            text=sports_text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        MemoryRecord(
            id="bridge-memory",
            text=bridge_text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
    ]

    await index_long_term_memories(
        memories,
        redis_client=async_redis_client,
        deduplicate=False,
    )

    merge_counter = 0

    async def fake_merge(memories_to_merge: list[MemoryRecord]) -> MemoryRecord:
        nonlocal merge_counter
        merge_counter += 1
        return MemoryRecord(
            id=f"merged-{merge_counter}",
            text=bridge_text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
            discrete_memory_extracted="t",
        )

    monkeypatch.setattr(
        ltm, "merge_memories_with_llm", AsyncMock(side_effect=fake_merge)
    )

    remaining_after_compaction = await compact_long_term_memories(
        limit=100,
        namespace=namespace,
        user_id=user_id,
        redis_client=async_redis_client,
        vector_distance_threshold=0.35,
        compact_hash_duplicates=False,
        compact_semantic_duplicates=True,
    )

    assert merge_counter == 0
    assert remaining_after_compaction == 3
    assert (
        await count_long_term_memories(
            namespace=namespace,
            user_id=user_id,
            redis_client=async_redis_client,
        )
        == 3
    )


@pytest.mark.asyncio
async def test_issue_200_chain_does_not_create_a_mega_memory(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    sports_history = "User played basketball in college and tennis in high school."
    tempeh = "User is vegetarian and recently grilled tempeh for the first time."
    rowing = "User warms up before workouts on the rowing machine for 5-10 minutes."
    dog_walk = "User walks golden retriever Max for about 45 minutes in the evenings."
    sleep = "User got about five hours of sleep on a recent day due to work stress."

    def polar(angle_degrees: float) -> list[float]:
        radians = math.radians(angle_degrees)
        return [math.cos(radians), math.sin(radians)]

    install_controlled_memory_db(
        monkeypatch,
        {
            sports_history: polar(0),
            tempeh: polar(35),
            rowing: polar(70),
            dog_walk: polar(105),
            sleep: polar(140),
        },
    )

    memories = [
        MemoryRecord(
            id="sports-history",
            text=sports_history,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        MemoryRecord(
            id="tempeh",
            text=tempeh,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        MemoryRecord(
            id="rowing",
            text=rowing,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        MemoryRecord(
            id="dog-walk",
            text=dog_walk,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        MemoryRecord(
            id="sleep",
            text=sleep,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
    ]

    await index_long_term_memories(
        memories,
        redis_client=async_redis_client,
        deduplicate=False,
    )

    merge_mock = AsyncMock()
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    remaining_after_compaction = await compact_long_term_memories(
        limit=100,
        namespace=namespace,
        user_id=user_id,
        redis_client=async_redis_client,
        vector_distance_threshold=0.35,
        compact_hash_duplicates=False,
        compact_semantic_duplicates=True,
    )

    merge_mock.assert_not_awaited()
    assert remaining_after_compaction == 5
    assert (
        await count_long_term_memories(
            namespace=namespace,
            user_id=user_id,
            redis_client=async_redis_client,
        )
        == 5
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_text", "candidate_text"),
    [
        (
            "User played basketball in college and tennis in high school.",
            "User used to play tennis in high school and basketball in college.",
        ),
        (
            "User aims for 120 grams of protein daily as a vegetarian.",
            "User aims to consume around 120 grams of protein daily and tracks macros for balance.",
        ),
        (
            "User recently hit a new squat personal record of 225 lbs for three reps as of March 13, 2026.",
            "User recently hit a new squat personal record of 225 lbs for three reps.",
        ),
    ],
)
async def test_issue_200_valid_same_topic_pairs_still_merge(
    async_redis_client,
    monkeypatch,
    existing_text,
    candidate_text,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    db = install_controlled_memory_db(
        monkeypatch,
        {
            existing_text: [1.0, 0.0],
            candidate_text: [0.99, 0.01],
        },
    )

    existing_memory = MemoryRecord(
        id="existing-memory",
        text=existing_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )
    candidate_memory = MemoryRecord(
        id="candidate-memory",
        text=candidate_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )

    await index_long_term_memories(
        [existing_memory],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    search_result = await db.search_memories(
        query=candidate_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=10,
    )

    assert {memory.id for memory in search_result.memories} == {existing_memory.id}

    async def fake_merge(memories: list[MemoryRecord]) -> MemoryRecord:
        return memories[0]

    merge_mock = AsyncMock(side_effect=fake_merge)
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    _, was_merged = await deduplicate_by_semantic_search(
        memory=candidate_memory,
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert was_merged
    merge_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_capped_semantic_merge_group_still_passes_cohesion_check(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    existing_texts = [
        f"User prefers flat white coffee with oat milk variant {index}."
        for index in range(10)
    ]
    candidate_text = "User prefers flat white coffee with oat milk."

    db = install_controlled_memory_db(
        monkeypatch,
        {
            **{text: [1.0, 0.0] for text in existing_texts},
            candidate_text: [1.0, 0.0],
        },
    )

    await index_long_term_memories(
        [
            MemoryRecord(
                id=f"existing-{index}",
                text=text,
                namespace=namespace,
                user_id=user_id,
                memory_type="semantic",
            )
            for index, text in enumerate(existing_texts)
        ],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    search_result = await db.search_memories(
        query=candidate_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=10,
    )

    assert search_result.total == 10

    async def fake_merge(memories: list[MemoryRecord]) -> MemoryRecord:
        return memories[0]

    merge_mock = AsyncMock(side_effect=fake_merge)
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    _, was_merged = await deduplicate_by_semantic_search(
        memory=MemoryRecord(
            id="candidate-memory",
            text=candidate_text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert was_merged
    merge_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_compaction_preserves_candidate_window_for_indexed_anchor(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    anchor_text = "User prefers flat white coffee with oat milk."
    candidate_texts = [
        f"User prefers flat white coffee with oat milk variant {index}."
        for index in range(9)
    ]
    extra_text = "User prefers a flat white with oat milk and a cinnamon sprinkle."

    vectors_by_text: dict[str, Sequence[float]] = {anchor_text: [1.0] + [0.0] * 10}
    for index, text in enumerate(candidate_texts, start=1):
        vector = [1.0] + [0.0] * 10
        vector[index] = 0.1
        vectors_by_text[text] = vector

    extra_vector = [1.0] + [0.0] * 10
    extra_vector[1] = 0.1
    extra_vector[10] = 0.05
    vectors_by_text[extra_text] = extra_vector

    db = install_controlled_memory_db(monkeypatch, vectors_by_text)

    anchor_memory = MemoryRecord(
        id="anchor-memory",
        text=anchor_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )
    candidate_memories = [
        MemoryRecord(
            id=f"candidate-{index}",
            text=text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        )
        for index, text in enumerate(candidate_texts)
    ]
    extra_memory = MemoryRecord(
        id="extra-memory",
        text=extra_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )

    await index_long_term_memories(
        [anchor_memory, *candidate_memories, extra_memory],
        redis_client=async_redis_client,
        deduplicate=False,
    )

    capped_result = await db.search_memories(
        query=anchor_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=ltm.SEMANTIC_DEDUP_SEARCH_LIMIT,
    )
    headroom_result = await db.search_memories(
        query=anchor_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=ltm.SEMANTIC_DEDUP_QUERY_LIMIT,
    )

    assert extra_memory.id not in {memory.id for memory in capped_result.memories}
    assert extra_memory.id in {memory.id for memory in headroom_result.memories}

    merged_groups: list[list[str]] = []

    async def fake_merge(memories: list[MemoryRecord]) -> MemoryRecord:
        merged_groups.append([memory.id for memory in memories])
        return memories[0]

    merge_mock = AsyncMock(side_effect=fake_merge)
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    _, was_merged = await deduplicate_by_semantic_search(
        memory=anchor_memory,
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert was_merged
    assert merge_mock.await_count == 1
    assert set(merged_groups[0]) == {
        anchor_memory.id,
        extra_memory.id,
        *(memory.id for memory in candidate_memories),
    }


@pytest.mark.asyncio
async def test_capped_dense_cluster_with_extra_neighbor_still_merges(
    async_redis_client,
    monkeypatch,
):
    await async_redis_client.flushdb()

    namespace = f"ns-{ulid.ULID()}"
    user_id = f"user-{ulid.ULID()}"

    existing_texts = [
        f"User prefers flat white coffee with oat milk variant {index}."
        for index in range(11)
    ]
    candidate_text = "User prefers flat white coffee with oat milk."

    db = install_controlled_memory_db(
        monkeypatch,
        {
            **{text: [1.0, 0.0] for text in existing_texts},
            candidate_text: [1.0, 0.0],
        },
    )

    existing_memories = [
        MemoryRecord(
            id=f"existing-{index}",
            text=text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        )
        for index, text in enumerate(existing_texts)
    ]

    await index_long_term_memories(
        existing_memories,
        redis_client=async_redis_client,
        deduplicate=False,
    )

    search_result = await db.search_memories(
        query=candidate_text,
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        distance_threshold=0.35,
        limit=ltm.SEMANTIC_DEDUP_QUERY_LIMIT,
    )

    assert search_result.total == ltm.SEMANTIC_DEDUP_QUERY_LIMIT

    merged_groups: list[list[str]] = []

    async def fake_merge(memories: list[MemoryRecord]) -> MemoryRecord:
        merged_groups.append([memory.id for memory in memories])
        return memories[0]

    merge_mock = AsyncMock(side_effect=fake_merge)
    monkeypatch.setattr(ltm, "merge_memories_with_llm", merge_mock)

    _, was_merged = await deduplicate_by_semantic_search(
        memory=MemoryRecord(
            id="candidate-memory",
            text=candidate_text,
            namespace=namespace,
            user_id=user_id,
            memory_type="semantic",
        ),
        redis_client=async_redis_client,
        namespace=namespace,
        user_id=user_id,
        vector_distance_threshold=0.35,
    )

    assert was_merged
    assert merge_mock.await_count == 1
    assert len(merged_groups[0]) == ltm.SEMANTIC_DEDUP_SEARCH_LIMIT + 1
