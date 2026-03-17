"""Test that task timestamp ordering is validated.

Regression test for https://github.com/redis/agent-memory-server/issues/207
"""

from datetime import UTC, datetime

import pytest
from ulid import ULID

from agent_memory_server.models import Task, TaskStatusEnum, TaskTypeEnum
from agent_memory_server.tasks import create_task, get_task, update_task_status


_ORDERING_ERROR_RE = r"started_at.*must not be after.*completed_at"


def _make_task(**overrides) -> Task:
    defaults = {
        "id": str(ULID()),
        "type": TaskTypeEnum.SUMMARY_VIEW_FULL_RUN,
        "view_id": "test-view",
    }
    defaults.update(overrides)
    return Task(**defaults)


class TestTaskTimestampOrdering:
    """started_at must be <= completed_at when both are present."""

    @pytest.mark.asyncio
    async def test_rejects_started_at_after_completed_at_in_single_call(
        self, async_redis_client
    ):
        """Providing started_at > completed_at in the same update should raise."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)

        late = datetime(2030, 1, 1, tzinfo=UTC)
        early = datetime(2020, 1, 1, tzinfo=UTC)

        with pytest.raises(ValueError, match=_ORDERING_ERROR_RE):
            await update_task_status(task.id, started_at=late, completed_at=early)

    @pytest.mark.asyncio
    async def test_rejects_new_completed_at_before_existing_started_at(
        self, async_redis_client
    ):
        """Setting completed_at earlier than the existing started_at should raise."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)

        started = datetime(2025, 6, 1, tzinfo=UTC)
        await update_task_status(task.id, started_at=started)

        too_early = datetime(2025, 5, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match=_ORDERING_ERROR_RE):
            await update_task_status(task.id, completed_at=too_early)

    @pytest.mark.asyncio
    async def test_rejects_new_started_at_after_existing_completed_at(
        self, async_redis_client
    ):
        """Setting started_at later than the existing completed_at should raise."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)

        completed = datetime(2025, 6, 1, tzinfo=UTC)
        await update_task_status(task.id, completed_at=completed)

        too_late = datetime(2025, 7, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match=_ORDERING_ERROR_RE):
            await update_task_status(task.id, started_at=too_late)

    @pytest.mark.asyncio
    async def test_accepts_valid_ordering(self, async_redis_client):
        """started_at <= completed_at should be accepted without error."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)

        started = datetime(2025, 6, 1, tzinfo=UTC)
        completed = datetime(2025, 6, 2, tzinfo=UTC)
        await update_task_status(task.id, started_at=started, completed_at=completed)

        t = await get_task(task.id)
        assert t.started_at == started
        assert t.completed_at == completed

    @pytest.mark.asyncio
    async def test_accepts_equal_timestamps(self, async_redis_client):
        """started_at == completed_at should be accepted (instant completion)."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)

        now = datetime(2025, 6, 1, tzinfo=UTC)
        await update_task_status(task.id, started_at=now, completed_at=now)

        t = await get_task(task.id)
        assert t.started_at == t.completed_at

    @pytest.mark.asyncio
    async def test_invalid_ordering_does_not_mutate_task(self, async_redis_client):
        """A rejected timestamp update should leave the task unchanged."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)

        original_started = datetime(2025, 6, 1, tzinfo=UTC)
        await update_task_status(task.id, started_at=original_started)

        with pytest.raises(ValueError):
            await update_task_status(
                task.id,
                completed_at=datetime(2025, 5, 1, tzinfo=UTC),
            )

        t = await get_task(task.id)
        assert t.started_at == original_started
        assert t.completed_at is None
