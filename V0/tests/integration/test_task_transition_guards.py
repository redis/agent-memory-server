"""Test that task status transitions are validated.

Regression tests for https://github.com/redis/agent-memory-server/issues/205
"""

import pytest
from ulid import ULID

from agent_memory_server.models import Task, TaskStatusEnum, TaskTypeEnum
from agent_memory_server.tasks import (
    InvalidTaskTransitionError,
    create_task,
    update_task_status,
)


def _make_task(**overrides) -> Task:
    defaults = {
        "id": str(ULID()),
        "type": TaskTypeEnum.SUMMARY_VIEW_FULL_RUN,
        "view_id": "test-view",
    }
    defaults.update(overrides)
    return Task(**defaults)


class TestTaskTransitionGuards:
    """update_task_status should reject invalid status transitions."""

    @pytest.mark.asyncio
    async def test_valid_pending_to_running(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.PENDING)
        await create_task(task)
        await update_task_status(task.id, status=TaskStatusEnum.RUNNING)
        # Should not raise

    @pytest.mark.asyncio
    async def test_valid_pending_to_failed(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.PENDING)
        await create_task(task)
        await update_task_status(task.id, status=TaskStatusEnum.FAILED)

    @pytest.mark.asyncio
    async def test_valid_running_to_success(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)
        await update_task_status(task.id, status=TaskStatusEnum.SUCCESS)

    @pytest.mark.asyncio
    async def test_valid_running_to_failed(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)
        await update_task_status(task.id, status=TaskStatusEnum.FAILED)

    @pytest.mark.asyncio
    async def test_rejects_success_to_running(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.SUCCESS)
        await create_task(task)
        with pytest.raises(InvalidTaskTransitionError):
            await update_task_status(task.id, status=TaskStatusEnum.RUNNING)

    @pytest.mark.asyncio
    async def test_rejects_success_to_pending(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.SUCCESS)
        await create_task(task)
        with pytest.raises(InvalidTaskTransitionError):
            await update_task_status(task.id, status=TaskStatusEnum.PENDING)

    @pytest.mark.asyncio
    async def test_rejects_failed_to_success(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.FAILED)
        await create_task(task)
        with pytest.raises(InvalidTaskTransitionError):
            await update_task_status(task.id, status=TaskStatusEnum.SUCCESS)

    @pytest.mark.asyncio
    async def test_rejects_failed_to_running(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.FAILED)
        await create_task(task)
        with pytest.raises(InvalidTaskTransitionError):
            await update_task_status(task.id, status=TaskStatusEnum.RUNNING)

    @pytest.mark.asyncio
    async def test_rejects_failed_to_pending(self, async_redis_client):
        task = _make_task(status=TaskStatusEnum.FAILED)
        await create_task(task)
        with pytest.raises(InvalidTaskTransitionError):
            await update_task_status(task.id, status=TaskStatusEnum.PENDING)

    @pytest.mark.asyncio
    async def test_same_status_update_is_noop(self, async_redis_client):
        """Updating to the same status should be allowed (idempotent)."""
        task = _make_task(status=TaskStatusEnum.RUNNING)
        await create_task(task)
        await update_task_status(task.id, status=TaskStatusEnum.RUNNING)
        # Should not raise

    @pytest.mark.asyncio
    async def test_no_status_change_skips_validation(self, async_redis_client):
        """When status=None, only other fields are updated — no transition check."""
        task = _make_task(status=TaskStatusEnum.SUCCESS)
        await create_task(task)
        await update_task_status(task.id, error_message="late note")
        # Should not raise — status wasn't changed

    @pytest.mark.asyncio
    async def test_invalid_transition_does_not_mutate_task(self, async_redis_client):
        """A rejected transition should leave the task in its original state."""
        task = _make_task(status=TaskStatusEnum.SUCCESS)
        await create_task(task)

        with pytest.raises(InvalidTaskTransitionError):
            await update_task_status(task.id, status=TaskStatusEnum.RUNNING)

        from agent_memory_server.tasks import get_task

        unchanged = await get_task(task.id)
        assert unchanged.status == TaskStatusEnum.SUCCESS
