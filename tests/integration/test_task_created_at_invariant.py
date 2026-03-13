"""Test that Task.created_at is always set and never null.

Regression test for https://github.com/redis/agent-memory-server/issues/208

The Task model defines created_at as a non-optional datetime with a
default_factory, so every task created through the normal API always has
created_at set.  This test verifies that invariant holds and that the
previously dead backfill code in update_task_status is no longer needed.
"""

import json
from datetime import UTC, datetime

import pytest
from ulid import ULID

from agent_memory_server.models import Task, TaskStatusEnum, TaskTypeEnum
from agent_memory_server.tasks import (
    _task_key,
    create_task,
    get_task,
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


class TestCreatedAtInvariant:
    """created_at is always set on valid tasks and never null."""

    @pytest.mark.asyncio
    async def test_created_at_always_populated_on_create(self, async_redis_client):
        """create_task should always produce a task with created_at set."""
        before = datetime.now(UTC)
        task = _make_task()
        await create_task(task)
        after = datetime.now(UTC)

        retrieved = await get_task(task.id)
        assert retrieved.created_at is not None
        assert before <= retrieved.created_at <= after

    @pytest.mark.asyncio
    async def test_created_at_preserved_through_updates(self, async_redis_client):
        """Updating a task should never change its created_at."""
        task = _make_task()
        await create_task(task)
        original = (await get_task(task.id)).created_at

        await update_task_status(task.id, status=TaskStatusEnum.RUNNING)
        await update_task_status(task.id, status=TaskStatusEnum.SUCCESS)

        final = await get_task(task.id)
        assert final.created_at == original

    @pytest.mark.asyncio
    async def test_null_created_at_in_redis_is_unrecoverable(self, async_redis_client):
        """A task with created_at=null in Redis cannot be parsed by the
        Task model.  Both get_task and update_task_status treat it as
        corrupt data (return None / no-op)."""
        from agent_memory_server.utils.redis import get_redis_conn

        task_id = str(ULID())
        corrupt_json = json.dumps(
            {
                "id": task_id,
                "type": "summary_view_full_run",
                "status": "pending",
                "view_id": None,
                "created_at": None,
                "started_at": None,
                "completed_at": None,
                "error_message": None,
            }
        )
        redis = await get_redis_conn()
        await redis.set(_task_key(task_id), corrupt_json, ex=300)

        # get_task returns None — the task is unrecoverable
        assert await get_task(task_id) is None

        # update_task_status is a no-op — it can't parse the corrupt task
        await update_task_status(task_id, status=TaskStatusEnum.RUNNING)
        assert await get_task(task_id) is None
