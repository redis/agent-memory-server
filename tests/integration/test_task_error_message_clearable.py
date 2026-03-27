"""Test that task error_message can be cleared.

Regression test for https://github.com/redis/agent-memory-server/issues/206
"""

import pytest
from ulid import ULID

from agent_memory_server.models import Task, TaskStatusEnum, TaskTypeEnum
from agent_memory_server.tasks import create_task, get_task, update_task_status


def _make_task(**overrides) -> Task:
    defaults = {
        "id": str(ULID()),
        "type": TaskTypeEnum.SUMMARY_VIEW_FULL_RUN,
        "view_id": "test-view",
    }
    defaults.update(overrides)
    return Task(**defaults)


class TestErrorMessageClearable:
    """error_message should be clearable by passing empty string."""

    @pytest.mark.asyncio
    async def test_clear_error_message_with_empty_string(self, async_redis_client):
        """Passing error_message='' should clear a previously set error."""
        task = _make_task(status=TaskStatusEnum.FAILED)
        await create_task(task)

        # Set an error
        await update_task_status(
            task.id,
            error_message="Something broke",
        )
        t1 = await get_task(task.id)
        assert t1.error_message == "Something broke"

        # Clear it
        await update_task_status(
            task.id,
            error_message="",
        )
        t2 = await get_task(task.id)
        assert (
            t2.error_message is None
        ), "Empty string should clear error_message to None"

    @pytest.mark.asyncio
    async def test_none_does_not_change_error_message(self, async_redis_client):
        """Omitting error_message (defaults to _UNSET) should leave the field unchanged."""
        task = _make_task(status=TaskStatusEnum.FAILED)
        await create_task(task)

        await update_task_status(task.id, error_message="Original error")
        await update_task_status(task.id)  # error_message defaults to None

        t = await get_task(task.id)
        assert t.error_message == "Original error"

    @pytest.mark.asyncio
    async def test_set_new_error_replaces_old(self, async_redis_client):
        """Passing a non-empty error_message should replace the existing one."""
        task = _make_task(status=TaskStatusEnum.FAILED)
        await create_task(task)

        await update_task_status(task.id, error_message="First")
        await update_task_status(task.id, error_message="Second")

        t = await get_task(task.id)
        assert t.error_message == "Second"
