import logging
from datetime import datetime

from agent_memory_server.models import Task, TaskStatusEnum
from agent_memory_server.utils.redis import get_redis_conn


logger = logging.getLogger(__name__)


# Tasks are operational metadata; we don't need to retain them forever.
# Use a conservative TTL so Redis state cannot grow without bound.
_TASK_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


class InvalidTaskTransitionError(Exception):
    """Raised when a task status transition is not allowed."""


# Valid state machine transitions.  A same-status "transition" (e.g.
# RUNNING → RUNNING) is always allowed as an idempotent no-op.
_VALID_TRANSITIONS: dict[TaskStatusEnum, set[TaskStatusEnum]] = {
    TaskStatusEnum.PENDING: {TaskStatusEnum.RUNNING, TaskStatusEnum.FAILED},
    TaskStatusEnum.RUNNING: {TaskStatusEnum.SUCCESS, TaskStatusEnum.FAILED},
    TaskStatusEnum.SUCCESS: set(),
    TaskStatusEnum.FAILED: set(),
}


def _task_key(task_id: str) -> str:
    """Return the Redis key for a task JSON payload."""

    return f"task:{task_id}"


async def create_task(task: Task) -> None:
    """Persist a new Task as JSON in Redis.

    This overwrites any existing task with the same ID.
    """

    redis = await get_redis_conn()
    await redis.set(
        _task_key(task.id),
        task.model_dump_json(),
        ex=_TASK_TTL_SECONDS,
    )


async def get_task(task_id: str) -> Task | None:
    """Load a Task from Redis JSON storage.

    Returns None if the task does not exist.
    """

    redis = await get_redis_conn()
    raw = await redis.get(_task_key(task_id))
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        return Task.model_validate_json(raw)
    except Exception:
        logger.exception("Failed to decode task JSON for %s", task_id)
        return None


async def update_task_status(
    task_id: str,
    *,
    status: TaskStatusEnum | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error_message: str | None = None,
) -> None:
    """Update status and timestamps for an existing Task.

    If the task does not exist, this is a no-op.

    Raises:
        InvalidTaskTransitionError: If the requested status transition
            violates the task state machine.
    """

    redis = await get_redis_conn()
    key = _task_key(task_id)
    raw = await redis.get(key)
    if raw is None:
        logger.warning("Attempted to update missing task %s", task_id)
        return

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        task = Task.model_validate_json(raw)
    except Exception:
        logger.exception("Failed to decode task JSON for %s during update", task_id)
        return

    if status is not None and status != task.status:
        allowed = _VALID_TRANSITIONS.get(task.status, set())
        if status not in allowed:
            raise InvalidTaskTransitionError(
                f"Cannot transition task {task_id} from {task.status.value!r} "
                f"to {status.value!r}"
            )
        task.status = status
    if started_at is not None:
        task.started_at = started_at
    if completed_at is not None:
        task.completed_at = completed_at
    if error_message is not None:
        task.error_message = error_message

    await redis.set(key, task.model_dump_json(), ex=_TASK_TTL_SECONDS)
