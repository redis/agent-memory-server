import logging
from datetime import UTC, datetime

from agent_memory_server.models import Task, TaskStatusEnum
from agent_memory_server.utils.redis import get_redis_conn


logger = logging.getLogger(__name__)


def _task_key(task_id: str) -> str:
    """Return the Redis key for a task JSON payload."""

    return f"task:{task_id}"


async def create_task(task: Task) -> None:
    """Persist a new Task as JSON in Redis.

    This overwrites any existing task with the same ID.
    """

    redis = await get_redis_conn()
    await redis.set(_task_key(task.id), task.model_dump_json())


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

    if status is not None:
        task.status = status
    if started_at is not None:
        task.started_at = started_at
    if completed_at is not None:
        task.completed_at = completed_at
    if error_message is not None:
        task.error_message = error_message

    # Ensure created_at is always set
    if task.created_at is None:
        task.created_at = datetime.now(UTC)

    await redis.set(key, task.model_dump_json())
