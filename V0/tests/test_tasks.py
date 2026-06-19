import pytest

from agent_memory_server.models import TaskStatusEnum


@pytest.mark.asyncio
async def test_task_lifecycle_via_api(client):
    """Basic sanity check for Task creation and retrieval via the API.

    This verifies that:
    - POST /v1/summary-views/{id}/run creates a Task
    - GET /v1/tasks/{task_id} returns that Task with the expected ID and type
    """

    # Create a minimal summary view we can run
    payload = {
        "name": "task_lifecycle_test_view",
        "source": "long_term",
        "group_by": ["user_id"],
        "filters": {},
        "time_window_days": None,
        "continuous": False,
        "prompt": None,
        "model_name": None,
    }
    resp = await client.post("/v1/summary-views", json=payload)
    assert resp.status_code == 200, resp.text
    view_id = resp.json()["id"]

    # Trigger a full run to create a Task
    resp_run = await client.post(f"/v1/summary-views/{view_id}/run", json={})
    assert resp_run.status_code == 200, resp_run.text
    task = resp_run.json()

    assert task["id"]
    assert task["view_id"] == view_id
    assert task["status"] == TaskStatusEnum.PENDING

    task_id = task["id"]

    # Fetch the task via the task status endpoint
    resp_task = await client.get(f"/v1/tasks/{task_id}")
    assert resp_task.status_code == 200, resp_task.text
    polled = resp_task.json()

    assert polled["id"] == task_id
    assert polled["view_id"] == view_id
    assert polled["status"] in {
        TaskStatusEnum.PENDING,
        TaskStatusEnum.RUNNING,
        TaskStatusEnum.SUCCESS,
    }
