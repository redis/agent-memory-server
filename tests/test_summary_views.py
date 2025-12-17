import pytest

from agent_memory_server.models import TaskStatusEnum


@pytest.mark.asyncio
async def test_create_and_get_summary_view(client):
    # Create a summary view
    payload = {
        "name": "ltm_by_user_30d",
        "source": "long_term",
        "group_by": ["user_id"],
        "filters": {"memory_type": "semantic"},
        "time_window_days": 30,
        "continuous": False,
        "prompt": None,
        "model_name": None,
    }
    resp = await client.post("/v1/summary-views", json=payload)
    assert resp.status_code == 200, resp.text
    view = resp.json()
    view_id = view["id"]

    # Fetch it back
    resp_get = await client.get(f"/v1/summary-views/{view_id}")
    assert resp_get.status_code == 200
    fetched = resp_get.json()
    assert fetched["id"] == view_id
    assert fetched["group_by"] == ["user_id"]


@pytest.mark.asyncio
async def test_run_single_partition_and_list_partitions(client):
    # Create a simple view grouped by user_id
    payload = {
        "name": "ltm_by_user",
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

    # Run a single partition synchronously
    run_payload = {"group": {"user_id": "alice"}}
    resp_run = await client.post(
        f"/v1/summary-views/{view_id}/partitions/run", json=run_payload
    )
    assert resp_run.status_code == 200, resp_run.text
    result = resp_run.json()
    assert result["group"] == {"user_id": "alice"}
    assert "summary" in result

    # List materialized partitions
    resp_list = await client.get(
        f"/v1/summary-views/{view_id}/partitions", params={"user_id": "alice"}
    )
    assert resp_list.status_code == 200
    partitions = resp_list.json()
    assert len(partitions) == 1
    assert partitions[0]["group"]["user_id"] == "alice"


@pytest.mark.asyncio
async def test_run_full_view_creates_task_and_updates_status(client):
    # Create a summary view
    payload = {
        "name": "ltm_full_run",
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

    # Trigger a full run
    resp_run = await client.post(f"/v1/summary-views/{view_id}/run", json={})
    assert resp_run.status_code == 200, resp_run.text
    task = resp_run.json()
    task_id = task["id"]

    # Poll the task status via the API
    resp_task = await client.get(f"/v1/tasks/{task_id}")
    assert resp_task.status_code == 200
    polled = resp_task.json()
    assert polled["status"] in {
        TaskStatusEnum.PENDING,
        TaskStatusEnum.RUNNING,
        TaskStatusEnum.SUCCESS,
    }
