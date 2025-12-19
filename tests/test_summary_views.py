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
async def test_create_summary_view_rejects_invalid_keys(client):
    """SummaryView creation should reject unsupported group_by / filter keys."""

    payload = {
        "name": "invalid_keys_view",
        "source": "long_term",
        # "invalid" is not in the allowed group_by set
        "group_by": ["user_id", "invalid"],
        "filters": {"memory_type": "semantic"},
        "time_window_days": 30,
        "continuous": False,
        "prompt": None,
        "model_name": None,
    }

    resp = await client.post("/v1/summary-views", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert "Unsupported group_by fields" in data["detail"]


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
async def test_delete_summary_view_removes_it_from_get_and_list(client):
    """Deleting a SummaryView should remove it from retrieval and listings."""

    # Create a view we can delete
    payload = {
        "name": "ltm_to_delete",
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

    # Ensure it appears in the list
    list_before = await client.get("/v1/summary-views")
    assert list_before.status_code == 200
    ids_before = {v["id"] for v in list_before.json()}
    assert view_id in ids_before

    # Delete the view
    resp_delete = await client.delete(f"/v1/summary-views/{view_id}")
    assert resp_delete.status_code == 200, resp_delete.text

    # GET should now return 404
    resp_get = await client.get(f"/v1/summary-views/{view_id}")
    assert resp_get.status_code == 404

    # And it should no longer appear in the list
    list_after = await client.get("/v1/summary-views")
    assert list_after.status_code == 200
    ids_after = {v["id"] for v in list_after.json()}
    assert view_id not in ids_after


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

    # Poll the task status via the API. We intentionally do not wait for the
    # background Docket worker here; the goal is to verify that the Task is
    # created and visible through the status endpoint, not that the worker
    # has actually completed the refresh.
    resp_task = await client.get(f"/v1/tasks/{task_id}")
    assert resp_task.status_code == 200
    polled = resp_task.json()
    assert polled["status"] in {
        TaskStatusEnum.PENDING,
        TaskStatusEnum.RUNNING,
        TaskStatusEnum.SUCCESS,
    }
