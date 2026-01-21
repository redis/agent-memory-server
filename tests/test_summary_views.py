"""Tests for Summary Views feature including documentation examples.

These tests verify that all examples from docs/summary-views.md work correctly.
"""

import pytest

from agent_memory_server.models import MemoryRecord, SummaryView, TaskStatusEnum


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


@pytest.mark.asyncio
async def test_fetch_long_term_memories_for_view_paginates(monkeypatch):
    """_fetch_long_term_memories_for_view should paginate through results.

    We monkeypatch long_term_memory.search_long_term_memories to return
    deterministic pages and verify that multiple calls are made when the
    number of results exceeds the configured page_size.
    """

    from agent_memory_server import summary_views

    calls: list[tuple[int, int]] = []

    class FakeResults:
        def __init__(self, memories: list[MemoryRecord]):
            self.memories = memories

    async def fake_search_long_term_memories(
        *, text: str, limit: int, offset: int, **_: object
    ):  # type: ignore[override]
        # Record the (limit, offset) pair for assertions.
        calls.append((limit, offset))

        # Pretend we have 2500 total memories; each page returns `limit`
        # until we reach that total.
        total = 2500
        remaining = max(total - offset, 0)
        batch_size = min(limit, remaining)

        memories = [
            MemoryRecord(
                id=f"mem-{offset + i}",
                text=f"memory {offset + i}",
                session_id=None,
                user_id=None,
                namespace=None,
            )
            for i in range(batch_size)
        ]
        return FakeResults(memories)

    monkeypatch.setattr(
        summary_views.long_term_memory,
        "search_long_term_memories",
        fake_search_long_term_memories,
    )

    view = SummaryView(
        id="view-1",
        name="test",
        source="long_term",
        group_by=["user_id"],
        filters={},
        time_window_days=None,
        continuous=False,
        prompt=None,
        model_name=None,
    )

    # Use a small page_size so multiple pages are required; also set
    # an overall_limit below the total so we exercise that branch.
    memories = await summary_views._fetch_long_term_memories_for_view(
        view,
        extra_group=None,
        page_size=1000,
        overall_limit=2100,
    )

    # We should have respected the overall_limit.
    assert len(memories) == 2100

    # And we should have made at least two paginated calls with advancing
    # offsets.
    assert calls[0] == (1000, 0)
    assert calls[1] == (1000, 1000)
    # The final page only needs 100 records to reach 2100.
    assert calls[2] == (100, 2000)


def test_encode_partition_key_handles_special_characters():
    """encode_partition_key should URL-encode special characters in values."""

    from agent_memory_server.summary_views import (
        decode_partition_key,
        encode_partition_key,
    )

    # Values containing the delimiter characters '|' and '='
    group = {"user_id": "alice|bob", "namespace": "test=value"}

    encoded = encode_partition_key(group)

    # The encoded key should not have raw '|' or '=' from values
    # (keys are sorted alphabetically, so namespace comes first)
    assert "alice%7Cbob" in encoded  # %7C is URL-encoded '|'
    assert "test%3Dvalue" in encoded  # %3D is URL-encoded '='

    # Decoding should restore the original values
    decoded = decode_partition_key(encoded)
    assert decoded == group


def test_encode_partition_key_is_stable():
    """encode_partition_key should produce the same key regardless of dict order."""

    from agent_memory_server.summary_views import encode_partition_key

    group1 = {"user_id": "alice", "namespace": "chat"}
    group2 = {"namespace": "chat", "user_id": "alice"}

    assert encode_partition_key(group1) == encode_partition_key(group2)


# =============================================================================
# Documentation Examples Tests
# These tests verify all examples from docs/summary-views.md work correctly
# =============================================================================


class TestDocumentationExamples:
    """Tests that verify documentation examples work correctly."""

    @pytest.mark.asyncio
    async def test_docs_user_profile_summaries_example(self, client):
        """Verify the 'User Profile Summaries' use case example from docs."""
        # From docs: Use Case 1 - User Profile Summaries
        payload = {
            "name": "user_profile_30d",
            "source": "long_term",
            "group_by": ["user_id"],
            "filters": {"memory_type": "semantic"},
            "time_window_days": 30,
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["name"] == "user_profile_30d"
        assert view["group_by"] == ["user_id"]
        assert view["filters"] == {"memory_type": "semantic"}
        assert view["time_window_days"] == 30

    @pytest.mark.asyncio
    async def test_docs_namespace_digest_example(self, client):
        """Verify the 'Namespace Knowledge Digests' use case example from docs."""
        # From docs: Use Case 2 - Namespace Knowledge Digests
        payload = {
            "name": "namespace_digest",
            "source": "long_term",
            "group_by": ["namespace"],
            "filters": {},
            "time_window_days": 7,
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["name"] == "namespace_digest"
        assert view["group_by"] == ["namespace"]
        assert view["time_window_days"] == 7

    @pytest.mark.asyncio
    async def test_docs_session_recap_example(self, client):
        """Verify the 'Session Recaps' use case example from docs."""
        # From docs: Use Case 3 - Session Recaps
        payload = {
            "name": "session_recap",
            "source": "long_term",
            "group_by": ["session_id"],
            "filters": {"memory_type": "episodic"},
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["name"] == "session_recap"
        assert view["group_by"] == ["session_id"]
        assert view["filters"] == {"memory_type": "episodic"}

    @pytest.mark.asyncio
    async def test_docs_create_summary_view_full_example(self, client):
        """Verify the full 'Create a Summary View' API example from docs."""
        # From docs: API Endpoints - Create a Summary View
        payload = {
            "name": "ltm_by_user_30d",
            "source": "long_term",
            "group_by": ["user_id"],
            "filters": {"memory_type": "semantic"},
            "time_window_days": 30,
            "continuous": False,
            "prompt": "Summarize key facts and preferences for this user.",
            "model_name": "gpt-4o-mini",
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()

        # Verify response structure matches docs
        assert "id" in view
        assert view["name"] == "ltm_by_user_30d"
        assert view["source"] == "long_term"
        assert view["group_by"] == ["user_id"]
        assert view["filters"] == {"memory_type": "semantic"}
        assert view["time_window_days"] == 30
        assert view["continuous"] is False
        assert view["prompt"] == "Summarize key facts and preferences for this user."
        assert view["model_name"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_docs_list_summary_views(self, client):
        """Verify the 'List Summary Views' API endpoint from docs."""
        # Create a view first
        payload = {"name": "test_list", "source": "long_term", "group_by": ["user_id"]}
        await client.post("/v1/summary-views", json=payload)

        # From docs: GET /v1/summary-views
        resp = await client.get("/v1/summary-views")
        assert resp.status_code == 200
        views = resp.json()
        assert isinstance(views, list)
        assert len(views) >= 1

    @pytest.mark.asyncio
    async def test_docs_get_summary_view(self, client):
        """Verify the 'Get a Summary View' API endpoint from docs."""
        # Create a view first
        payload = {"name": "test_get", "source": "long_term", "group_by": ["user_id"]}
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        # From docs: GET /v1/summary-views/{view_id}
        resp = await client.get(f"/v1/summary-views/{view_id}")
        assert resp.status_code == 200
        view = resp.json()
        assert view["id"] == view_id
        assert view["name"] == "test_get"

    @pytest.mark.asyncio
    async def test_docs_delete_summary_view(self, client):
        """Verify the 'Delete a Summary View' API endpoint from docs."""
        # Create a view first
        payload = {
            "name": "test_delete",
            "source": "long_term",
            "group_by": ["user_id"],
        }
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        # From docs: DELETE /v1/summary-views/{view_id}
        resp = await client.delete(f"/v1/summary-views/{view_id}")
        assert resp.status_code == 200

        # Verify deletion
        get_resp = await client.get(f"/v1/summary-views/{view_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_docs_run_single_partition_example(self, client):
        """Verify the 'Run a Single Partition' API example from docs."""
        # Create a view first
        payload = {
            "name": "test_partition",
            "source": "long_term",
            "group_by": ["user_id"],
        }
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        # From docs: POST /v1/summary-views/{view_id}/partitions/run
        run_payload = {"group": {"user_id": "alice"}}
        resp = await client.post(
            f"/v1/summary-views/{view_id}/partitions/run", json=run_payload
        )
        assert resp.status_code == 200, resp.text
        result = resp.json()

        # Verify response structure matches docs
        assert result["view_id"] == view_id
        assert result["group"] == {"user_id": "alice"}
        assert "summary" in result
        assert "memory_count" in result
        assert result["memory_count"] >= 0
        assert "computed_at" in result

    @pytest.mark.asyncio
    async def test_docs_run_all_partitions_async_example(self, client):
        """Verify the 'Run All Partitions (Async)' API example from docs."""
        # Create a view first
        payload = {
            "name": "test_full_run",
            "source": "long_term",
            "group_by": ["user_id"],
        }
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        # From docs: POST /v1/summary-views/{view_id}/run
        run_payload = {"task_id": "optional-client-provided-id"}
        resp = await client.post(f"/v1/summary-views/{view_id}/run", json=run_payload)
        assert resp.status_code == 200, resp.text
        task = resp.json()

        # Verify response structure matches docs
        assert task["id"] == "optional-client-provided-id"
        assert task["type"] == "summary_view_full_run"
        assert task["status"] in {"pending", "running", "success"}
        assert task["view_id"] == view_id

    @pytest.mark.asyncio
    async def test_docs_list_partition_results_with_filter(self, client):
        """Verify the 'List Partition Results' API with filtering from docs."""
        # Create a view and run a partition
        payload = {
            "name": "test_list_partitions",
            "source": "long_term",
            "group_by": ["user_id"],
        }
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        # Run partition for alice
        await client.post(
            f"/v1/summary-views/{view_id}/partitions/run",
            json={"group": {"user_id": "alice"}},
        )

        # From docs: GET /v1/summary-views/{view_id}/partitions?user_id=alice
        resp = await client.get(
            f"/v1/summary-views/{view_id}/partitions", params={"user_id": "alice"}
        )
        assert resp.status_code == 200
        partitions = resp.json()
        assert isinstance(partitions, list)
        assert len(partitions) == 1
        assert partitions[0]["group"]["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_docs_continuous_mode_example(self, client):
        """Verify the 'Continuous Mode' example from docs."""
        # From docs: Continuous Mode example
        payload = {
            "name": "always_fresh_user_summaries",
            "source": "long_term",
            "group_by": ["user_id"],
            "continuous": True,
            "time_window_days": 7,
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["continuous"] is True
        assert view["time_window_days"] == 7

    @pytest.mark.asyncio
    async def test_docs_custom_prompt_example(self, client):
        """Verify the 'Custom Prompts' example from docs."""
        # From docs: Custom Prompts example
        payload = {
            "name": "technical_summary",
            "source": "long_term",
            "group_by": ["user_id"],
            "prompt": "Focus on technical skills, programming languages, and project experience. Output as bullet points.",
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert (
            view["prompt"]
            == "Focus on technical skills, programming languages, and project experience. Output as bullet points."
        )

    @pytest.mark.asyncio
    async def test_docs_task_polling_example(self, client):
        """Verify the 'Task Polling' workflow from docs."""
        # Create and run a view
        payload = {
            "name": "test_task_polling",
            "source": "long_term",
            "group_by": ["user_id"],
        }
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        run_resp = await client.post(f"/v1/summary-views/{view_id}/run", json={})
        task_id = run_resp.json()["id"]

        # From docs: GET /v1/tasks/{task_id}
        resp = await client.get(f"/v1/tasks/{task_id}")
        assert resp.status_code == 200
        task = resp.json()

        # Verify possible statuses from docs
        assert task["status"] in {"pending", "running", "success", "failed"}


class TestSupportedGroupByAndFilters:
    """Tests verifying all supported group_by and filter keys from documentation."""

    @pytest.mark.asyncio
    async def test_all_supported_group_by_keys(self, client):
        """Verify all group_by keys documented work: user_id, namespace, session_id, memory_type."""
        supported_keys = ["user_id", "namespace", "session_id", "memory_type"]

        for key in supported_keys:
            payload = {
                "name": f"test_group_by_{key}",
                "source": "long_term",
                "group_by": [key],
            }
            resp = await client.post("/v1/summary-views", json=payload)
            assert resp.status_code == 200, f"group_by [{key}] failed: {resp.text}"

    @pytest.mark.asyncio
    async def test_all_supported_filter_keys(self, client):
        """Verify all filter keys documented work: user_id, namespace, session_id, memory_type."""
        supported_filters = {
            "user_id": "test_user",
            "namespace": "test_ns",
            "session_id": "test_session",
            "memory_type": "semantic",
        }

        for key, value in supported_filters.items():
            payload = {
                "name": f"test_filter_{key}",
                "source": "long_term",
                "group_by": [],
                "filters": {key: value},
            }
            resp = await client.post("/v1/summary-views", json=payload)
            assert resp.status_code == 200, f"filter {key}={value} failed: {resp.text}"

    @pytest.mark.asyncio
    async def test_combined_group_by_keys(self, client):
        """Verify multiple group_by keys work together."""
        payload = {
            "name": "test_combined_group_by",
            "source": "long_term",
            "group_by": ["user_id", "namespace"],
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert set(view["group_by"]) == {"user_id", "namespace"}

    @pytest.mark.asyncio
    async def test_combined_filters(self, client):
        """Verify multiple filters work together."""
        payload = {
            "name": "test_combined_filters",
            "source": "long_term",
            "group_by": ["user_id"],
            "filters": {"namespace": "test_ns", "memory_type": "semantic"},
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["filters"] == {"namespace": "test_ns", "memory_type": "semantic"}


class TestConfigurationOptionsTable:
    """Tests verifying configuration options documented in the SummaryView Fields table."""

    @pytest.mark.asyncio
    async def test_all_optional_fields_can_be_omitted(self, client):
        """Verify only 'source' is required, all others are optional."""
        # Minimal payload with only required field
        payload = {"source": "long_term"}
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 200, resp.text
        view = resp.json()

        # Verify defaults
        assert view["name"] is None
        assert view["group_by"] == []
        assert view["filters"] == {}
        assert view["time_window_days"] is None
        assert view["continuous"] is False
        assert view["prompt"] is None
        assert view["model_name"] is None

    @pytest.mark.asyncio
    async def test_time_window_days_must_be_positive(self, client):
        """Verify time_window_days validation (ge=1 from model)."""
        payload = {
            "source": "long_term",
            "time_window_days": 0,
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_source_must_be_long_term(self, client):
        """Verify only 'long_term' source is supported (per docs)."""
        payload = {
            "source": "working_memory",
            "group_by": ["user_id"],
        }
        resp = await client.post("/v1/summary-views", json=payload)
        assert resp.status_code == 400
        assert "long_term" in resp.json()["detail"]


class TestPartitionResultStructure:
    """Tests verifying SummaryViewPartitionResult structure from documentation."""

    @pytest.mark.asyncio
    async def test_partition_result_has_all_documented_fields(self, client):
        """Verify partition result contains all fields documented."""
        # Create view
        payload = {
            "name": "test_result_structure",
            "source": "long_term",
            "group_by": ["user_id"],
        }
        create_resp = await client.post("/v1/summary-views", json=payload)
        view_id = create_resp.json()["id"]

        # Run partition
        resp = await client.post(
            f"/v1/summary-views/{view_id}/partitions/run",
            json={"group": {"user_id": "bob"}},
        )
        assert resp.status_code == 200
        result = resp.json()

        # Verify all documented fields exist
        assert "view_id" in result
        assert "group" in result
        assert "summary" in result
        assert "memory_count" in result
        assert "computed_at" in result

        # Verify types
        assert isinstance(result["view_id"], str)
        assert isinstance(result["group"], dict)
        assert isinstance(result["summary"], str)
        assert isinstance(result["memory_count"], int)
        assert isinstance(result["computed_at"], str)  # ISO format string
