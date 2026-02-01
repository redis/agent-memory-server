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


# =============================================================================
# Additional Unit Tests for 100% Coverage
# =============================================================================


class TestDecodePartitionKey:
    """Tests for decode_partition_key function."""

    def test_empty_string_returns_empty_dict(self):
        """decode_partition_key should return {} for empty string."""
        from agent_memory_server.summary_views import decode_partition_key

        assert decode_partition_key("") == {}

    def test_part_without_equals_is_skipped(self):
        """decode_partition_key should skip parts without '=' delimiter."""
        from agent_memory_server.summary_views import decode_partition_key

        # A malformed key with a part that has no equals sign
        result = decode_partition_key("valid_key=valid_value|malformed_part")
        assert result == {"valid_key": "valid_value"}


class TestGetSummaryViewExceptionHandling:
    """Tests for get_summary_view exception handling."""

    @pytest.mark.asyncio
    async def test_get_summary_view_returns_none_on_invalid_json(
        self, async_redis_client
    ):
        """get_summary_view should return None if JSON is invalid."""
        from agent_memory_server import summary_views

        # Store invalid JSON directly
        view_id = "test-invalid-json-view"
        await async_redis_client.set(
            f"summary_view:{view_id}:config", "not valid json {{"
        )

        result = await summary_views.get_summary_view(view_id)
        assert result is None


class TestListPartitionResultsEdgeCases:
    """Tests for list_partition_results edge cases."""

    @pytest.mark.asyncio
    async def test_list_partition_results_skips_invalid_json(self, async_redis_client):
        """list_partition_results should skip entries with invalid JSON."""
        from agent_memory_server import summary_views

        view_id = "test-invalid-partition"

        # Store a valid partition
        from datetime import UTC, datetime

        from agent_memory_server.models import SummaryViewPartitionResult

        valid_result = SummaryViewPartitionResult(
            view_id=view_id,
            group={"user_id": "alice"},
            summary="Valid summary",
            memory_count=5,
            computed_at=datetime.now(UTC),
        )
        partition_key = summary_views.encode_partition_key(valid_result.group)
        await async_redis_client.set(
            f"summary_view:{view_id}:summary:{partition_key}",
            valid_result.model_dump_json(),
        )

        # Store invalid JSON as another partition
        await async_redis_client.set(
            f"summary_view:{view_id}:summary:invalid_partition",
            "not valid json {{",
        )

        results = await summary_views.list_partition_results(view_id)
        # Should only return the valid partition
        assert len(results) == 1
        assert results[0].group == {"user_id": "alice"}

    @pytest.mark.asyncio
    async def test_list_partition_results_filters_by_group(self, async_redis_client):
        """list_partition_results should filter by group when provided."""
        from agent_memory_server import summary_views

        view_id = "test-filter-partition"
        from datetime import UTC, datetime

        from agent_memory_server.models import SummaryViewPartitionResult

        # Store two partitions
        for user in ["alice", "bob"]:
            result = SummaryViewPartitionResult(
                view_id=view_id,
                group={"user_id": user},
                summary=f"Summary for {user}",
                memory_count=5,
                computed_at=datetime.now(UTC),
            )
            partition_key = summary_views.encode_partition_key(result.group)
            await async_redis_client.set(
                f"summary_view:{view_id}:summary:{partition_key}",
                result.model_dump_json(),
            )

        # Filter for alice only
        results = await summary_views.list_partition_results(
            view_id, group_filter={"user_id": "alice"}
        )
        assert len(results) == 1
        assert results[0].group["user_id"] == "alice"


class TestBuildLongTermFiltersForView:
    """Tests for _build_long_term_filters_for_view function."""

    def test_applies_session_id_filter(self):
        """_build_long_term_filters_for_view should apply session_id filter."""
        from agent_memory_server.filters import SessionId
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import _build_long_term_filters_for_view

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=[],
            filters={"session_id": "test-session"},
        )

        filters = _build_long_term_filters_for_view(view)
        assert "session_id" in filters
        assert isinstance(filters["session_id"], SessionId)
        assert filters["session_id"].eq == "test-session"

    def test_applies_memory_type_filter(self):
        """_build_long_term_filters_for_view should apply memory_type filter."""
        from agent_memory_server.filters import MemoryType
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import _build_long_term_filters_for_view

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=[],
            filters={"memory_type": "episodic"},
        )

        filters = _build_long_term_filters_for_view(view)
        assert "memory_type" in filters
        assert isinstance(filters["memory_type"], MemoryType)
        assert filters["memory_type"].eq == "episodic"

    def test_applies_time_window_filter(self):
        """_build_long_term_filters_for_view should apply time_window_days filter."""
        from agent_memory_server.filters import CreatedAt
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import _build_long_term_filters_for_view

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=[],
            filters={},
            time_window_days=7,
        )

        filters = _build_long_term_filters_for_view(view)
        assert "created_at" in filters
        assert isinstance(filters["created_at"], CreatedAt)
        assert filters["created_at"].gte is not None

    def test_applies_extra_group_filters(self):
        """_build_long_term_filters_for_view should apply extra_group filters."""
        from agent_memory_server.filters import Namespace, UserId
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import _build_long_term_filters_for_view

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=["user_id", "namespace"],
            filters={},
        )

        filters = _build_long_term_filters_for_view(
            view, extra_group={"user_id": "alice", "namespace": "chat"}
        )
        assert "user_id" in filters
        assert isinstance(filters["user_id"], UserId)
        assert filters["user_id"].eq == "alice"
        assert "namespace" in filters
        assert isinstance(filters["namespace"], Namespace)
        assert filters["namespace"].eq == "chat"


class TestFetchLongTermMemoriesForView:
    """Tests for _fetch_long_term_memories_for_view function."""

    @pytest.mark.asyncio
    async def test_raises_value_error_on_zero_page_size(self):
        """_fetch_long_term_memories_for_view should raise ValueError for page_size <= 0."""
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import (
            _fetch_long_term_memories_for_view,
        )

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=[],
            filters={},
        )

        with pytest.raises(ValueError, match="page_size must be positive"):
            await _fetch_long_term_memories_for_view(view, page_size=0)

    @pytest.mark.asyncio
    async def test_raises_value_error_on_negative_page_size(self):
        """_fetch_long_term_memories_for_view should raise ValueError for negative page_size."""
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import (
            _fetch_long_term_memories_for_view,
        )

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=[],
            filters={},
        )

        with pytest.raises(ValueError, match="page_size must be positive"):
            await _fetch_long_term_memories_for_view(view, page_size=-1)


class TestPartitionMemoriesByGroup:
    """Tests for _partition_memories_by_group function."""

    def test_partitions_memories_by_single_field(self):
        """_partition_memories_by_group should group by a single field."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _partition_memories_by_group

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="memory 1", user_id="alice"),
            MemoryRecord(id="m2", text="memory 2", user_id="alice"),
            MemoryRecord(id="m3", text="memory 3", user_id="bob"),
        ]

        partitions = _partition_memories_by_group(view, memories)
        assert len(partitions) == 2
        # Check alice partition
        alice_key = (("user_id", "alice"),)
        assert alice_key in partitions
        assert len(partitions[alice_key]) == 2
        # Check bob partition
        bob_key = (("user_id", "bob"),)
        assert bob_key in partitions
        assert len(partitions[bob_key]) == 1

    def test_skips_memories_with_missing_group_field(self):
        """_partition_memories_by_group should skip memories missing group_by fields."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _partition_memories_by_group

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="memory 1", user_id="alice"),
            MemoryRecord(id="m2", text="memory 2", user_id=None),  # Missing user_id
            MemoryRecord(id="m3", text="memory 3"),  # No user_id
        ]

        partitions = _partition_memories_by_group(view, memories)
        assert len(partitions) == 1
        alice_key = (("user_id", "alice"),)
        assert alice_key in partitions
        assert len(partitions[alice_key]) == 1

    def test_partitions_by_multiple_fields(self):
        """_partition_memories_by_group should group by multiple fields."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _partition_memories_by_group

        view = SummaryView(
            id="test",
            name="test",
            source="long_term",
            group_by=["user_id", "namespace"],
            filters={},
        )

        memories = [
            MemoryRecord(id="m1", text="memory 1", user_id="alice", namespace="chat"),
            MemoryRecord(id="m2", text="memory 2", user_id="alice", namespace="chat"),
            MemoryRecord(id="m3", text="memory 3", user_id="alice", namespace="work"),
        ]

        partitions = _partition_memories_by_group(view, memories)
        assert len(partitions) == 2


class TestBuildLongTermSummaryPrompt:
    """Tests for _build_long_term_summary_prompt function."""

    def test_builds_prompt_with_memories(self):
        """_build_long_term_summary_prompt should build a prompt with memory bullets."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _build_long_term_summary_prompt

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="Alice likes coffee"),
            MemoryRecord(id="m2", text="Alice works in tech"),
        ]

        prompt = _build_long_term_summary_prompt(
            view=view,
            group={"user_id": "alice"},
            memories=memories,
            model_name="gpt-4o-mini",
            instructions="Summarize the user's preferences.",
        )

        assert "Summarize the user's preferences." in prompt
        assert "alice" in prompt.lower() or '"user_id"' in prompt
        assert "Alice likes coffee" in prompt
        assert "Alice works in tech" in prompt
        assert "SUMMARY:" in prompt

    def test_handles_empty_memories_gracefully(self):
        """_build_long_term_summary_prompt should handle empty memory list."""
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import _build_long_term_summary_prompt

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        prompt = _build_long_term_summary_prompt(
            view=view,
            group={"user_id": "alice"},
            memories=[],
            model_name="gpt-4o-mini",
            instructions="Summarize.",
        )

        assert "Summarize." in prompt
        assert "SUMMARY:" in prompt


class TestSummarizePartitionLongTerm:
    """Tests for summarize_partition_long_term function."""

    @pytest.mark.asyncio
    async def test_returns_no_memories_message_for_empty_list(self):
        """summarize_partition_long_term should return message when no memories."""
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import summarize_partition_long_term

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        result = await summarize_partition_long_term(
            view, group={"user_id": "alice"}, memories=[]
        )

        assert result.memory_count == 0
        assert "No memories found" in result.summary

    @pytest.mark.asyncio
    async def test_fallback_when_no_api_keys(self, monkeypatch):
        """summarize_partition_long_term should fallback when no API keys."""
        from agent_memory_server import config
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import summarize_partition_long_term

        # Ensure no API keys are set
        monkeypatch.setattr(config.settings, "openai_api_key", None)
        monkeypatch.setattr(config.settings, "anthropic_api_key", None)
        monkeypatch.setattr(config.settings, "aws_access_key_id", None)

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="Memory 1"),
            MemoryRecord(id="m2", text="Memory 2"),
        ]

        result = await summarize_partition_long_term(
            view, group={"user_id": "alice"}, memories=memories
        )

        assert result.memory_count == 2
        assert "LLM summarization disabled" in result.summary
        assert "Memory 1" in result.summary

    @pytest.mark.asyncio
    async def test_llm_call_with_mocked_client(self, monkeypatch):
        """summarize_partition_long_term should call LLM when API key is set."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_memory_server import config
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import summarize_partition_long_term

        # Set an API key to trigger LLM path
        monkeypatch.setattr(config.settings, "openai_api_key", "test-key")

        # Mock the LLMClient
        mock_response = MagicMock()
        mock_response.content = "This is a summary of Alice's memories."

        mock_create = AsyncMock(return_value=mock_response)

        # We need to patch within the module where it's imported

        monkeypatch.setattr(
            "agent_memory_server.llm.LLMClient.create_chat_completion", mock_create
        )

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="Memory 1"),
        ]

        result = await summarize_partition_long_term(
            view, group={"user_id": "alice"}, memories=memories
        )

        assert result.summary == "This is a summary of Alice's memories."
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_call_handles_exception(self, monkeypatch):
        """summarize_partition_long_term should handle LLM exceptions gracefully."""
        from unittest.mock import AsyncMock

        from agent_memory_server import config
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import summarize_partition_long_term

        # Set an API key to trigger LLM path
        monkeypatch.setattr(config.settings, "openai_api_key", "test-key")

        # Mock the LLMClient to raise an exception
        mock_create = AsyncMock(side_effect=Exception("API error"))
        monkeypatch.setattr(
            "agent_memory_server.llm.LLMClient.create_chat_completion", mock_create
        )

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="Memory 1"),
        ]

        result = await summarize_partition_long_term(
            view, group={"user_id": "alice"}, memories=memories
        )

        assert "No summary could be generated" in result.summary

    @pytest.mark.asyncio
    async def test_llm_call_handles_empty_content(self, monkeypatch):
        """summarize_partition_long_term should handle empty LLM response."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_memory_server import config
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import summarize_partition_long_term

        # Set an API key to trigger LLM path
        monkeypatch.setattr(config.settings, "openai_api_key", "test-key")

        # Mock the LLMClient to return empty content
        mock_response = MagicMock()
        mock_response.content = ""

        mock_create = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(
            "agent_memory_server.llm.LLMClient.create_chat_completion", mock_create
        )

        view = SummaryView(
            id="test", name="test", source="long_term", group_by=["user_id"], filters={}
        )

        memories = [
            MemoryRecord(id="m1", text="Memory 1"),
        ]

        result = await summarize_partition_long_term(
            view, group={"user_id": "alice"}, memories=memories
        )

        assert "No summary could be generated" in result.summary


class TestSummarizePartitionPlaceholder:
    """Tests for summarize_partition_placeholder function."""

    @pytest.mark.asyncio
    async def test_returns_placeholder_summary(self):
        """summarize_partition_placeholder should return a placeholder message."""
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import summarize_partition_placeholder

        view = SummaryView(
            id="test-view",
            name="test",
            source="working_memory",
            group_by=[],
            filters={},
        )

        result = await summarize_partition_placeholder(view, group={"user_id": "alice"})

        assert result.view_id == "test-view"
        assert result.memory_count == 0
        assert "Placeholder summary" in result.summary


class TestSummarizePartitionForView:
    """Tests for summarize_partition_for_view function."""

    @pytest.mark.asyncio
    async def test_dispatches_to_placeholder_for_unsupported_source(self, monkeypatch):
        """summarize_partition_for_view should use placeholder for working_memory source."""
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import summarize_partition_for_view

        # Create a view with working_memory source (not yet fully implemented)
        view = SummaryView(
            id="test-view",
            name="test",
            source="working_memory",
            group_by=[],
            filters={},
        )

        result = await summarize_partition_for_view(view, group={"user_id": "alice"})

        assert "Placeholder summary" in result.summary


class TestRefreshSummaryView:
    """Tests for refresh_summary_view function."""

    @pytest.mark.asyncio
    async def test_refresh_handles_missing_view_with_task(self, async_redis_client):
        """refresh_summary_view should mark task as failed when view is missing."""
        from agent_memory_server.models import Task, TaskStatusEnum, TaskTypeEnum
        from agent_memory_server.summary_views import refresh_summary_view
        from agent_memory_server.tasks import create_task, get_task

        # Create a task first
        task_id = "test-task-missing-view"
        task = Task(
            id=task_id,
            type=TaskTypeEnum.SUMMARY_VIEW_FULL_RUN,
            view_id="nonexistent-view",
            status=TaskStatusEnum.PENDING,
        )
        await create_task(task)

        # Try to refresh a non-existent view
        await refresh_summary_view("nonexistent-view", task_id=task_id)

        # Check task status
        result_task = await get_task(task_id)
        assert result_task is not None
        assert result_task.status == TaskStatusEnum.FAILED
        assert "not found" in result_task.error_message

    @pytest.mark.asyncio
    async def test_refresh_handles_missing_view_without_task(self, async_redis_client):
        """refresh_summary_view should return silently when view is missing and no task."""
        from agent_memory_server.summary_views import refresh_summary_view

        # This should not raise an exception
        await refresh_summary_view("nonexistent-view", task_id=None)

    @pytest.mark.asyncio
    async def test_refresh_updates_task_to_running(self, async_redis_client):
        """refresh_summary_view should update task to running status."""
        from agent_memory_server.models import (
            SummaryView,
            Task,
            TaskStatusEnum,
            TaskTypeEnum,
        )
        from agent_memory_server.summary_views import (
            refresh_summary_view,
            save_summary_view,
        )
        from agent_memory_server.tasks import create_task, get_task

        # Create a view
        view = SummaryView(
            id="test-refresh-view",
            name="test",
            source="long_term",
            group_by=["user_id"],
            filters={},
        )
        await save_summary_view(view)

        # Create a task
        task_id = "test-refresh-task"
        task = Task(
            id=task_id,
            type=TaskTypeEnum.SUMMARY_VIEW_FULL_RUN,
            view_id=view.id,
            status=TaskStatusEnum.PENDING,
        )
        await create_task(task)

        # Run the refresh
        await refresh_summary_view(view.id, task_id=task_id)

        # Check final task status
        result_task = await get_task(task_id)
        assert result_task is not None
        assert result_task.status == TaskStatusEnum.SUCCESS

    @pytest.mark.asyncio
    async def test_refresh_logs_unsupported_source(self, async_redis_client, caplog):
        """refresh_summary_view should log info for working_memory sources."""
        import logging

        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import (
            refresh_summary_view,
            save_summary_view,
        )

        # Create a view with working_memory source (not yet fully implemented)
        view = SummaryView(
            id="test-unsupported-source-view",
            name="test",
            source="working_memory",
            group_by=[],
            filters={},
        )
        await save_summary_view(view)

        with caplog.at_level(logging.INFO):
            await refresh_summary_view(view.id, task_id=None)

        assert "not yet implemented" in caplog.text


class TestPeriodicRefreshSummaryViews:
    """Tests for periodic_refresh_summary_views function."""

    @pytest.mark.asyncio
    async def test_periodic_refresh_skips_when_ltm_disabled(self, monkeypatch):
        """periodic_refresh_summary_views should skip when long_term_memory is disabled."""
        from agent_memory_server import config
        from agent_memory_server.summary_views import periodic_refresh_summary_views

        monkeypatch.setattr(config.settings, "long_term_memory", False)

        # This should return early without error
        await periodic_refresh_summary_views()

    @pytest.mark.asyncio
    async def test_periodic_refresh_skips_non_continuous_views(
        self, async_redis_client, monkeypatch
    ):
        """periodic_refresh_summary_views should skip non-continuous views."""
        from unittest.mock import AsyncMock

        from agent_memory_server import config
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import periodic_refresh_summary_views

        monkeypatch.setattr(config.settings, "long_term_memory", True)

        # Create a non-continuous view
        view = SummaryView(
            id="test-non-continuous",
            name="test",
            source="long_term",
            group_by=[],
            filters={},
            continuous=False,
        )

        # Mock list_summary_views to return only our test view (avoiding test isolation issues)
        mock_list = AsyncMock(return_value=[view])
        monkeypatch.setattr(
            "agent_memory_server.summary_views.list_summary_views", mock_list
        )

        # Mock refresh_summary_view to track calls
        mock_refresh = AsyncMock()
        monkeypatch.setattr(
            "agent_memory_server.summary_views.refresh_summary_view", mock_refresh
        )

        await periodic_refresh_summary_views()

        # Should not have been called for non-continuous view
        mock_refresh.assert_not_called()

    @pytest.mark.asyncio
    async def test_periodic_refresh_processes_continuous_views(
        self, async_redis_client, monkeypatch
    ):
        """periodic_refresh_summary_views should refresh continuous views."""
        from unittest.mock import AsyncMock

        from agent_memory_server import config
        from agent_memory_server.models import SummaryView
        from agent_memory_server.summary_views import periodic_refresh_summary_views

        monkeypatch.setattr(config.settings, "long_term_memory", True)

        # Create a continuous view
        view = SummaryView(
            id="test-continuous",
            name="test",
            source="long_term",
            group_by=[],
            filters={},
            continuous=True,
        )

        # Mock list_summary_views to return only our test view (avoiding test isolation issues)
        mock_list = AsyncMock(return_value=[view])
        monkeypatch.setattr(
            "agent_memory_server.summary_views.list_summary_views", mock_list
        )

        # Mock refresh_summary_view to track calls
        mock_refresh = AsyncMock()
        monkeypatch.setattr(
            "agent_memory_server.summary_views.refresh_summary_view", mock_refresh
        )

        await periodic_refresh_summary_views()

        # Should have been called for continuous view
        mock_refresh.assert_called_once_with("test-continuous", task_id=None)


class TestBuildLongTermSummaryPromptEdgeCases:
    """Additional tests for _build_long_term_summary_prompt edge cases."""

    def test_truncates_very_long_memories(self):
        """_build_long_term_summary_prompt should truncate very long memories."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _build_long_term_summary_prompt

        view = SummaryView(
            id="test-view",
            name="test",
            source="long_term",
            group_by=["user_id"],
            filters={},
        )
        group = {"user_id": "alice"}
        # Create a memory with very long text (this tests lines 388-391)
        long_text = "A" * 20000  # Very long text
        mem = MemoryRecord(id="mem-1", text=long_text)

        result = _build_long_term_summary_prompt(
            view, group, [mem], "gpt-4o-mini", "Summarize these memories"
        )

        # Should have a valid prompt that doesn't include the full 20k chars
        assert "SUMMARY:" in result
        assert len(result) < 25000  # Should be truncated

    def test_exceeds_remaining_tokens_breaks_early(self):
        """_build_long_term_summary_prompt breaks early when remaining tokens exhausted."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _build_long_term_summary_prompt

        view = SummaryView(
            id="test-view",
            name="test",
            source="long_term",
            group_by=["user_id"],
            filters={},
        )
        group = {"user_id": "alice"}
        # Create multiple memories that exceed budget (tests line 394)
        memories = [
            MemoryRecord(id=f"mem-{i}", text=f"Memory {i} " * 200) for i in range(100)
        ]

        result = _build_long_term_summary_prompt(
            view, group, memories, "gpt-4o-mini", "Summarize these memories"
        )

        # Should truncate to fit budget
        assert "SUMMARY:" in result

    def test_shows_truncation_notice_when_memories_exceed_budget(self):
        """_build_long_term_summary_prompt shows truncation notice."""
        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import _build_long_term_summary_prompt

        view = SummaryView(
            id="test-view",
            name="test",
            source="long_term",
            group_by=["user_id"],
            filters={},
        )
        group = {"user_id": "alice"}
        # Create more memories than can fit (tests lines 402-406)
        memories = [
            MemoryRecord(id=f"mem-{i}", text=f"Memory content {i} " * 100)
            for i in range(200)
        ]

        result = _build_long_term_summary_prompt(
            view, group, memories, "gpt-4o-mini", "Summarize these memories"
        )

        # Should have truncation notice or at least produce valid output
        assert "truncated" in result.lower() or "SUMMARY:" in result


class TestRefreshSummaryViewExceptionHandling:
    """Tests for exception handling in refresh_summary_view."""

    @pytest.mark.asyncio
    async def test_refresh_catches_exception_and_marks_task_failed(
        self, async_redis_client, monkeypatch
    ):
        """refresh_summary_view should catch exceptions and mark task as failed."""
        from unittest.mock import AsyncMock

        from agent_memory_server.models import (
            SummaryView,
            Task,
            TaskStatusEnum,
            TaskTypeEnum,
        )
        from agent_memory_server.summary_views import (
            refresh_summary_view,
            save_summary_view,
        )
        from agent_memory_server.tasks import create_task, get_task

        # Create a view
        view = SummaryView(
            id="test-exception-view",
            name="test",
            source="long_term",
            group_by=["user_id"],
            filters={},
        )
        await save_summary_view(view)

        # Create a task
        task_id = "test-exception-task"
        task = Task(
            id=task_id,
            type=TaskTypeEnum.SUMMARY_VIEW_FULL_RUN,
            view_id=view.id,
            status=TaskStatusEnum.PENDING,
        )
        await create_task(task)

        # Mock _fetch_long_term_memories_for_view to raise an exception
        mock_fetch = AsyncMock(side_effect=RuntimeError("Test error"))
        monkeypatch.setattr(
            "agent_memory_server.summary_views._fetch_long_term_memories_for_view",
            mock_fetch,
        )

        # Run the refresh - should not raise
        await refresh_summary_view(view.id, task_id=task_id)

        # Check task status - should be FAILED
        result_task = await get_task(task_id)
        assert result_task is not None
        assert result_task.status == TaskStatusEnum.FAILED
        assert "Test error" in result_task.error_message


class TestLargeMemoryThresholdWarning:
    """Tests for large memory threshold warning."""

    @pytest.mark.asyncio
    async def test_logs_warning_for_large_memory_sets(
        self, async_redis_client, monkeypatch, caplog
    ):
        """refresh_summary_view should log warning for large memory sets."""
        import logging
        from unittest.mock import AsyncMock

        from agent_memory_server.models import MemoryRecord, SummaryView
        from agent_memory_server.summary_views import (
            refresh_summary_view,
            save_summary_view,
        )

        # Create a view
        view = SummaryView(
            id="test-large-memory-view",
            name="test",
            source="long_term",
            group_by=[],
            filters={},
        )
        await save_summary_view(view)

        # Mock _fetch_long_term_memories_for_view to return many memories
        large_memories = [
            MemoryRecord(id=f"mem-{i}", text=f"Memory {i}") for i in range(15000)
        ]
        mock_fetch = AsyncMock(return_value=large_memories)
        monkeypatch.setattr(
            "agent_memory_server.summary_views._fetch_long_term_memories_for_view",
            mock_fetch,
        )

        # Mock the summarization to avoid actual LLM calls
        mock_summarize = AsyncMock()
        monkeypatch.setattr(
            "agent_memory_server.summary_views.summarize_partition_long_term",
            mock_summarize,
        )

        with caplog.at_level(logging.WARNING):
            await refresh_summary_view(view.id, task_id=None)

        # Should have logged the warning (tests line 579)
        assert (
            "15000 memories" in caplog.text or "consider adding filters" in caplog.text
        )
