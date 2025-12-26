"""Test that --no-worker mode actually runs background tasks inline."""

import pytest
from httpx import ASGITransport, AsyncClient

from agent_memory_server.config import settings
from agent_memory_server.main import app


@pytest.mark.asyncio
async def test_no_worker_mode_runs_tasks_inline(mock_vectorstore_adapter):
    """Test that background tasks run inline when use_docket=False."""
    # Set to no-worker mode
    original_use_docket = settings.use_docket
    settings.use_docket = False

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Create a long-term memory
            payload = {
                "memories": [
                    {
                        "id": "test-no-worker-123",
                        "text": "Test memory for no-worker mode",
                        "memory_type": "semantic",
                    }
                ]
            }

            response = await client.post("/v1/long-term-memory/", json=payload)
            assert response.status_code == 200

            # In no-worker mode, the task should have been executed inline
            # So we should be able to search for it immediately
            search_payload = {
                "text": "Test memory for no-worker mode",
                "limit": 10,
            }

            search_response = await client.post(
                "/v1/long-term-memory/search", json=search_payload
            )
            assert search_response.status_code == 200

            results = search_response.json()
            # The memory should be found because it was indexed inline
            assert len(results["memories"]) > 0
            found = any(m["id"] == "test-no-worker-123" for m in results["memories"])
            assert found, "Memory should have been indexed inline in no-worker mode"

    finally:
        # Restore original setting
        settings.use_docket = original_use_docket


@pytest.mark.asyncio
async def test_worker_mode_queues_tasks(mock_vectorstore_adapter):
    """Test that background tasks are configured to use Docket when use_docket=True.

    Note: This test verifies the API accepts the request and returns successfully
    when worker mode is enabled. With the mock setup, tasks may still complete
    synchronously. Testing true async worker behavior requires a running Docket worker.
    """
    # Set to worker mode
    original_use_docket = settings.use_docket
    settings.use_docket = True

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Create a long-term memory
            payload = {
                "memories": [
                    {
                        "id": "test-worker-456",
                        "text": "Test memory for worker mode",
                        "memory_type": "semantic",
                    }
                ]
            }

            response = await client.post("/v1/long-term-memory/", json=payload)
            # The main assertion is that the API responds successfully
            # when worker mode is enabled
            assert response.status_code == 200

            # In a real scenario with separate workers, the memory would be queued.
            # With mocks, it may complete synchronously - that's acceptable for this test.
            search_payload = {
                "text": "Test memory for worker mode",
                "limit": 10,
            }

            search_response = await client.post(
                "/v1/long-term-memory/search", json=search_payload
            )
            assert search_response.status_code == 200

    finally:
        # Restore original setting
        settings.use_docket = original_use_docket
