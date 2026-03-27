"""Tests for GitHub issue #235 fix: GET /working-memory returns 404 after PUT when
user_id/namespace are in body but omitted from GET query params.

The fix uses the working memory search index as a fallback to resolve the correct
Redis key when user_id/namespace are not provided on GET/DELETE.
"""

import pytest

from agent_memory_server.models import MemoryMessage, WorkingMemory
from agent_memory_server.utils.keys import Keys
from agent_memory_server.working_memory import (
    delete_working_memory,
    get_working_memory,
    set_working_memory,
)


class TestIssue235KeyResolution:
    """Verify that GET/DELETE resolve the correct key even when user_id/namespace
    are omitted from the lookup parameters."""

    def test_keys_differ_when_scoping_params_vary(self):
        """Confirm that the raw key function produces different keys -- this is
        intentional for multi-tenancy.  The fix is in the lookup layer, not here."""
        session_id = "test-session"
        key_full = Keys.working_memory_key(
            session_id=session_id, user_id="alice", namespace="demo"
        )
        key_bare = Keys.working_memory_key(session_id=session_id)

        assert key_full == "working_memory:demo:alice:test-session"
        assert key_bare == "working_memory:test-session"
        assert key_full != key_bare

    @pytest.mark.asyncio
    async def test_get_resolves_via_index_when_params_omitted(self, async_redis_client):
        """SET with user_id+namespace, GET without them -> still finds the session."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "issue-235-get-resolve"
        user_id = "alice"
        namespace = "demo"

        working_mem = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[MemoryMessage(role="user", content="Hello")],
            memories=[],
        )
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # GET without user_id/namespace -- should still find it via index fallback
        result = await get_working_memory(
            session_id=session_id,
            redis_client=async_redis_client,
        )

        assert result is not None, "Should find session via index fallback"
        assert result.session_id == session_id
        assert result.user_id == user_id
        assert result.namespace == namespace
        assert len(result.messages) == 1
        assert result.messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_get_with_partial_params_resolves(self, async_redis_client):
        """SET with user_id+namespace, GET with namespace only -> finds session."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "issue-235-partial"
        user_id = "bob"
        namespace = "staging"

        working_mem = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[MemoryMessage(role="user", content="Partial test")],
            memories=[],
        )
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # GET with namespace but without user_id
        result = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert (
            result is not None
        ), "Should find session with partial params via fallback"
        assert result.user_id == user_id
        assert result.namespace == namespace

    @pytest.mark.asyncio
    async def test_get_with_correct_params_still_uses_fast_path(
        self, async_redis_client
    ):
        """GET with matching params hits the direct key (no fallback needed)."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "issue-235-fast"
        user_id = "carol"
        namespace = "prod"

        working_mem = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[MemoryMessage(role="assistant", content="Fast path")],
            memories=[],
        )
        await set_working_memory(working_mem, redis_client=async_redis_client)

        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert result is not None
        assert result.session_id == session_id

    @pytest.mark.asyncio
    async def test_delete_resolves_via_index_when_params_omitted(
        self, async_redis_client
    ):
        """DELETE without user_id/namespace should still delete the correct session."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "issue-235-delete"
        user_id = "dave"
        namespace = "test"

        working_mem = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[MemoryMessage(role="user", content="To be deleted")],
            memories=[],
        )
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Verify it exists
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert result is not None

        # DELETE without user_id/namespace
        await delete_working_memory(
            session_id=session_id,
            redis_client=async_redis_client,
        )

        # Verify it's gone
        result_after = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert result_after is None, "Session should be deleted"

    @pytest.mark.asyncio
    async def test_api_put_body_params_get_without_query_params(self, client):
        """Full API round-trip: PUT with body params, GET without query params -> 200."""
        if client is None:
            pytest.skip("Client not available")

        session_id = "issue-235-api"

        # PUT with user_id and namespace in the request body
        put_response = await client.put(
            f"/v1/working-memory/{session_id}",
            json={
                "messages": [{"role": "user", "content": "Hello from API test"}],
                "user_id": "alice",
                "namespace": "demo",
            },
        )
        assert put_response.status_code == 200

        # GET without query params -- should now return 200 thanks to the fix
        get_response = await client.get(f"/v1/working-memory/{session_id}")
        assert get_response.status_code == 200, (
            f"Expected 200 after fix, got {get_response.status_code}: "
            f"{get_response.text}"
        )

        data = get_response.json()
        assert data["session_id"] == session_id
        assert data["user_id"] == "alice"
        assert data["namespace"] == "demo"
        assert len(data["messages"]) == 1

    @pytest.mark.asyncio
    async def test_api_delete_without_query_params(self, client):
        """API DELETE without query params should still delete the session."""
        if client is None:
            pytest.skip("Client not available")

        session_id = "issue-235-api-delete"

        # PUT with user_id and namespace in body
        put_response = await client.put(
            f"/v1/working-memory/{session_id}",
            json={
                "messages": [{"role": "user", "content": "Delete me"}],
                "user_id": "alice",
                "namespace": "demo",
            },
        )
        assert put_response.status_code == 200

        # DELETE without query params
        delete_response = await client.delete(f"/v1/working-memory/{session_id}")
        assert delete_response.status_code == 200

        # Verify it's gone (even with correct params)
        get_response = await client.get(
            f"/v1/working-memory/{session_id}",
            params={"user_id": "alice", "namespace": "demo"},
        )
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_nonexistent_session_still_returns_none(self, async_redis_client):
        """A session that never existed should still return None."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        result = await get_working_memory(
            session_id="truly-nonexistent-session-xyz",
            redis_client=async_redis_client,
        )
        assert result is None


class TestIssue235MultiTenantIsolation:
    """Verify that multi-tenant isolation is preserved and ambiguity is handled."""

    @pytest.mark.asyncio
    async def test_direct_lookups_return_correct_tenant(self, async_redis_client):
        """Two sessions with the same ID in different namespaces stay separate
        when looked up with full scoping parameters."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "shared-session-id"

        # Tenant A
        mem_a = WorkingMemory(
            session_id=session_id,
            user_id="user-a",
            namespace="tenant-a",
            messages=[MemoryMessage(role="user", content="Tenant A message")],
            memories=[],
        )
        await set_working_memory(mem_a, redis_client=async_redis_client)

        # Tenant B
        mem_b = WorkingMemory(
            session_id=session_id,
            user_id="user-b",
            namespace="tenant-b",
            messages=[MemoryMessage(role="user", content="Tenant B message")],
            memories=[],
        )
        await set_working_memory(mem_b, redis_client=async_redis_client)

        # Direct lookups return correct data
        result_a = await get_working_memory(
            session_id=session_id,
            user_id="user-a",
            namespace="tenant-a",
            redis_client=async_redis_client,
        )
        assert result_a is not None
        assert result_a.namespace == "tenant-a"
        assert result_a.messages[0].content == "Tenant A message"

        result_b = await get_working_memory(
            session_id=session_id,
            user_id="user-b",
            namespace="tenant-b",
            redis_client=async_redis_client,
        )
        assert result_b is not None
        assert result_b.namespace == "tenant-b"
        assert result_b.messages[0].content == "Tenant B message"

    @pytest.mark.asyncio
    async def test_namespace_filter_disambiguates_shared_session_id(
        self, async_redis_client
    ):
        """Partial scoping (namespace only) resolves the correct tenant."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "shared-session-ns"

        for tenant in ("tenant-x", "tenant-y"):
            mem = WorkingMemory(
                session_id=session_id,
                user_id=f"user-{tenant}",
                namespace=tenant,
                messages=[MemoryMessage(role="user", content=f"Message from {tenant}")],
                memories=[],
            )
            await set_working_memory(mem, redis_client=async_redis_client)

        result_x = await get_working_memory(
            session_id=session_id,
            namespace="tenant-x",
            redis_client=async_redis_client,
        )
        assert result_x is not None
        assert result_x.namespace == "tenant-x"

        result_y = await get_working_memory(
            session_id=session_id,
            namespace="tenant-y",
            redis_client=async_redis_client,
        )
        assert result_y is not None
        assert result_y.namespace == "tenant-y"

    @pytest.mark.asyncio
    async def test_ambiguous_lookup_returns_none(self, async_redis_client):
        """When multiple sessions share a session_id and no scoping params are
        provided, the fallback refuses to guess and returns None."""
        if async_redis_client is None:
            pytest.skip("Redis not available")

        session_id = "ambiguous-session-id"

        for i in range(2):
            mem = WorkingMemory(
                session_id=session_id,
                user_id=f"user-{i}",
                namespace=f"ns-{i}",
                messages=[MemoryMessage(role="user", content=f"Message {i}")],
                memories=[],
            )
            await set_working_memory(mem, redis_client=async_redis_client)

        # Bare lookup (no namespace/user_id) -- ambiguous, should return None
        result = await get_working_memory(
            session_id=session_id,
            redis_client=async_redis_client,
        )
        assert (
            result is None
        ), "Ambiguous session_id without scoping params should return None"

        # But each tenant can still be reached with correct scoping
        for i in range(2):
            result = await get_working_memory(
                session_id=session_id,
                user_id=f"user-{i}",
                namespace=f"ns-{i}",
                redis_client=async_redis_client,
            )
            assert result is not None
            assert result.namespace == f"ns-{i}"
