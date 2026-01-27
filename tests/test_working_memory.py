"""Tests for working memory functionality."""

import asyncio

import pytest
from pydantic import ValidationError

from agent_memory_server.models import MemoryRecord, MemoryTypeEnum, WorkingMemory
from agent_memory_server.utils.keys import Keys
from agent_memory_server.working_memory import (
    delete_working_memory,
    get_working_memory,
    list_sessions,
    set_working_memory,
)


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_set_and_get_working_memory(self, async_redis_client):
        """Test setting and getting working memory"""
        session_id = "test-session"
        namespace = "test-namespace"

        # Create test memory records with id
        memories = [
            MemoryRecord(
                text="User prefers dark mode",
                id="client-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
                user_id="user123",
            ),
            MemoryRecord(
                text="User is working on a Python project",
                id="client-2",
                memory_type=MemoryTypeEnum.EPISODIC,
                user_id="user123",
            ),
        ]

        # Create working memory
        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
            ttl_seconds=1800,  # 30 minutes
        )

        # Set working memory
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Get working memory
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert retrieved_mem is not None
        assert retrieved_mem.session_id == session_id
        assert retrieved_mem.namespace == namespace
        assert len(retrieved_mem.memories) == 2
        assert retrieved_mem.memories[0].text == "User prefers dark mode"
        assert retrieved_mem.memories[0].id == "client-1"
        assert retrieved_mem.memories[1].text == "User is working on a Python project"
        assert retrieved_mem.memories[1].id == "client-2"
        assert retrieved_mem.ttl_seconds == 1800  # Verify TTL is preserved

    @pytest.mark.asyncio
    async def test_get_nonexistent_working_memory(self, async_redis_client):
        """Test getting working memory that doesn't exist"""
        result = await get_working_memory(
            session_id="nonexistent",
            namespace="test-namespace",
            redis_client=async_redis_client,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_working_memory(self, async_redis_client):
        """Test deleting working memory"""
        session_id = "test-session"
        namespace = "test-namespace"

        # Create and set working memory
        memories = [
            MemoryRecord(
                text="Test memory",
                id="client-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
        )

        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Verify it exists
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert retrieved_mem is not None

        # Delete it
        await delete_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        # Verify it's gone
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert retrieved_mem is None

    @pytest.mark.asyncio
    async def test_working_memory_validation(self, async_redis_client):
        """Test that working memory validates id requirement"""
        session_id = "test-session"

        # Test that creating MemoryRecord without id raises a validation error
        with pytest.raises(ValidationError, match="Field required"):
            MemoryRecord(  # type: ignore[call-arg]
                text="Memory without id",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )

        # Test that creating working memory with a valid memory record works
        memories = [
            MemoryRecord(
                id="test-memory-1",  # Add required id field
                text="Memory with id",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
        )

        # Should work without error
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Verify it was stored
        retrieved = await get_working_memory(
            session_id=session_id,
            redis_client=async_redis_client,
        )
        assert retrieved is not None
        assert len(retrieved.memories) == 1
        assert retrieved.memories[0].id == "test-memory-1"

    @pytest.mark.asyncio
    async def test_working_memory_ttl_none(self, async_redis_client):
        """Test working memory without TTL (persistent)"""
        session_id = "test-session-no-ttl"
        namespace = "test-namespace"

        memories = [
            MemoryRecord(
                text="Persistent memory",
                id="persistent-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
            ttl_seconds=None,  # No TTL - should be persistent
        )

        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Get working memory and verify TTL is None
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert retrieved_mem is not None
        assert retrieved_mem.ttl_seconds is None

        # Verify the Redis key has no TTL set (-1 means no TTL)
        key = Keys.working_memory_key(
            session_id=session_id,
            namespace=namespace,
        )
        ttl = await async_redis_client.ttl(key)
        assert ttl == -1  # No TTL set

    @pytest.mark.asyncio
    async def test_working_memory_ttl_set(self, async_redis_client):
        """Test working memory with TTL set"""
        session_id = "test-session-with-ttl"
        namespace = "test-namespace"

        memories = [
            MemoryRecord(
                text="Memory with TTL",
                id="ttl-memory-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        ttl_seconds = 60  # 1 minute
        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
            ttl_seconds=ttl_seconds,
        )

        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Get working memory and verify TTL is preserved
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert retrieved_mem is not None
        assert retrieved_mem.ttl_seconds == ttl_seconds

        # Verify the Redis key has TTL set (should be <= 60 seconds)
        key = Keys.working_memory_key(
            session_id=session_id,
            namespace=namespace,
        )
        ttl = await async_redis_client.ttl(key)
        assert 0 < ttl <= ttl_seconds

    @pytest.mark.asyncio
    async def test_working_memory_ttl_expiration(self, async_redis_client):
        """Test working memory expires after TTL"""
        session_id = "test-session-expire"
        namespace = "test-namespace"

        memories = [
            MemoryRecord(
                text="Memory that expires",
                id="expire-memory-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        ttl_seconds = 1  # 1 second
        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
            ttl_seconds=ttl_seconds,
        )

        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Verify it exists immediately
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert retrieved_mem is not None

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Verify it's gone after TTL
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert retrieved_mem is None

    @pytest.mark.asyncio
    async def test_working_memory_ttl_update_preserves_ttl(self, async_redis_client):
        """Test that updating working memory preserves TTL"""
        session_id = "test-session-update-ttl"
        namespace = "test-namespace"

        memories = [
            MemoryRecord(
                text="Original memory",
                id="original-memory-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        ttl_seconds = 120  # 2 minutes
        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
            ttl_seconds=ttl_seconds,
        )

        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Update the working memory
        working_mem.memories.append(
            MemoryRecord(
                text="Updated memory",
                id="updated-memory-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
        )

        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Get updated working memory and verify TTL is preserved
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert retrieved_mem is not None
        assert retrieved_mem.ttl_seconds == ttl_seconds
        assert len(retrieved_mem.memories) == 2

        # Verify the Redis key still has TTL set
        key = Keys.working_memory_key(
            session_id=session_id,
            namespace=namespace,
        )
        ttl = await async_redis_client.ttl(key)
        assert 0 < ttl <= ttl_seconds

    @pytest.mark.asyncio
    async def test_backward_compatibility_string_to_json_migration(
        self, async_redis_client
    ):
        """Test that old string-format working memory is migrated to JSON format on read."""
        import json

        from agent_memory_server.working_memory import (
            check_and_set_migration_status,
            is_migration_complete,
            reset_migration_status,
        )

        # Reset migration status to ensure lazy migration is active
        await reset_migration_status(async_redis_client)
        assert not await is_migration_complete(async_redis_client)

        session_id = "test-migration-session"
        namespace = "test-namespace"

        # Create old-format data (stringified JSON)
        old_format_data = {
            "messages": [
                {
                    "id": "msg-1",
                    "role": "user",
                    "content": "Hello",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            ],
            "memories": [
                {
                    "id": "mem-1",
                    "text": "User prefers dark mode",
                    "memory_type": "semantic",
                }
            ],
            "context": None,
            "user_id": "user123",
            "tokens": 10,
            "session_id": session_id,
            "namespace": namespace,
            "ttl_seconds": None,
            "data": {},
            "long_term_memory_strategy": {"strategy": "discrete"},
            "last_accessed": 1704067200,
            "created_at": 1704067200,
            "updated_at": 1704067200,
        }

        # Store as old string format directly
        key = Keys.working_memory_key(session_id=session_id, namespace=namespace)
        await async_redis_client.set(key, json.dumps(old_format_data))

        # Verify it's stored as string (not JSON)
        key_type = await async_redis_client.type(key)
        # Redis returns bytes, decode if needed
        if isinstance(key_type, bytes):
            key_type = key_type.decode("utf-8")
        assert key_type == "string"

        # Run check to set up the counter (finds 1 string key)
        await check_and_set_migration_status(async_redis_client)
        assert not await is_migration_complete(async_redis_client)

        # Now read using get_working_memory - should trigger migration
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        # Verify data was retrieved correctly
        assert retrieved_mem is not None
        assert retrieved_mem.session_id == session_id
        assert retrieved_mem.namespace == namespace
        assert len(retrieved_mem.messages) == 1
        assert retrieved_mem.messages[0].role == "user"
        assert retrieved_mem.messages[0].content == "Hello"
        assert len(retrieved_mem.memories) == 1
        assert retrieved_mem.memories[0].text == "User prefers dark mode"

        # Verify the key was migrated to JSON format
        key_type_after = await async_redis_client.type(key)
        if isinstance(key_type_after, bytes):
            key_type_after = key_type_after.decode("utf-8")
        assert key_type_after == "ReJSON-RL"

        # Migration auto-completes when counter reaches 0 (only 1 key, now migrated)
        assert await is_migration_complete(async_redis_client)

        # Verify we can read it again (now from JSON format, using fast path)
        retrieved_again = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert retrieved_again is not None
        assert retrieved_again.session_id == session_id

    @pytest.mark.asyncio
    async def test_migration_preserves_ttl(self, async_redis_client):
        """Test that TTL is preserved when migrating from string to JSON format."""
        import json

        from agent_memory_server.working_memory import reset_migration_status

        # Reset migration status to ensure lazy migration is active
        await reset_migration_status(async_redis_client)

        session_id = "test-ttl-migration-session"
        namespace = "test-namespace"
        ttl_seconds = 3600  # 1 hour

        # Create old-format data with TTL
        old_format_data = {
            "messages": [],
            "memories": [],
            "session_id": session_id,
            "namespace": namespace,
            "context": None,
            "user_id": None,
            "tokens": 0,
            "ttl_seconds": ttl_seconds,
            "data": {},
            "long_term_memory_strategy": {"strategy": "discrete"},
            "last_accessed": 1704067200,
            "created_at": 1704067200,
            "updated_at": 1704067200,
        }

        # Store as old string format with TTL
        key = Keys.working_memory_key(session_id=session_id, namespace=namespace)
        await async_redis_client.set(key, json.dumps(old_format_data), ex=ttl_seconds)

        # Verify TTL is set
        ttl_before = await async_redis_client.ttl(key)
        assert ttl_before > 0
        assert ttl_before <= ttl_seconds

        # Trigger migration by reading
        retrieved_mem = await get_working_memory(
            session_id=session_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )
        assert retrieved_mem is not None

        # Verify key was migrated to JSON
        key_type = await async_redis_client.type(key)
        if isinstance(key_type, bytes):
            key_type = key_type.decode("utf-8")
        assert key_type == "ReJSON-RL"

        # Verify TTL was preserved
        ttl_after = await async_redis_client.ttl(key)
        assert ttl_after > 0
        # TTL should be close to original (within a few seconds of test execution)
        assert ttl_after <= ttl_seconds
        assert ttl_after >= ttl_before - 5  # Allow 5 seconds for test execution

    @pytest.mark.asyncio
    async def test_check_and_set_migration_status_with_no_keys(
        self, async_redis_client
    ):
        """Test migration status check when no working memory keys exist."""
        from agent_memory_server.working_memory import (
            check_and_set_migration_status,
            is_migration_complete,
            reset_migration_status,
        )

        # Reset to ensure clean state
        await reset_migration_status(async_redis_client)
        assert not await is_migration_complete(async_redis_client)

        # Check status with no keys - should mark as migrated (nothing to migrate)
        result = await check_and_set_migration_status(async_redis_client)
        assert result is True
        assert await is_migration_complete(async_redis_client)

    @pytest.mark.asyncio
    async def test_check_and_set_migration_status_with_json_keys_only(
        self, async_redis_client
    ):
        """Test migration status check when only JSON keys exist."""
        from agent_memory_server.working_memory import (
            check_and_set_migration_status,
            is_migration_complete,
            reset_migration_status,
        )

        # Reset to ensure clean state
        await reset_migration_status(async_redis_client)

        # Create a JSON key
        session_id = "test-json-session"
        namespace = "test-namespace"
        memories = [
            MemoryRecord(
                text="Test memory",
                id="mem-1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]
        working_mem = WorkingMemory(
            memories=memories,
            session_id=session_id,
            namespace=namespace,
        )
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Check status - should mark as migrated (only JSON keys)
        result = await check_and_set_migration_status(async_redis_client)
        assert result is True
        assert await is_migration_complete(async_redis_client)

    @pytest.mark.asyncio
    async def test_check_and_set_migration_status_with_string_keys(
        self, async_redis_client
    ):
        """Test migration status check when string keys exist."""
        import json

        from agent_memory_server.working_memory import (
            check_and_set_migration_status,
            is_migration_complete,
            reset_migration_status,
        )

        # Reset to ensure clean state
        await reset_migration_status(async_redis_client)

        # Create an old-format string key
        key = Keys.working_memory_key(
            session_id="test-string-session", namespace="test-namespace"
        )
        old_data = {
            "messages": [],
            "memories": [],
            "session_id": "test-string-session",
            "namespace": "test-namespace",
        }
        await async_redis_client.set(key, json.dumps(old_data))

        # Check status - should NOT mark as migrated (string keys exist)
        result = await check_and_set_migration_status(async_redis_client)
        assert result is False
        assert not await is_migration_complete(async_redis_client)

    @pytest.mark.asyncio
    async def test_migration_status_set_by_set_migration_complete(
        self, async_redis_client
    ):
        """Test that set_migration_complete() marks migration as done."""
        import json

        from agent_memory_server.working_memory import (
            check_and_set_migration_status,
            get_remaining_string_keys,
            is_migration_complete,
            reset_migration_status,
        )

        # Reset to ensure clean state
        await reset_migration_status(async_redis_client)

        # Clean up any existing working_memory keys from other tests
        cursor = 0
        while True:
            cursor, keys = await async_redis_client.scan(
                cursor=cursor, match="working_memory:*", count=100
            )
            if keys:
                await async_redis_client.delete(*keys)
            if cursor == 0:
                break

        # Create two old-format string keys
        for i in range(2):
            key = Keys.working_memory_key(
                session_id=f"test-migrate-session-{i}", namespace="test-namespace"
            )
            old_data = {
                "messages": [],
                "memories": [],
                "session_id": f"test-migrate-session-{i}",
                "namespace": "test-namespace",
                "context": None,
                "user_id": None,
                "tokens": 0,
                "ttl_seconds": None,
                "data": {},
                "long_term_memory_strategy": {"strategy": "discrete"},
                "last_accessed": 1704067200,
                "created_at": 1704067200,
                "updated_at": 1704067200,
            }
            await async_redis_client.set(key, json.dumps(old_data))

        # Check status - should NOT be migrated, counter should be 2
        await check_and_set_migration_status(async_redis_client)
        assert not await is_migration_complete(async_redis_client)
        assert await get_remaining_string_keys(async_redis_client) == 2

        # Read first key - triggers migration, counter decrements to 1
        await get_working_memory(
            session_id="test-migrate-session-0",
            namespace="test-namespace",
            redis_client=async_redis_client,
        )
        assert not await is_migration_complete(async_redis_client)
        assert await get_remaining_string_keys(async_redis_client) == 1

        # Read second key - triggers migration, counter decrements to 0, auto-completes
        await get_working_memory(
            session_id="test-migrate-session-1",
            namespace="test-namespace",
            redis_client=async_redis_client,
        )
        # Now migration should be complete (auto-completed when counter hit 0)
        assert await is_migration_complete(async_redis_client)
        assert await get_remaining_string_keys(async_redis_client) == 0

    @pytest.mark.asyncio
    async def test_migration_skipped_when_env_variable_set(
        self, async_redis_client, monkeypatch
    ):
        """Test that migration check is skipped when WORKING_MEMORY_MIGRATION_COMPLETE=true."""
        import json

        from agent_memory_server import config
        from agent_memory_server.working_memory import (
            check_and_set_migration_status,
            is_migration_complete,
            reset_migration_status,
        )

        # Reset to ensure clean state
        await reset_migration_status(async_redis_client)

        # Create an old-format string key (would normally trigger lazy migration)
        key = Keys.working_memory_key(
            session_id="test-env-skip-session", namespace="test-namespace"
        )
        old_data = {
            "messages": [],
            "memories": [],
            "session_id": "test-env-skip-session",
            "namespace": "test-namespace",
        }
        await async_redis_client.set(key, json.dumps(old_data))

        # Set the env variable via settings
        monkeypatch.setattr(config.settings, "working_memory_migration_complete", True)

        # Check status - should skip scan and mark as complete immediately
        result = await check_and_set_migration_status(async_redis_client)
        assert result is True
        assert await is_migration_complete(async_redis_client)

        # Clean up
        await async_redis_client.delete(key)
        monkeypatch.setattr(config.settings, "working_memory_migration_complete", False)

    @pytest.mark.asyncio
    async def test_set_working_memory_indexes_session(self, async_redis_client):
        """Test that set_working_memory adds session to the sessions sorted set."""
        session_id = "test_session_index_123"
        namespace = "test_namespace_index"

        # Create working memory
        working_mem = WorkingMemory(
            session_id=session_id,
            namespace=namespace,
            messages=[],
            memories=[],
        )

        # Save working memory
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Verify session is in sorted set
        sessions_key = Keys.sessions_key(namespace=namespace)
        score = await async_redis_client.zscore(sessions_key, session_id)

        assert score is not None, "Session should be indexed in sorted set"
        assert score > 0, "Score should be a positive timestamp"

    @pytest.mark.asyncio
    async def test_delete_working_memory_removes_session_from_index(self, async_redis_client):
        """Test that delete_working_memory removes session from the sessions sorted set."""
        session_id = "test_session_to_delete"
        namespace = "test_namespace_delete"

        # Create and save working memory
        working_mem = WorkingMemory(
            session_id=session_id,
            namespace=namespace,
            messages=[],
            memories=[],
        )
        await set_working_memory(working_mem, redis_client=async_redis_client)

        # Verify session is indexed
        sessions_key = Keys.sessions_key(namespace=namespace)
        score_before = await async_redis_client.zscore(sessions_key, session_id)
        assert score_before is not None, "Session should be indexed before delete"

        # Delete working memory
        await delete_working_memory(session_id, namespace=namespace, redis_client=async_redis_client)

        # Verify session is removed from index
        score_after = await async_redis_client.zscore(sessions_key, session_id)
        assert score_after is None, "Session should be removed from index after delete"

    @pytest.mark.asyncio
    async def test_list_sessions_returns_indexed_sessions(self, async_redis_client):
        """Test that list_sessions returns sessions that were indexed by set_working_memory."""
        namespace = "list_test_namespace"
        session_ids = ["session_a", "session_b", "session_c"]

        # Create multiple sessions
        for session_id in session_ids:
            working_mem = WorkingMemory(
                session_id=session_id,
                namespace=namespace,
                messages=[],
                memories=[],
            )
            await set_working_memory(working_mem, redis_client=async_redis_client)

        # List sessions
        total, listed_sessions = await list_sessions(
            redis=async_redis_client,
            namespace=namespace,
            limit=10,
            offset=0,
        )

        assert total == 3, f"Expected 3 sessions, got {total}"
        assert set(listed_sessions) == set(session_ids), f"Expected {session_ids}, got {listed_sessions}"
