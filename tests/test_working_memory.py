"""Tests for working memory functionality."""

import asyncio

import pytest
from pydantic import ValidationError

from agent_memory_server.models import MemoryRecord, MemoryTypeEnum, WorkingMemory
from agent_memory_server.utils.keys import Keys
from agent_memory_server.working_memory import (
    delete_working_memory,
    get_working_memory,
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
