"""Tests for working memory functionality."""

import pytest
from pydantic import ValidationError

from agent_memory_server.models import MemoryRecord, MemoryTypeEnum, WorkingMemory
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
