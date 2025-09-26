"""
Tests for working memory reconstruction from long-term memory.
"""

from datetime import UTC, datetime

import pytest

from agent_memory_server.config import settings
from agent_memory_server.long_term_memory import index_long_term_memories
from agent_memory_server.models import MemoryMessage, MemoryRecord, WorkingMemory
from agent_memory_server.working_memory import get_working_memory, set_working_memory


class TestWorkingMemoryReconstruction:
    """Test working memory reconstruction from long-term storage"""

    @pytest.mark.asyncio
    async def test_reconstruction_disabled_by_default(self, async_redis_client):
        """Test that reconstruction doesn't happen when index_all_messages_in_long_term_memory is False"""
        # Ensure the setting is disabled
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = False

        try:
            # Try to get non-existent working memory
            result = await get_working_memory(
                session_id="nonexistent-session",
                user_id="test-user",
                namespace="test",
                redis_client=async_redis_client,
            )

            # Should return None since reconstruction is disabled
            assert result is None

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting

    @pytest.mark.asyncio
    async def test_reconstruction_with_no_messages(self, async_redis_client):
        """Test reconstruction when no messages exist in long-term memory"""
        # Enable the setting
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            # Try to get non-existent working memory with no messages in long-term
            result = await get_working_memory(
                session_id="empty-session",
                user_id="test-user",
                namespace="test",
                redis_client=async_redis_client,
            )

            # Should return None since no messages found
            assert result is None

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting

    @pytest.mark.asyncio
    async def test_reconstruction_with_messages(self, async_redis_client):
        """Test successful reconstruction from messages in long-term memory"""
        # Enable the setting
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            session_id = "test-reconstruction-session"
            user_id = "test-user"
            namespace = "test"

            # Create message-type memory records (simulating what would be stored)
            now = datetime.now(UTC)
            message_memories = [
                MemoryRecord(
                    id="msg-1",
                    text="user: Hello, how are you?",
                    memory_type="message",
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    persisted_at=now,
                ),
                MemoryRecord(
                    id="msg-2",
                    text="assistant: I'm doing well, thank you for asking!",
                    memory_type="message",
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    persisted_at=now,
                ),
                MemoryRecord(
                    id="msg-3",
                    text="user: Can you help me with something?",
                    memory_type="message",
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    persisted_at=now,
                ),
            ]

            # Index these messages in long-term memory
            await index_long_term_memories(
                message_memories,
                redis_client=async_redis_client,
                deduplicate=False,
            )

            # Now try to get working memory - should reconstruct from long-term
            result = await get_working_memory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
            )

            # Should successfully reconstruct
            assert result is not None
            assert result.session_id == session_id
            assert result.user_id == user_id
            assert result.namespace == namespace
            assert len(result.messages) == 3

            # Check that all expected messages are present (order might vary)
            message_contents = [msg.content for msg in result.messages]
            message_ids = [msg.id for msg in result.messages]

            assert "Hello, how are you?" in message_contents
            assert "I'm doing well, thank you for asking!" in message_contents
            assert "Can you help me with something?" in message_contents

            assert "msg-1" in message_ids
            assert "msg-2" in message_ids
            assert "msg-3" in message_ids

            # All messages should have persisted_at set
            for msg in result.messages:
                assert msg.persisted_at is not None

            # Should have empty memories, context, and data
            assert result.memories == []
            assert result.context == ""
            assert result.data == {}

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting

    @pytest.mark.asyncio
    async def test_reconstruction_ignores_existing_working_memory(
        self, async_redis_client
    ):
        """Test that reconstruction doesn't happen if working memory already exists"""
        # Enable the setting
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            session_id = "existing-session"
            user_id = "test-user"
            namespace = "test"

            # Create existing working memory
            existing_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=[
                    MemoryMessage(
                        id="existing-msg",
                        role="user",
                        content="This is existing content",
                    )
                ],
            )

            # Store the existing working memory
            await set_working_memory(existing_memory, redis_client=async_redis_client)

            # Create different messages in long-term memory
            message_memories = [
                MemoryRecord(
                    id="lt-msg-1",
                    text="user: This is from long-term",
                    memory_type="message",
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    persisted_at=datetime.now(UTC),
                ),
            ]

            await index_long_term_memories(
                message_memories,
                redis_client=async_redis_client,
                deduplicate=False,
            )

            # Get working memory - should return existing, not reconstruct
            result = await get_working_memory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
            )

            # Should return existing working memory, not reconstructed
            assert result is not None
            assert len(result.messages) == 1
            assert result.messages[0].content == "This is existing content"
            assert result.messages[0].id == "existing-msg"

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting

    @pytest.mark.asyncio
    async def test_reconstruction_with_malformed_messages(self, async_redis_client):
        """Test reconstruction handles malformed message memories gracefully"""
        # Enable the setting
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            session_id = "malformed-session"
            user_id = "test-user"
            namespace = "test"

            # Create mix of valid and malformed message memories
            message_memories = [
                MemoryRecord(
                    id="valid-msg",
                    text="user: This is valid",
                    memory_type="message",
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    persisted_at=datetime.now(UTC),
                ),
                MemoryRecord(
                    id="malformed-msg",
                    text="This has no role separator",  # Missing ": "
                    memory_type="message",
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    persisted_at=datetime.now(UTC),
                ),
            ]

            await index_long_term_memories(
                message_memories,
                redis_client=async_redis_client,
                deduplicate=False,
            )

            # Should reconstruct with only valid messages
            result = await get_working_memory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
            )

            assert result is not None
            assert len(result.messages) == 1  # Only the valid message
            assert result.messages[0].content == "This is valid"

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting
