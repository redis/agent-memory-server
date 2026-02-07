"""
Tests for the extraction logic fixes in long_term_memory.py

With trailing-edge debounce, extraction no longer happens synchronously during
promote_working_memory_to_long_term. Instead, extraction is scheduled to run
after a period of inactivity via run_delayed_extraction.
"""

from unittest.mock import patch

import pytest

from agent_memory_server.long_term_memory import (
    promote_working_memory_to_long_term,
    run_delayed_extraction,
)
from agent_memory_server.models import MemoryMessage, MemoryRecord, WorkingMemory
from agent_memory_server.working_memory import get_working_memory, set_working_memory


class TestExtractionLogicFixes:
    """Test the fixes for extraction logic issues"""

    @pytest.mark.asyncio
    async def test_extracted_memories_variable_always_defined(self, async_redis_client):
        """Test that extracted_memories variable is always defined, even when extraction is disabled"""
        from agent_memory_server.config import settings

        # Disable extraction
        original_setting = settings.enable_discrete_memory_extraction
        settings.enable_discrete_memory_extraction = False

        try:
            session_id = "test-extraction-disabled"
            user_id = "test-user"
            namespace = "test"

            # Create working memory with unextracted messages
            messages = [
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="Test message",
                    discrete_memory_extracted="f",  # Unextracted
                ),
            ]

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=messages,
                memories=[],  # No existing memories
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # This should not raise a NameError for undefined extracted_memories
            promoted_count = await promote_working_memory_to_long_term(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
            )

            # Should complete successfully
            assert promoted_count == 0  # No memories to promote

        finally:
            settings.enable_discrete_memory_extraction = original_setting

    @pytest.mark.asyncio
    async def test_trailing_extraction_schedules_correctly(
        self, async_redis_client, mock_memory_vector_db
    ):
        """Test that promote_working_memory_to_long_term schedules trailing extraction"""
        from agent_memory_server.config import settings

        # Enable extraction
        original_setting = settings.enable_discrete_memory_extraction
        settings.enable_discrete_memory_extraction = True

        try:
            session_id = "test-extraction-promotion"
            user_id = "test-user"
            namespace = "test"

            # Create working memory with unextracted messages
            messages = [
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="Test message for extraction",
                    discrete_memory_extracted="f",  # Unextracted
                ),
            ]

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=messages,
                memories=[],  # No existing memories
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # With trailing-edge debounce, promote_working_memory_to_long_term
            # schedules extraction for later, not synchronously
            with patch(
                "agent_memory_server.long_term_memory.schedule_trailing_extraction"
            ) as mock_schedule:
                await promote_working_memory_to_long_term(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                # Should have scheduled trailing extraction
                mock_schedule.assert_called_once()
                call_kwargs = mock_schedule.call_args[1]
                assert call_kwargs["session_id"] == session_id
                assert call_kwargs["user_id"] == user_id
                assert call_kwargs["namespace"] == namespace

        finally:
            settings.enable_discrete_memory_extraction = original_setting

    @pytest.mark.asyncio
    async def test_run_delayed_extraction_extracts_and_indexes(
        self, async_redis_client, mock_memory_vector_db
    ):
        """Test that run_delayed_extraction properly extracts and indexes memories"""
        from agent_memory_server.config import settings

        # Enable extraction
        original_setting = settings.enable_discrete_memory_extraction
        settings.enable_discrete_memory_extraction = True

        try:
            session_id = "test-delayed-extraction"
            user_id = "test-user"
            namespace = "test"

            # Create working memory with unextracted messages
            messages = [
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="Test message for extraction",
                    discrete_memory_extracted="f",  # Unextracted
                ),
            ]

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=messages,
                memories=[],  # No existing memories
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # Mock the extraction to return a test memory
            mock_extracted_memory = MemoryRecord(
                id="extracted-1",
                text="Extracted memory from conversation",
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                memory_type="episodic",
            )

            with (
                patch(
                    "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
                    return_value=[mock_extracted_memory],
                ),
                patch(
                    "agent_memory_server.long_term_memory.index_long_term_memories"
                ) as mock_index,
            ):
                # Run delayed extraction directly
                count = await run_delayed_extraction(
                    session_id=session_id,
                    namespace=namespace,
                    user_id=user_id,
                    scheduled_timestamp=None,  # No timestamp check
                )

                # Should have extracted 1 memory
                assert count == 1

                # Should have indexed the memories
                mock_index.assert_called_once()
                indexed_memories = mock_index.call_args[0][0]
                assert len(indexed_memories) == 1
                assert indexed_memories[0].id == "extracted-1"

                # Verify working memory was updated - messages marked as extracted
                updated_wm = await get_working_memory(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                assert updated_wm is not None
                assert updated_wm.messages[0].discrete_memory_extracted == "t"

        finally:
            settings.enable_discrete_memory_extraction = original_setting

    @pytest.mark.asyncio
    async def test_delayed_extraction_marks_messages_even_with_no_memories(
        self, async_redis_client
    ):
        """Test that run_delayed_extraction marks messages even when no memories are extracted"""
        from agent_memory_server.config import settings

        # Enable extraction
        original_setting = settings.enable_discrete_memory_extraction
        settings.enable_discrete_memory_extraction = True

        try:
            session_id = "test-extraction-marking"
            user_id = "test-user"
            namespace = "test"

            # Create working memory with unextracted messages
            messages = [
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="Test message for marking",
                    discrete_memory_extracted="f",  # Unextracted
                ),
            ]

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=messages,
                memories=[],  # No existing memories
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # Mock extraction to return no memories
            with patch(
                "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
                return_value=[],
            ):
                # Run delayed extraction directly
                count = await run_delayed_extraction(
                    session_id=session_id,
                    namespace=namespace,
                    user_id=user_id,
                    scheduled_timestamp=None,
                )

                # Should have extracted 0 memories
                assert count == 0

                # But working memory should still be updated with extraction status
                updated_wm = await get_working_memory(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                assert updated_wm is not None
                # Message should be marked as extracted even though no memories were extracted
                assert updated_wm.messages[0].discrete_memory_extracted == "t"

        finally:
            settings.enable_discrete_memory_extraction = original_setting

    @pytest.mark.asyncio
    async def test_no_extraction_when_debounced(self, async_redis_client):
        """Test that extraction is skipped when debounced and extracted_memories is still defined"""
        from agent_memory_server.config import settings

        # Enable extraction
        original_setting = settings.enable_discrete_memory_extraction
        settings.enable_discrete_memory_extraction = True

        try:
            session_id = "test-extraction-debounced"
            user_id = "test-user"
            namespace = "test"

            # Create working memory with unextracted messages
            messages = [
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="Test message for debouncing",
                    discrete_memory_extracted="f",  # Unextracted
                ),
            ]

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=messages,
                memories=[],
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # Mock extraction to be debounced (should_extract returns False)
            with patch(
                "agent_memory_server.long_term_memory.should_extract_session_thread",
                return_value=False,
            ):
                promoted_count = await promote_working_memory_to_long_term(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                # Should complete without error (extracted_memories should be defined as empty list)
                assert promoted_count == 0

                # Working memory should not be updated since nothing changed
                updated_wm = await get_working_memory(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                assert updated_wm is not None
                # Message should still be marked as unextracted
                assert updated_wm.messages[0].discrete_memory_extracted == "f"

        finally:
            settings.enable_discrete_memory_extraction = original_setting
