"""
Tests for the extraction logic fixes in long_term_memory.py
"""

from unittest.mock import patch

import pytest

from agent_memory_server.long_term_memory import promote_working_memory_to_long_term
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
    async def test_extracted_memories_are_promoted(self, async_redis_client):
        """Test that extracted memories are actually promoted to long-term storage"""
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

            # Mock the extraction functions to return a test memory
            mock_extracted_memory = MemoryRecord(
                id="extracted-1",
                text="Extracted memory from conversation",
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                memory_type="episodic",  # Use valid enum value
            )

            with (
                patch(
                    "agent_memory_server.long_term_memory.should_extract_session_thread",
                    return_value=True,
                ),
                patch(
                    "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
                    return_value=[mock_extracted_memory],
                ),
            ):
                promoted_count = await promote_working_memory_to_long_term(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                # Should have promoted the extracted memory
                assert promoted_count == 1

                # Verify the working memory was updated with extraction status
                updated_wm = await get_working_memory(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                assert updated_wm is not None
                # Message should be marked as extracted
                assert updated_wm.messages[0].discrete_memory_extracted == "t"
                # Extracted memory should be in working memory (now with persisted_at set)
                assert len(updated_wm.memories) == 1
                assert updated_wm.memories[0].id == "extracted-1"
                assert updated_wm.memories[0].persisted_at is not None

        finally:
            settings.enable_discrete_memory_extraction = original_setting

    @pytest.mark.asyncio
    async def test_working_memory_updated_when_messages_marked_extracted(
        self, async_redis_client
    ):
        """Test that working memory is updated even when no memories are extracted but messages are marked"""
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

            # Mock extraction to return no memories but still trigger marking
            with (
                patch(
                    "agent_memory_server.long_term_memory.should_extract_session_thread",
                    return_value=True,
                ),
                patch(
                    "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
                    return_value=[],
                ),
            ):  # No extracted memories
                promoted_count = await promote_working_memory_to_long_term(
                    session_id=session_id,
                    user_id=user_id,
                    namespace=namespace,
                    redis_client=async_redis_client,
                )

                # Should have promoted 0 memories
                assert promoted_count == 0

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
