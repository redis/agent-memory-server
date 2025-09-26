"""
Tests for recent messages limit functionality.
"""

from datetime import UTC, datetime

import pytest

from agent_memory_server.models import MemoryMessage, WorkingMemory
from agent_memory_server.working_memory import get_working_memory, set_working_memory


class TestRecentMessagesLimit:
    """Test recent messages limit functionality"""

    @pytest.mark.asyncio
    async def test_recent_messages_limit_with_working_memory(self, async_redis_client):
        """Test recent messages limit with existing working memory using JSONPath"""
        session_id = "test-limit-session"
        user_id = "test-user"
        namespace = "test"

        # Create working memory with many messages
        messages = []
        for i in range(10):
            messages.append(
                MemoryMessage(
                    id=f"msg-{i}",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}: This is message number {i}",
                )
            )

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=messages,
        )

        # Store the working memory
        await set_working_memory(working_memory, redis_client=async_redis_client)

        # Test: Get with recent_messages_limit=3
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=3,
        )

        assert result is not None
        assert len(result.messages) == 3

        # Should get the last 3 messages (messages 7, 8, 9)
        assert result.messages[0].content == "Message 7: This is message number 7"
        assert result.messages[1].content == "Message 8: This is message number 8"
        assert result.messages[2].content == "Message 9: This is message number 9"

        # Test: Get with recent_messages_limit=5
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=5,
        )

        assert result is not None
        assert len(result.messages) == 5

        # Should get the last 5 messages (messages 5, 6, 7, 8, 9)
        assert result.messages[0].content == "Message 5: This is message number 5"
        assert result.messages[4].content == "Message 9: This is message number 9"

        # Test: Get without limit (should get all messages)
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
        )

        assert result is not None
        assert len(result.messages) == 10
        assert result.messages[0].content == "Message 0: This is message number 0"
        assert result.messages[9].content == "Message 9: This is message number 9"

    @pytest.mark.asyncio
    async def test_recent_messages_limit_larger_than_available(
        self, async_redis_client
    ):
        """Test recent messages limit when limit is larger than available messages"""
        session_id = "test-limit-large"
        user_id = "test-user"
        namespace = "test"

        # Create working memory with only 3 messages
        messages = []
        for i in range(3):
            messages.append(
                MemoryMessage(
                    id=f"msg-{i}",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                )
            )

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=messages,
        )

        await set_working_memory(working_memory, redis_client=async_redis_client)

        # Test: Get with recent_messages_limit=10 (larger than available)
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=10,
        )

        assert result is not None
        assert len(result.messages) == 3  # Should return all available messages
        assert result.messages[0].content == "Message 0"
        assert result.messages[2].content == "Message 2"

    @pytest.mark.asyncio
    async def test_recent_messages_limit_zero_and_negative(self, async_redis_client):
        """Test recent messages limit with zero and negative values"""
        session_id = "test-limit-edge"
        user_id = "test-user"
        namespace = "test"

        # Create working memory with messages
        messages = []
        for i in range(5):
            messages.append(
                MemoryMessage(
                    id=f"msg-{i}",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                )
            )

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=messages,
        )

        await set_working_memory(working_memory, redis_client=async_redis_client)

        # Test: Get with recent_messages_limit=0 (should return all messages)
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=0,
        )

        assert result is not None
        assert len(result.messages) == 5  # Should return all messages when limit is 0

        # Test: Get with recent_messages_limit=-1 (should return all messages)
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=-1,
        )

        assert result is not None
        assert (
            len(result.messages) == 5
        )  # Should return all messages when limit is negative

    @pytest.mark.asyncio
    async def test_recent_messages_limit_with_reconstruction(self, async_redis_client):
        """Test recent messages limit with reconstruction from long-term memory"""
        from agent_memory_server.config import settings
        from agent_memory_server.long_term_memory import index_long_term_memories
        from agent_memory_server.models import MemoryRecord

        # Enable message indexing
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            session_id = "test-limit-reconstruction"
            user_id = "test-user"
            namespace = "test"

            # Create message memories in long-term storage
            now = datetime.now(UTC)
            message_memories = []
            for msg_idx in range(8):
                message_memories.append(
                    MemoryRecord(
                        id=f"lt-msg-{msg_idx}",
                        text=f"{'user' if msg_idx % 2 == 0 else 'assistant'}: Long-term message {msg_idx}",
                        memory_type="message",
                        session_id=session_id,
                        user_id=user_id,
                        namespace=namespace,
                        persisted_at=now,
                    )
                )

            # Index messages in long-term memory
            await index_long_term_memories(
                message_memories,
                redis_client=async_redis_client,
                deduplicate=False,
            )

            # Test: Reconstruct with recent_messages_limit=3
            result = await get_working_memory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
                recent_messages_limit=3,
            )

            assert result is not None
            assert len(result.messages) <= 3  # Should limit to 3 messages

            # Messages should be in chronological order (oldest first)
            for _i, msg in enumerate(result.messages):
                assert "Long-term message" in msg.content

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting

    @pytest.mark.asyncio
    async def test_recent_messages_limit_preserves_other_data(self, async_redis_client):
        """Test that recent messages limit doesn't affect other working memory data"""
        session_id = "test-limit-preserve"
        user_id = "test-user"
        namespace = "test"

        # Create working memory with messages and other data
        messages = []
        for i in range(5):
            messages.append(
                MemoryMessage(
                    id=f"msg-{i}",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                )
            )

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=messages,
            context="This is the context",
            data={"key": "value", "setting": "test"},
            memories=[],
        )

        await set_working_memory(working_memory, redis_client=async_redis_client)

        # Test: Get with recent_messages_limit=2
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=2,
        )

        assert result is not None
        assert len(result.messages) == 2  # Limited messages

        # Other data should be preserved
        assert result.context == "This is the context"
        assert result.data == {"key": "value", "setting": "test"}
        assert result.memories == []
        assert result.session_id == session_id
        assert result.user_id == user_id
        assert result.namespace == namespace

    @pytest.mark.asyncio
    async def test_working_memory_takes_precedence_over_long_term(
        self, async_redis_client
    ):
        """Test that working memory is used instead of long-term memory when both exist"""
        from datetime import UTC, datetime

        from agent_memory_server.config import settings
        from agent_memory_server.long_term_memory import index_long_term_memories
        from agent_memory_server.models import MemoryRecord

        # Enable message indexing
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            session_id = "test-precedence"
            user_id = "test-user"
            namespace = "test"

            # First, create long-term memories
            now = datetime.now(UTC)
            lt_memories = []
            for i in range(3):
                lt_memories.append(
                    MemoryRecord(
                        id=f"lt-msg-{i}",
                        text=f"{'user' if i % 2 == 0 else 'assistant'}: Long-term message {i}",
                        memory_type="message",
                        session_id=session_id,
                        user_id=user_id,
                        namespace=namespace,
                        persisted_at=now,
                    )
                )

            await index_long_term_memories(
                lt_memories,
                redis_client=async_redis_client,
                deduplicate=False,
            )

            # Now create working memory with different messages
            wm_messages = []
            for i in range(2):
                wm_messages.append(
                    MemoryMessage(
                        id=f"wm-msg-{i}",
                        role="user" if i % 2 == 0 else "assistant",
                        content=f"Working memory message {i}",
                    )
                )

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=wm_messages,
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # Test: Get working memory - should return working memory, not long-term
            result = await get_working_memory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
                recent_messages_limit=1,
            )

            assert result is not None
            assert len(result.messages) == 1
            # Should be from working memory, not long-term memory
            assert "Working memory message" in result.messages[0].content
            assert "Long-term message" not in result.messages[0].content

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting

    @pytest.mark.asyncio
    async def test_recent_messages_limit_respects_created_at_order(
        self, async_redis_client
    ):
        """Test that recent messages limit uses created_at for proper chronological ordering"""
        from datetime import UTC, datetime, timedelta

        session_id = "test-created-at-order"
        user_id = "test-user"
        namespace = "test"

        # Create messages with specific created_at timestamps (out of order)
        base_time = datetime.now(UTC)
        messages = [
            MemoryMessage(
                id="msg-1",
                role="user",
                content="First message (oldest)",
                created_at=base_time - timedelta(minutes=10),
            ),
            MemoryMessage(
                id="msg-3",
                role="user",
                content="Third message (newest)",
                created_at=base_time,
            ),
            MemoryMessage(
                id="msg-2",
                role="assistant",
                content="Second message (middle)",
                created_at=base_time - timedelta(minutes=5),
            ),
        ]

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=messages,  # Stored in non-chronological order
        )

        await set_working_memory(working_memory, redis_client=async_redis_client)

        # Test: Get with recent_messages_limit=2 (should get the 2 most recent by created_at)
        result = await get_working_memory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            redis_client=async_redis_client,
            recent_messages_limit=2,
        )

        assert result is not None
        assert len(result.messages) == 2

        # Should get messages in chronological order (oldest first)
        # The 2 most recent should be msg-2 and msg-3
        assert result.messages[0].content == "Second message (middle)"
        assert result.messages[1].content == "Third message (newest)"

        # Verify the timestamps are in correct order
        assert result.messages[0].created_at < result.messages[1].created_at

    @pytest.mark.asyncio
    async def test_message_persistence_sets_correct_memory_type(
        self, async_redis_client
    ):
        """Test that messages persisted to long-term storage have memory_type='message'"""
        from agent_memory_server.config import settings
        from agent_memory_server.filters import MemoryType, SessionId
        from agent_memory_server.long_term_memory import (
            promote_working_memory_to_long_term,
            search_long_term_memories,
        )

        # Enable message indexing
        original_setting = settings.index_all_messages_in_long_term_memory
        settings.index_all_messages_in_long_term_memory = True

        try:
            session_id = "test-message-type"
            user_id = "test-user"
            namespace = "test"

            # Create working memory with messages
            messages = [
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="Test message for memory type verification",
                ),
                MemoryMessage(
                    id="msg-2",
                    role="assistant",
                    content="Response message for memory type verification",
                ),
            ]

            working_memory = WorkingMemory(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                messages=messages,
            )

            await set_working_memory(working_memory, redis_client=async_redis_client)

            # Promote messages to long-term storage
            promoted_count = await promote_working_memory_to_long_term(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=async_redis_client,
            )

            assert (
                promoted_count >= 2
            )  # At least both messages should be promoted (may include extracted memories)

            # Search for the persisted messages
            results = await search_long_term_memories(
                text="",  # Empty query to get all
                session_id=SessionId(eq=session_id),
                memory_type=MemoryType(eq="message"),
                limit=10,
                offset=0,
            )

            assert len(results.memories) == 2  # Should have exactly 2 message memories

            # Verify both messages have the correct memory type
            for memory in results.memories:
                assert memory.memory_type == "message"
                assert memory.session_id == session_id
                assert memory.user_id == user_id
                assert memory.namespace == namespace
                # Verify the text format is "role: content"
                assert ": " in memory.text

        finally:
            settings.index_all_messages_in_long_term_memory = original_setting
