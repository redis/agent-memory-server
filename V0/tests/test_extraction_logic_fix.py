"""
Tests for the extraction logic fixes in long_term_memory.py

With trailing-edge debounce, extraction no longer happens synchronously during
promote_working_memory_to_long_term. Instead, extraction is scheduled to run
after a period of inactivity via run_delayed_extraction.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from agent_memory_server.long_term_memory import (
    _coalesce_summary_memory_data,
    extract_memories_from_session_thread,
    promote_working_memory_to_long_term,
    run_delayed_extraction,
)
from agent_memory_server.models import (
    MemoryMessage,
    MemoryRecord,
    MemoryStrategyConfig,
    WorkingMemory,
)
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
                assert mock_index.call_args.kwargs.get("deduplicate") is True

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
    async def test_thread_aware_extraction_maps_event_date(self, async_redis_client):
        """Test thread-aware extraction preserves event_date on episodic memories."""
        session_id = "test-thread-event-date"
        user_id = "test-user"
        namespace = "test"

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="I met the client on January 15, 2024.",
                    created_at=datetime.now(UTC),
                    discrete_memory_extracted="f",
                ),
            ],
            memories=[],
            long_term_memory_strategy=MemoryStrategyConfig(strategy="discrete"),
        )

        await set_working_memory(working_memory, redis_client=async_redis_client)

        mock_strategy = AsyncMock()
        mock_strategy.extract_memories.return_value = [
            {
                "type": "episodic",
                "text": "User met the client on January 15, 2024.",
                "topics": ["meeting"],
                "entities": ["User", "client"],
                "event_date": "2024-01-15T00:00:00Z",
            }
        ]

        with patch(
            "agent_memory_server.memory_strategies.get_memory_strategy",
            return_value=mock_strategy,
        ):
            extracted_memories = await extract_memories_from_session_thread(
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )

        assert len(extracted_memories) == 1
        assert extracted_memories[0].event_date == datetime(
            2024, 1, 15, 0, 0, tzinfo=UTC
        )

    @pytest.mark.asyncio
    async def test_thread_aware_extraction_skips_non_string_event_date(
        self, async_redis_client
    ):
        """Test thread-aware extraction skips malformed non-string event_date."""
        session_id = "test-thread-event-date-non-string"
        user_id = "test-user"
        namespace = "test"

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="I met the client on January 15, 2024.",
                    created_at=datetime.now(UTC),
                    discrete_memory_extracted="f",
                ),
            ],
            memories=[],
            long_term_memory_strategy=MemoryStrategyConfig(strategy="discrete"),
        )

        await set_working_memory(working_memory, redis_client=async_redis_client)

        mock_strategy = AsyncMock()
        mock_strategy.extract_memories.return_value = [
            {
                "type": "episodic",
                "text": "User met the client on January 15, 2024.",
                "topics": ["meeting"],
                "entities": ["User", "client"],
                "event_date": 1705276800,
            }
        ]

        with patch(
            "agent_memory_server.memory_strategies.get_memory_strategy",
            return_value=mock_strategy,
        ):
            extracted_memories = await extract_memories_from_session_thread(
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )

        assert len(extracted_memories) == 1
        assert extracted_memories[0].event_date is None

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

    @pytest.mark.asyncio
    async def test_summary_strategy_produces_first_class_thread_summary(
        self, async_redis_client
    ):
        """Summary extraction should produce one durable semantic thread memory."""
        session_id = "test-summary-thread"
        user_id = "test-user"
        namespace = "test"

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            messages=[
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="We should use Redis for search.",
                    created_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
                    discrete_memory_extracted="f",
                ),
                MemoryMessage(
                    id="msg-2",
                    role="assistant",
                    content="Redis is a good fit for that.",
                    created_at=datetime(2026, 5, 1, 12, 1, tzinfo=UTC),
                    discrete_memory_extracted="f",
                ),
            ],
            memories=[],
            long_term_memory_strategy=MemoryStrategyConfig(
                strategy="summary",
                config={"topics": ["coding-agent"], "summary_version": "v2"},
            ),
        )
        await set_working_memory(working_memory, redis_client=async_redis_client)

        mock_strategy = AsyncMock()
        mock_strategy.extract_memories.return_value = [
            {
                "type": "semantic",
                "text": "User and assistant discussed using Redis for search.",
                "topics": ["redis"],
                "entities": ["User", "Redis"],
            }
        ]

        with (
            patch(
                "agent_memory_server.long_term_memory._existing_summary_matches_source",
                return_value=False,
            ),
            patch(
                "agent_memory_server.memory_strategies.get_memory_strategy",
                return_value=mock_strategy,
            ),
        ):
            extracted = await extract_memories_from_session_thread(
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )

        assert len(extracted) == 1
        summary = extracted[0]
        assert summary.id.startswith("thread_summary_")
        assert summary.memory_type == "semantic"
        assert summary.extraction_strategy == "summary"
        assert summary.event_date is None
        assert summary.session_id == session_id
        assert summary.namespace == namespace
        assert summary.user_id == user_id
        assert summary.extracted_from == ["msg-1", "msg-2"]
        assert summary.topics == ["coding-agent", "redis", "thread-summary"]
        assert summary.metadata["source_session_id"] == session_id
        assert summary.metadata["message_count"] == 2
        assert summary.metadata["summary_version"] == "v2"
        assert summary.metadata["source_message_ids"] == ["msg-1", "msg-2"]
        assert summary.metadata["source_created_at_min"] == (
            "2026-05-01T12:00:00+00:00"
        )
        assert summary.metadata["source_created_at_max"] == (
            "2026-05-01T12:01:00+00:00"
        )
        assert "source_message_fingerprint" in summary.metadata

        mock_strategy.extract_memories.assert_awaited_once()
        call_text = mock_strategy.extract_memories.await_args.args[0]
        call_context = mock_strategy.extract_memories.await_args.kwargs["context"]
        assert "[USER]: We should use Redis for search." in call_text
        assert call_context["source_message_ids"] == ["msg-1", "msg-2"]

    def test_summary_strategy_coalesces_multiple_model_memories(self):
        """Summary extraction should produce one record even if the model returns many."""
        coalesced = _coalesce_summary_memory_data(
            [
                {
                    "type": "semantic",
                    "text": "User chose Redis for search.",
                    "topics": ["redis"],
                    "entities": ["Redis"],
                },
                {
                    "type": "semantic",
                    "text": "User plans to add dashboard filters.",
                    "topics": ["dashboard"],
                    "entities": ["User"],
                },
            ]
        )

        assert coalesced == [
            {
                "type": "semantic",
                "text": (
                    "User chose Redis for search.\n\n"
                    "User plans to add dashboard filters."
                ),
                "topics": ["dashboard", "redis"],
                "entities": ["Redis", "User"],
            }
        ]

    @pytest.mark.asyncio
    async def test_summary_strategy_skips_unchanged_existing_summary(
        self, async_redis_client
    ):
        """Unchanged summary extraction should skip the model call."""
        session_id = "test-summary-thread-skip"
        working_memory = WorkingMemory(
            session_id=session_id,
            messages=[
                MemoryMessage(
                    id="msg-1",
                    role="user",
                    content="No changes.",
                    created_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
                    discrete_memory_extracted="f",
                )
            ],
            memories=[],
            long_term_memory_strategy=MemoryStrategyConfig(strategy="summary"),
        )
        await set_working_memory(working_memory, redis_client=async_redis_client)

        with (
            patch(
                "agent_memory_server.long_term_memory._existing_summary_matches_source",
                return_value=True,
            ) as mock_existing,
            patch(
                "agent_memory_server.memory_strategies.get_memory_strategy"
            ) as mock_get_strategy,
        ):
            extracted = await extract_memories_from_session_thread(
                session_id=session_id
            )

        assert extracted == []
        mock_existing.assert_awaited_once()
        mock_get_strategy.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_strategy_changed_thread_keeps_same_id(
        self, async_redis_client
    ):
        """Changed sessions should refresh the same deterministic summary ID."""
        session_id = "test-summary-thread-same-id"

        async def run_once(message_text: str) -> str:
            working_memory = WorkingMemory(
                session_id=session_id,
                messages=[
                    MemoryMessage(
                        id="msg-1",
                        role="user",
                        content=message_text,
                        created_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
                        discrete_memory_extracted="f",
                    )
                ],
                memories=[],
                long_term_memory_strategy=MemoryStrategyConfig(strategy="summary"),
            )
            await set_working_memory(working_memory, redis_client=async_redis_client)

            mock_strategy = AsyncMock()
            mock_strategy.extract_memories.return_value = [
                {"type": "semantic", "text": f"Summary: {message_text}"}
            ]
            with (
                patch(
                    "agent_memory_server.long_term_memory._existing_summary_matches_source",
                    return_value=False,
                ),
                patch(
                    "agent_memory_server.memory_strategies.get_memory_strategy",
                    return_value=mock_strategy,
                ),
            ):
                extracted = await extract_memories_from_session_thread(
                    session_id=session_id
                )
            return extracted[0].id

        first_id = await run_once("Initial content.")
        second_id = await run_once("Changed content.")

        assert first_id == second_id
