import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    CreateMemoryRecordRequest,
    ExtractedMemoryRecord,
    LenientMemoryRecord,
    MemoryMessage,
    MemoryRecordResult,
    SearchRequest,
    UpdateWorkingMemory,
    WorkingMemory,
    WorkingMemoryResponse,
)


class TestModels:
    def test_memory_message(self):
        """Test MemoryMessage model"""
        msg = MemoryMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_working_memory(self):
        """Test WorkingMemory model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test with required fields
        payload = WorkingMemory(
            messages=messages,
            memories=[],
            session_id="test-session",
        )
        assert payload.messages == messages
        assert payload.memories == []
        assert payload.session_id == "test-session"
        assert payload.context is None
        assert payload.user_id is None
        assert payload.namespace is None
        assert payload.tokens == 0
        assert payload.last_accessed > datetime(2020, 1, 1, tzinfo=UTC)
        assert payload.created_at > datetime(2020, 1, 1, tzinfo=UTC)
        assert isinstance(payload.last_accessed, datetime)
        assert isinstance(payload.created_at, datetime)

        # Test with all fields
        test_datetime = datetime(2023, 1, 1, tzinfo=UTC)
        payload = WorkingMemory(
            messages=messages,
            memories=[],
            context="Previous conversation summary",
            user_id="user_id",
            session_id="session_id",
            namespace="namespace",
            tokens=100,
            last_accessed=test_datetime,
            created_at=test_datetime,
        )
        assert payload.messages == messages
        assert payload.memories == []
        assert payload.context == "Previous conversation summary"
        assert payload.user_id == "user_id"
        assert payload.session_id == "session_id"
        assert payload.namespace == "namespace"
        assert payload.tokens == 100
        assert payload.last_accessed == test_datetime
        assert payload.created_at == test_datetime

    def test_working_memory_response(self):
        """Test WorkingMemoryResponse model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test with required fields
        response = WorkingMemoryResponse(
            messages=messages,
            memories=[],
            session_id="test-session",
        )
        assert response.messages == messages
        assert response.memories == []
        assert response.session_id == "test-session"
        assert response.context is None
        assert response.tokens == 0
        assert response.user_id is None
        assert response.namespace is None
        assert response.last_accessed > datetime(2020, 1, 1, tzinfo=UTC)
        assert response.created_at > datetime(2020, 1, 1, tzinfo=UTC)
        assert isinstance(response.last_accessed, datetime)
        assert isinstance(response.created_at, datetime)

        # Test with all fields
        test_datetime = datetime(2023, 1, 1, tzinfo=UTC)
        response = WorkingMemoryResponse(
            messages=messages,
            memories=[],
            context="Conversation summary",
            tokens=150,
            user_id="user_id",
            session_id="session_id",
            namespace="namespace",
            last_accessed=test_datetime,
            created_at=test_datetime,
        )
        assert response.messages == messages
        assert response.memories == []
        assert response.context == "Conversation summary"
        assert response.tokens == 150
        assert response.user_id == "user_id"
        assert response.session_id == "session_id"
        assert response.namespace == "namespace"
        assert response.last_accessed == test_datetime
        assert response.created_at == test_datetime

    def test_memory_record_result(self):
        """Test MemoryRecordResult model"""
        test_datetime = datetime(2023, 1, 1, tzinfo=UTC)
        result = MemoryRecordResult(
            id="record-123",
            text="Paris is the capital of France",
            dist=0.75,
            session_id="session_id",
            user_id="user_id",
            last_accessed=test_datetime,
            created_at=test_datetime,
            namespace="namespace",
        )
        assert result.text == "Paris is the capital of France"
        assert result.dist == 0.75

    def test_search_payload_with_filter_objects(self):
        """Test SearchPayload model with filter objects"""

        # Create filter objects directly
        session_id = SessionId(eq="test-session")
        namespace = Namespace(eq="test-namespace")
        topics = Topics(any=["topic1", "topic2"])
        entities = Entities(any=["entity1", "entity2"])
        created_at = CreatedAt(
            gt=datetime(2023, 1, 1, tzinfo=UTC),
            lt=datetime(2023, 12, 31, tzinfo=UTC),
        )
        last_accessed = LastAccessed(
            gt=datetime(2023, 6, 1, tzinfo=UTC),
            lt=datetime(2023, 12, 1, tzinfo=UTC),
        )
        user_id = UserId(eq="test-user")

        # Create payload with filter objects
        payload = SearchRequest(
            text="Test query",
            session_id=session_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            user_id=user_id,
            distance_threshold=0.7,
            limit=15,
            offset=5,
        )

        # Check if payload contains filter objects
        assert payload.text == "Test query"
        assert payload.session_id == session_id
        assert payload.namespace == namespace
        assert payload.topics == topics
        assert payload.entities == entities
        assert payload.created_at == created_at
        assert payload.last_accessed == last_accessed
        assert payload.user_id == user_id
        assert payload.distance_threshold == 0.7
        assert payload.limit == 15
        assert payload.offset == 5

        # Test get_filters method
        filters = payload.get_filters()
        assert filters["session_id"] == session_id
        assert filters["namespace"] == namespace
        assert filters["topics"] == topics
        assert filters["entities"] == entities
        assert filters["created_at"] == created_at
        assert filters["last_accessed"] == last_accessed
        assert filters["user_id"] == user_id


class TestMemoryMessageTimestampValidation:
    """Tests for MemoryMessage created_at timestamp validation"""

    def setup_method(self):
        """Clear the warned message IDs cache before each test"""
        MemoryMessage._warned_message_ids.clear()

    def test_message_with_explicit_created_at_no_warning(self, caplog):
        """Test that providing created_at does not trigger a warning"""
        with caplog.at_level(logging.WARNING):
            msg = MemoryMessage(
                role="user",
                content="Hello",
                created_at=datetime.now(UTC),
            )
            assert msg.role == "user"
            assert msg.content == "Hello"
            # No warning should be logged
            assert "created_at timestamp" not in caplog.text

    def test_message_without_created_at_logs_warning(self, caplog):
        """Test that missing created_at triggers a deprecation warning"""
        with caplog.at_level(logging.WARNING):
            msg = MemoryMessage(role="user", content="Hello")
            assert msg.role == "user"
            # Warning should be logged
            assert "created_at timestamp" in caplog.text
            assert "required in a future version" in caplog.text

    def test_message_warning_rate_limited(self, caplog):
        """Test that warnings are rate-limited per message ID"""
        with caplog.at_level(logging.WARNING):
            # First message without created_at - should warn
            MemoryMessage(id="msg-1", role="user", content="Hello")
            assert "created_at timestamp" in caplog.text

            caplog.clear()

            # Same message ID again - should NOT warn (rate-limited)
            MemoryMessage(id="msg-1", role="user", content="Hello again")
            assert "created_at timestamp" not in caplog.text

            # Different message ID - should warn
            MemoryMessage(id="msg-2", role="user", content="Different message")
            assert "created_at timestamp" in caplog.text

    def test_message_with_future_timestamp_rejected(self):
        """Test that future timestamps (beyond tolerance) are rejected"""
        # Create a timestamp 10 minutes in the future (beyond 5 min tolerance)
        future_time = datetime.now(UTC) + timedelta(minutes=10)

        with pytest.raises(ValueError) as exc_info:
            MemoryMessage(
                role="user",
                content="Hello",
                created_at=future_time,
            )

        assert "cannot be more than" in str(exc_info.value)
        assert "seconds in the future" in str(exc_info.value)

    def test_message_with_near_future_timestamp_allowed(self):
        """Test that timestamps within tolerance are allowed"""
        # Create a timestamp 2 minutes in the future (within 5 min tolerance)
        near_future = datetime.now(UTC) + timedelta(minutes=2)

        msg = MemoryMessage(
            role="user",
            content="Hello",
            created_at=near_future,
        )
        assert msg.created_at == near_future

    def test_message_with_past_timestamp_allowed(self):
        """Test that past timestamps are allowed"""
        past_time = datetime(2023, 1, 1, tzinfo=UTC)

        msg = MemoryMessage(
            role="user",
            content="Hello",
            created_at=past_time,
        )
        assert msg.created_at == past_time

    def test_message_with_iso_string_timestamp(self):
        """Test that ISO format string timestamps are parsed correctly"""
        msg = MemoryMessage(
            role="user",
            content="Hello",
            created_at="2023-06-15T10:30:00+00:00",
        )
        assert msg.created_at == datetime(2023, 6, 15, 10, 30, 0, tzinfo=UTC)

    def test_message_with_z_suffix_timestamp(self):
        """Test that Z suffix timestamps are parsed correctly"""
        msg = MemoryMessage(
            role="user",
            content="Hello",
            created_at="2023-06-15T10:30:00Z",
        )
        assert msg.created_at == datetime(2023, 6, 15, 10, 30, 0, tzinfo=UTC)

    def test_require_message_timestamps_setting(self):
        """Test that require_message_timestamps=True rejects missing timestamps"""
        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.require_message_timestamps = True
            mock_settings.max_future_timestamp_seconds = 300

            with pytest.raises(ValueError) as exc_info:
                MemoryMessage(role="user", content="Hello")

            assert "created_at is required" in str(exc_info.value)

    def test_created_at_was_provided_flag(self):
        """Test that _created_at_was_provided flag is set correctly"""
        # With explicit created_at
        msg1 = MemoryMessage(
            role="user",
            content="Hello",
            created_at=datetime.now(UTC),
        )
        assert msg1._created_at_was_provided is True

        # Without created_at
        msg2 = MemoryMessage(role="user", content="Hello")
        assert msg2._created_at_was_provided is False


class TestEmptyTextValidation:
    """Tests for validation that rejects empty text in memory records."""

    def test_create_memory_record_request_rejects_empty_text(self):
        """Test that CreateMemoryRecordRequest rejects memories with empty text."""
        valid_memory = ExtractedMemoryRecord(
            id="valid-id",
            text="Valid memory text",
        )
        empty_text_memory = ExtractedMemoryRecord(
            id="empty-text-id",
            text="",
        )

        with pytest.raises(ValueError, match="has empty text"):
            CreateMemoryRecordRequest(memories=[valid_memory, empty_text_memory])

    def test_create_memory_record_request_rejects_empty_id(self):
        """Test that CreateMemoryRecordRequest rejects memories with empty id."""
        empty_id_memory = ExtractedMemoryRecord(
            id="",
            text="Valid text",
        )

        with pytest.raises(ValueError, match="has empty id"):
            CreateMemoryRecordRequest(memories=[empty_id_memory])

    def test_create_memory_record_request_accepts_valid_memories(self):
        """Test that CreateMemoryRecordRequest accepts valid memories."""
        valid_memory = ExtractedMemoryRecord(
            id="valid-id",
            text="Valid memory text",
        )
        request = CreateMemoryRecordRequest(memories=[valid_memory])
        assert len(request.memories) == 1
        assert request.memories[0].id == "valid-id"
        assert request.memories[0].text == "Valid memory text"

    def test_update_working_memory_rejects_empty_text(self):
        """Test that UpdateWorkingMemory rejects memories with empty text."""
        empty_text_memory = ExtractedMemoryRecord(
            id="empty-text-id",
            text="",
        )

        with pytest.raises(ValueError, match="has empty text"):
            UpdateWorkingMemory(memories=[empty_text_memory])

    def test_update_working_memory_rejects_empty_id(self):
        """Test that UpdateWorkingMemory rejects memories with empty id."""
        empty_id_memory = ExtractedMemoryRecord(
            id="",
            text="Valid text",
        )

        with pytest.raises(ValueError, match="has empty id"):
            UpdateWorkingMemory(memories=[empty_id_memory])

    def test_update_working_memory_accepts_valid_memories(self):
        """Test that UpdateWorkingMemory accepts valid memories."""
        valid_memory = ExtractedMemoryRecord(
            id="valid-id",
            text="Valid memory text",
        )
        request = UpdateWorkingMemory(memories=[valid_memory])
        assert len(request.memories) == 1

    def test_lenient_memory_record_rejects_empty_text(self):
        """Test that LenientMemoryRecord rejects empty text."""
        with pytest.raises(ValueError, match="Memory text cannot be empty"):
            LenientMemoryRecord(text="")

    def test_lenient_memory_record_accepts_valid_text(self):
        """Test that LenientMemoryRecord accepts valid text."""
        record = LenientMemoryRecord(text="Valid memory text")
        assert record.text == "Valid memory text"

    def test_lenient_memory_record_accepts_whitespace(self):
        """Test that LenientMemoryRecord accepts whitespace-only text."""
        # Whitespace is accepted (it's not empty, just whitespace)
        record = LenientMemoryRecord(text="   ")
        assert record.text == "   "
