from datetime import UTC, datetime

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
    MemoryMessage,
    MemoryRecordResult,
    SearchRequest,
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
            text="Paris is the capital of France",
            dist=0.75,
            id_="123",
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
