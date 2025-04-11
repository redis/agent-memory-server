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
    LongTermMemoryResult,
    MemoryMessage,
    SearchPayload,
    SessionMemory,
    SessionMemoryResponse,
)


class TestModels:
    def test_memory_message(self):
        """Test MemoryMessage model"""
        msg = MemoryMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_session_memory(self):
        """Test SessionMemory model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test without any optional fields
        payload = SessionMemory(messages=messages)
        assert payload.messages == messages
        assert payload.context is None
        assert payload.user_id is None
        assert payload.session_id is None
        assert payload.namespace is None
        assert payload.tokens == 0
        assert payload.last_accessed > 1
        assert payload.created_at > 1

        # Test with all fields
        payload = SessionMemory(
            messages=messages,
            context="Previous conversation summary",
            user_id="user_id",
            session_id="session_id",
            namespace="namespace",
            tokens=100,
            last_accessed=100,
            created_at=100,
        )
        assert payload.messages == messages
        assert payload.context == "Previous conversation summary"
        assert payload.user_id == "user_id"
        assert payload.session_id == "session_id"
        assert payload.namespace == "namespace"
        assert payload.tokens == 100
        assert payload.last_accessed == 100
        assert payload.created_at == 100

    def test_memory_response(self):
        """Test SessionMemoryResponse model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test without any optional fields
        response = SessionMemoryResponse(messages=messages)
        assert response.messages == messages
        assert response.context is None
        assert response.tokens == 0
        assert response.user_id is None
        assert response.session_id is None
        assert response.namespace is None
        assert response.last_accessed > 1
        assert response.created_at > 1

        # Test with all fields
        response = SessionMemoryResponse(
            messages=messages,
            context="Conversation summary",
            tokens=150,
            user_id="user_id",
            session_id="session_id",
            namespace="namespace",
            last_accessed=100,
            created_at=100,
        )
        assert response.messages == messages
        assert response.context == "Conversation summary"
        assert response.tokens == 150
        assert response.user_id == "user_id"
        assert response.session_id == "session_id"
        assert response.namespace == "namespace"
        assert response.last_accessed == 100
        assert response.created_at == 100

    def test_long_term_memory_result(self):
        """Test LongTermMemoryResult model"""
        result = LongTermMemoryResult(
            text="Paris is the capital of France",
            dist=0.75,
            id_="123",
            session_id="session_id",
            user_id="user_id",
            last_accessed=100,
            created_at=100,
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
        created_at = CreatedAt(gt=1000, lt=2000)
        last_accessed = LastAccessed(gt=3000, lt=4000)
        user_id = UserId(eq="test-user")

        # Create payload with filter objects
        payload = SearchPayload(
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
