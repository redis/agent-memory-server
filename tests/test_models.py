from redis_memory_server.models import (
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
        assert payload.last_accessed is None
        assert payload.created_at is None

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
        assert response.last_accessed is None
        assert response.created_at is None

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

    def test_search_payload(self):
        """Test SearchPayload model"""

        # Test default pagination
        payload = SearchPayload(text="What is the capital of France?")
        assert payload.text == "What is the capital of France?"
        assert payload.limit == 10
        assert payload.offset == 0

        # Test with all fields
        payload = SearchPayload(
            text="What is the capital of France?",
            session_id="session_id",
            namespace="namespace",
            topics=["France", "Paris"],
            entities=["France", "Paris"],
            distance_threshold=0.5,
        )
        assert payload.text == "What is the capital of France?"
        assert payload.session_id == "session_id"
        assert payload.namespace == "namespace"
        assert payload.topics == ["France", "Paris"]
        assert payload.entities == ["France", "Paris"]
        assert payload.distance_threshold == 0.5
        assert payload.limit == 10
        assert payload.offset == 0

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
