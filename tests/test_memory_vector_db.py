"""Tests for the MemoryVectorDatabase abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.memory_vector_db import (
    MemoryVectorDatabase,
    RedisVLMemoryVectorDatabase,
)
from agent_memory_server.memory_vector_db_factory import (
    create_embeddings,
    create_memory_vector_db,
)
from agent_memory_server.models import MemoryRecord, MemoryRecordResults, MemoryTypeEnum


class MockEmbeddings:
    """Mock embeddings for testing."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._dimensions = dimensions
        self.model = "mock-embedding-model"

    def embed_documents(self, texts):
        return [[0.1] * self.dimensions for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dimensions

    async def aembed_documents(self, texts):
        return [[0.1] * self.dimensions for _ in texts]

    async def aembed_query(self, text):
        return [0.1] * self.dimensions


class TestMemoryVectorDatabase:
    """Test cases for MemoryVectorDatabase functionality."""

    def test_memory_hash_generation(self):
        """Test memory hash generation."""
        # Create a concrete implementation for testing
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Create a sample memory
        memory = MemoryRecord(
            text="This is a test memory",
            id="test-hash-123",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Generate hash
        hash1 = db.generate_memory_hash(memory)
        hash2 = db.generate_memory_hash(memory)

        # Verify hash is stable
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

        # Verify different memories produce different hashes
        different_memory = MemoryRecord(
            text="This is a different memory",
            id="test-hash-456",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        different_hash = db.generate_memory_hash(different_memory)
        assert hash1 != different_hash

    def test_parse_list_field(self):
        """Test parsing of list fields."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()
        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Test with None
        assert db._parse_list_field(None) == []

        # Test with empty string
        assert db._parse_list_field("") == []

        # Test with comma-separated string
        assert db._parse_list_field("a,b,c") == ["a", "b", "c"]

        # Test with list
        assert db._parse_list_field(["a", "b"]) == ["a", "b"]

    def test_memory_to_data_conversion(self):
        """Test converting MemoryRecord to data dict."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()
        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        memory = MemoryRecord(
            text="This is a test memory",
            id="test-123",
            session_id="session-456",
            user_id="user-789",
            namespace="test",
            topics=["testing", "memory"],
            entities=["test"],
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        data = db._memory_to_data(memory)

        assert data["text"] == "This is a test memory"
        assert data["id_"] == "test-123"
        assert data["session_id"] == "session-456"
        assert data["user_id"] == "user-789"
        assert data["namespace"] == "test"
        assert data["topics"] == "testing,memory"
        assert data["entities"] == "test"
        assert data["memory_type"] == "semantic"

    def test_data_to_memory_result_conversion(self):
        """Test converting data dict to MemoryRecordResult."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()
        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        fields = {
            "id_": "test-123",
            "text": "This is a test memory",
            "session_id": "session-456",
            "user_id": "user-789",
            "namespace": "test",
            "topics": "testing,memory",
            "entities": "test",
            "memory_type": "semantic",
            "created_at": "1704067200",  # 2024-01-01T00:00:00Z
            "last_accessed": "1704067200",
            "updated_at": "1704067200",
            "discrete_memory_extracted": "t",
        }

        result = db._data_to_memory_result(fields, score=0.2)

        assert result.text == "This is a test memory"
        assert result.id == "test-123"
        assert result.session_id == "session-456"
        assert result.user_id == "user-789"
        assert result.namespace == "test"
        assert result.topics == ["testing", "memory"]
        assert result.entities == ["test"]
        assert result.memory_type == "semantic"
        assert result.dist == 0.2
        assert result.discrete_memory_extracted == "t"

    @pytest.mark.asyncio
    async def test_add_memories_with_mock_index(self):
        """Test adding memories to a mock index."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_index.load = AsyncMock(return_value=["key1", "key2"])
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        memories = [
            MemoryRecord(
                text="Memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            MemoryRecord(
                text="Memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        ids = await db.add_memories(memories)

        assert ids == ["mem1", "mem2"]
        mock_index.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_memories_handling(self):
        """Test handling of empty memory lists."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Test adding empty list
        ids = await db.add_memories([])
        assert ids == []

        # Test deleting empty list
        deleted = await db.delete_memories([])
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_update_memories(self):
        """Test update_memories method calls add_memories."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_index.load = AsyncMock(return_value=["key1", "key2"])
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        memories = [
            MemoryRecord(
                text="Updated memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",
            ),
            MemoryRecord(
                text="Updated memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",
            ),
        ]

        count = await db.update_memories(memories)

        # update_memories delegates to add_memories
        mock_index.load.assert_called_once()
        assert count == 2

    @pytest.mark.asyncio
    async def test_update_memories_empty_list(self):
        """Test update_memories with empty list."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        count = await db.update_memories([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_memories(self):
        """Test delete_memories calls drop_documents."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_index.drop_documents = AsyncMock(return_value=2)
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        deleted = await db.delete_memories(["mem1", "mem2"])

        mock_index.drop_documents.assert_called_once_with(["mem1", "mem2"])
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_factory_creates_redisvl_db(self):
        """Test that the factory creates a RedisVLMemoryVectorDatabase."""
        import agent_memory_server.memory_vector_db_factory

        # Clear the global instance to force recreation
        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

        # Mock embeddings to avoid API key requirement
        with patch(
            "agent_memory_server.memory_vector_db_factory.create_embeddings"
        ) as mock_create_embeddings:
            mock_create_embeddings.return_value = MockEmbeddings()

            db = create_memory_vector_db()

            # Should get RedisVLMemoryVectorDatabase
            assert isinstance(db, RedisVLMemoryVectorDatabase)

        # Reset the global instance
        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

    @pytest.mark.asyncio
    async def test_factory_supports_custom_factory(self):
        """Test that the factory supports custom MemoryVectorDatabase implementations."""
        import agent_memory_server.memory_vector_db_factory

        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

        class CustomMemoryVectorDatabase(MemoryVectorDatabase):
            def __init__(self):
                pass

            async def add_memories(self, memories):
                return []

            async def search_memories(self, query, **kwargs):
                return MemoryRecordResults(memories=[], total=0, next_offset=None)

            async def count_memories(self, **kwargs):
                return 0

            async def delete_memories(self, memory_ids):
                return 0

            async def update_memories(self, memories):
                return 0

            async def list_memories(self, **kwargs):
                return MemoryRecordResults(memories=[], total=0, next_offset=None)

        with (
            patch(
                "agent_memory_server.memory_vector_db_factory.create_embeddings"
            ) as mock_create_embeddings,
            patch(
                "agent_memory_server.memory_vector_db_factory._import_and_call_factory"
            ) as mock_import_factory,
        ):
            mock_embeddings = MockEmbeddings()
            mock_create_embeddings.return_value = mock_embeddings

            custom_db = CustomMemoryVectorDatabase()
            mock_import_factory.return_value = custom_db

            db = create_memory_vector_db()

            assert db == custom_db

        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

    def test_redis_adapter_preserves_discrete_memory_extracted_flag(self):
        """Regression test: Ensure data_to_memory_result preserves discrete_memory_extracted='t'."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Simulate fields from a Redis search result
        from datetime import UTC, datetime

        fields = {
            "id_": "memory_001",
            "text": "User likes green tea",
            "session_id": "",
            "user_id": "",
            "namespace": "user_preferences",
            "created_at": str(datetime.now(UTC).timestamp()),
            "updated_at": str(datetime.now(UTC).timestamp()),
            "last_accessed": str(datetime.now(UTC).timestamp()),
            "topics": "preferences,beverages",
            "entities": "",
            "memory_hash": "abc123",
            "discrete_memory_extracted": "t",  # This should be preserved!
            "memory_type": "semantic",
            "persisted_at": None,
            "extracted_from": "",
            "event_date": None,
        }

        result = db._data_to_memory_result(fields, score=0.1)

        # REGRESSION TEST: This should be 't', not 'f'
        assert result.discrete_memory_extracted == "t", (
            f"Regression: Expected discrete_memory_extracted='t', got '{result.discrete_memory_extracted}'. "
            f"This indicates the adapter is not preserving the flag."
        )

        assert result.memory_type == "semantic"
        assert result.namespace == "user_preferences"
        assert result.text == "User likes green tea"


class TestCreateEmbeddings:
    """Test cases for the create_embeddings function.

    Note: The embedding creation logic is now in LLMClient.create_embeddings(),
    which returns LiteLLMEmbeddings for all providers.
    """

    def test_create_embeddings_returns_litellm_embeddings(self):
        """Test that create_embeddings returns LiteLLMEmbeddings."""
        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_model_config = MagicMock()
        mock_model_config.provider = ModelProvider.OPENAI
        mock_model_config.embedding_dimensions = 1536

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_api_base = None

            result = create_embeddings()

            assert isinstance(result, LiteLLMEmbeddings)
            assert result.model == "text-embedding-3-small"

    def test_create_embeddings_aws_bedrock_adds_prefix(self):
        """Test that Bedrock models get bedrock/ prefix added."""
        import warnings

        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_model_config = MagicMock()
        mock_model_config.provider = ModelProvider.AWS_BEDROCK
        mock_model_config.embedding_dimensions = 1024

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "amazon.titan-embed-text-v2:0"
            mock_settings.openai_api_key = None
            mock_settings.openai_api_base = None

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = create_embeddings()

                # Should emit deprecation warning for unprefixed model
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)

            assert isinstance(result, LiteLLMEmbeddings)
            assert result.model == "bedrock/amazon.titan-embed-text-v2:0"

    def test_create_embeddings_anthropic_raises_error(self):
        """Test that Anthropic provider raises error (no embedding models)."""
        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.exceptions import ModelValidationError

        mock_model_config = MagicMock()
        mock_model_config.provider = ModelProvider.ANTHROPIC

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "claude-embedding"

            with pytest.raises(
                ModelValidationError, match="Anthropic does not provide embedding"
            ):
                create_embeddings()
