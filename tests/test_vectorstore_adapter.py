"""Tests for the VectorStore adapter functionality."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.filters import Namespace
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum
from agent_memory_server.vectorstore_adapter import (
    LangChainVectorStoreAdapter,
    RedisVectorStoreAdapter,
    VectorStoreAdapter,
)
from agent_memory_server.vectorstore_factory import (
    create_embeddings,
    create_vectorstore_adapter,
)


class TestVectorStoreAdapter:
    """Test cases for VectorStore adapter functionality."""

    def test_memory_to_document_conversion(self):
        """Test converting MemoryRecord to LangChain Document."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create a sample memory
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

        # Convert to document
        doc = adapter.memory_to_document(memory)

        # Verify conversion
        assert doc.page_content == "This is a test memory"
        assert doc.metadata["id_"] == "test-123"
        assert doc.metadata["id"] == "test-123"
        assert doc.metadata["session_id"] == "session-456"
        assert doc.metadata["user_id"] == "user-789"
        assert doc.metadata["namespace"] == "test"
        assert doc.metadata["topics"] == ["testing", "memory"]
        assert doc.metadata["entities"] == ["test"]
        assert doc.metadata["memory_type"] == "semantic"

    def test_document_to_memory_conversion(self):
        """Test converting LangChain Document to MemoryRecordResult."""
        from langchain_core.documents import Document

        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create a sample document
        doc = Document(
            page_content="This is a test memory",
            metadata={
                "id": "test-123",
                "session_id": "session-456",
                "user_id": "user-789",
                "namespace": "test",
                "topics": ["testing", "memory"],
                "entities": ["test"],
                "memory_type": "semantic",
                "created_at": "2024-01-01T00:00:00Z",
                "last_accessed": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
        )

        # Convert to memory
        memory_result = adapter.document_to_memory(doc, score=0.8)

        # Verify conversion
        assert memory_result.text == "This is a test memory"
        assert memory_result.id == "test-123"
        assert memory_result.session_id == "session-456"
        assert memory_result.user_id == "user-789"
        assert memory_result.namespace == "test"
        assert memory_result.topics == ["testing", "memory"]
        assert memory_result.entities == ["test"]
        assert memory_result.memory_type == "semantic"
        assert memory_result.dist == 0.8

    @pytest.mark.asyncio
    async def test_add_memories_with_mock_vectorstore(self):
        """Test adding memories to a mock vector store."""
        # Create a mock VectorStore with proper async mocking
        mock_vectorstore = MagicMock()
        mock_vectorstore.aadd_documents = AsyncMock(return_value=["doc1", "doc2"])
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create sample memories
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

        # Add memories
        ids = await adapter.add_memories(memories)

        # Verify
        assert ids == ["doc1", "doc2"]
        mock_vectorstore.aadd_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_vectorstore_factory_creates_adapter(self):
        """Integration test: verify that the factory can create an adapter."""
        import agent_memory_server.vectorstore_factory
        from tests.conftest import MockEmbeddings

        # Clear the global adapter to force recreation
        agent_memory_server.vectorstore_factory._adapter = None

        # Test with Redis backend (default factory) - mock embeddings to avoid API key requirement
        with patch(
            "agent_memory_server.vectorstore_factory.create_embeddings"
        ) as mock_create_embeddings:
            mock_create_embeddings.return_value = MockEmbeddings()

            adapter = create_vectorstore_adapter()

            # For Redis backend, we should get RedisVectorStoreAdapter
            assert isinstance(adapter, RedisVectorStoreAdapter)

        # Reset the global adapter
        agent_memory_server.vectorstore_factory._adapter = None

        # Test with custom factory function that returns a VectorStore
        with (
            patch(
                "agent_memory_server.vectorstore_factory.create_embeddings"
            ) as mock_create_embeddings,
            patch(
                "agent_memory_server.vectorstore_factory._import_and_call_factory"
            ) as mock_import_factory,
            patch("agent_memory_server.vectorstore_factory.settings") as mock_settings,
        ):
            # Mock the embeddings
            mock_embeddings = MagicMock()
            mock_create_embeddings.return_value = mock_embeddings

            # Mock settings to use a non-Redis factory path
            mock_settings.vectorstore_factory = "my_module.create_custom_vectorstore"

            # Create a proper mock VectorStore that actually inherits from VectorStore
            from langchain_core.vectorstores import VectorStore

            class MockVectorStore(VectorStore):
                def add_texts(self, texts, metadatas=None, **kwargs):
                    return []

                def similarity_search(self, query, k=4, **kwargs):
                    return []

                @classmethod
                def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
                    return cls()

            mock_vectorstore = MockVectorStore()
            mock_import_factory.return_value = mock_vectorstore

            # Create adapter with mocked factory
            adapter = create_vectorstore_adapter()

            # For non-Redis backends, we should get LangChainVectorStoreAdapter
            assert isinstance(adapter, LangChainVectorStoreAdapter)
            assert adapter.vectorstore == mock_vectorstore
            assert adapter.embeddings == mock_embeddings

        # Test that factory function can also return VectorStoreAdapter directly
        agent_memory_server.vectorstore_factory._adapter = None

        with (
            patch(
                "agent_memory_server.vectorstore_factory.create_embeddings"
            ) as mock_create_embeddings,
            patch(
                "agent_memory_server.vectorstore_factory._import_and_call_factory"
            ) as mock_import_factory,
        ):
            # Mock the embeddings and custom adapter
            mock_embeddings = MagicMock()

            # Create a proper mock VectorStoreAdapter that actually inherits from VectorStoreAdapter
            class MockVectorStoreAdapter(VectorStoreAdapter):
                def __init__(self):
                    pass  # Skip parent constructor for test

                # Add minimal required methods for test
                async def add_memories(self, memories):
                    return []

                async def search_memories(self, query, **kwargs):
                    return []

                async def count_memories(self, **kwargs):
                    return 0

                async def delete_memories(self, memory_ids):
                    return 0

                async def update_memories(self, memories: list[MemoryRecord]) -> int:
                    return 0

            mock_custom_adapter = MockVectorStoreAdapter()

            mock_create_embeddings.return_value = mock_embeddings
            mock_import_factory.return_value = mock_custom_adapter

            # Create adapter with mocked factory that returns adapter directly
            adapter = create_vectorstore_adapter()

            # Should get the custom adapter returned directly
            assert adapter == mock_custom_adapter

    def test_memory_hash_generation(self):
        """Test memory hash generation."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create a sample memory
        memory = MemoryRecord(
            text="This is a test memory",
            id="test-hash-123",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Generate hash
        hash1 = adapter.generate_memory_hash(memory)
        hash2 = adapter.generate_memory_hash(memory)

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
        different_hash = adapter.generate_memory_hash(different_memory)
        assert hash1 != different_hash

    @pytest.mark.asyncio
    async def test_empty_memories_handling(self):
        """Test handling of empty memory lists."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Test adding empty list
        ids = await adapter.add_memories([])
        assert ids == []

        # Test deleting empty list
        deleted = await adapter.delete_memories([])
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_update_memories_functionality(self):
        """Test the update_memories method works correctly."""
        # Create a mock VectorStore with proper async mocking
        mock_vectorstore = MagicMock()
        mock_vectorstore.aadd_documents = AsyncMock(return_value=["doc1", "doc2"])
        # Mock adelete method as async since LangChainVectorStoreAdapter checks for adelete first
        mock_vectorstore.adelete = AsyncMock(return_value=True)
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create sample memories to update
        memories = [
            MemoryRecord(
                text="Updated memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",  # Updated value
            ),
            MemoryRecord(
                text="Updated memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",  # Updated value
            ),
        ]

        # Update memories
        count = await adapter.update_memories(memories)

        # Verify that adelete was called once with all memory IDs
        assert mock_vectorstore.adelete.call_count == 1
        # Check that it was called with the correct IDs
        mock_vectorstore.adelete.assert_called_with(["mem1", "mem2"])
        # Verify that add was called
        mock_vectorstore.aadd_documents.assert_called_once()
        # Verify return count
        assert count == 2

    @pytest.mark.asyncio
    async def test_update_memories_empty_list(self):
        """Test update_memories with empty list."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Update with empty list
        count = await adapter.update_memories([])

        # Should return 0 and not call any vectorstore methods
        assert count == 0
        # Don't check for method calls since they shouldn't be called with empty list

    @pytest.mark.asyncio
    async def test_redis_adapter_update_memories(self):
        """Test RedisVectorStoreAdapter update_memories method."""
        # Create a mock RedisVectorStore
        mock_redis_vectorstore = MagicMock()
        mock_redis_vectorstore.aadd_documents = AsyncMock(return_value=["key1", "key2"])
        mock_embeddings = MagicMock()

        # Create Redis adapter
        adapter = RedisVectorStoreAdapter(mock_redis_vectorstore, mock_embeddings)

        # Create sample memories to update
        memories = [
            MemoryRecord(
                text="Updated Redis memory 1",
                id="redis-mem1",
                memory_type=MemoryTypeEnum.MESSAGE,
                discrete_memory_extracted="t",  # Updated value
            ),
            MemoryRecord(
                text="Updated Redis memory 2",
                id="redis-mem2",
                memory_type=MemoryTypeEnum.MESSAGE,
                discrete_memory_extracted="t",  # Updated value
            ),
        ]

        # Update memories
        count = await adapter.update_memories(memories)

        # For Redis adapter, update just calls add_memories
        mock_redis_vectorstore.aadd_documents.assert_called_once()
        assert count == 2

    @pytest.mark.asyncio
    async def test_search_with_discrete_memory_extracted_filter(self):
        """Test searching with discrete_memory_extracted filter."""
        from agent_memory_server.filters import DiscreteMemoryExtracted

        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Mock search results
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Processed memory"
        mock_doc1.metadata = {
            "id_": "mem1",
            "discrete_memory_extracted": "t",
            "memory_type": "semantic",
            "created_at": 1609459200,
            "last_accessed": 1609459200,
            "updated_at": 1609459200,
        }

        # Mock the async search method that the adapter actually uses
        mock_vectorstore.asimilarity_search_with_relevance_scores = AsyncMock(
            return_value=[(mock_doc1, 0.8)]
        )

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Search with discrete_memory_extracted filter
        discrete_filter = DiscreteMemoryExtracted(eq="t")
        results = await adapter.search_memories(
            query="test query",
            discrete_memory_extracted=discrete_filter,
        )

        # Verify search was called
        mock_vectorstore.asimilarity_search_with_relevance_scores.assert_called_once()

        # Verify results
        assert len(results.memories) == 1
        assert results.memories[0].discrete_memory_extracted == "t"

    @pytest.mark.asyncio
    async def test_update_then_search_integration(self):
        """Integration test: update memories and verify they can be found with new values."""
        # This test simulates the real scenario where memories are updated
        # and then searched to verify the update worked

        # Create a mock VectorStore that tracks state
        class MockVectorStoreWithState:
            def __init__(self):
                self.documents = {}

            async def aadd_documents(self, documents, ids=None):
                if ids:
                    for doc, doc_id in zip(documents, ids, strict=False):
                        self.documents[doc_id] = doc
                else:
                    for i, doc in enumerate(documents):
                        self.documents[f"doc_{i}"] = doc
                return list(self.documents.keys())[-len(documents) :]

            async def adelete(self, ids):
                for doc_id in ids:
                    self.documents.pop(doc_id, None)
                return True

            async def asimilarity_search_with_relevance_scores(
                self, query, k=4, filter=None, **kwargs
            ):
                # Simple mock search that returns documents matching filter
                results = []
                for _key, doc in self.documents.items():
                    # Check if document matches discrete_memory_extracted filter
                    if filter and hasattr(filter, "get"):
                        filter_value = filter.get("discrete_memory_extracted")
                        if (
                            filter_value
                            and doc.metadata.get("discrete_memory_extracted")
                            == filter_value
                        ):
                            results.append((doc, 0.9))
                    elif not filter:
                        results.append((doc, 0.9))
                return results[:k]

        mock_vectorstore = MockVectorStoreWithState()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # First, add some memories with discrete_memory_extracted='f'
        original_memories = [
            MemoryRecord(
                text="Unprocessed memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.MESSAGE,
                discrete_memory_extracted="f",
            ),
            MemoryRecord(
                text="Unprocessed memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.MESSAGE,
                discrete_memory_extracted="f",
            ),
        ]

        # Add original memories
        await adapter.add_memories(original_memories)

        # Verify we can find memories with discrete_memory_extracted='f'
        from agent_memory_server.filters import DiscreteMemoryExtracted

        # Mock the filter conversion for this test
        with patch.object(
            adapter, "_convert_filters_to_backend_format"
        ) as mock_convert:
            mock_convert.return_value = {"discrete_memory_extracted": "f"}

            unprocessed_results = await adapter.search_memories(
                query="",
                discrete_memory_extracted=DiscreteMemoryExtracted(eq="f"),
            )

            assert len(unprocessed_results.memories) == 2

        # Now update the memories to mark them as processed
        updated_memories = []
        for memory in original_memories:
            updated_memory = memory.model_copy(
                update={"discrete_memory_extracted": "t"}
            )
            updated_memories.append(updated_memory)

        # Update the memories
        update_count = await adapter.update_memories(updated_memories)
        assert update_count == 2

        # Verify we can now find memories with discrete_memory_extracted='t'
        with patch.object(
            adapter, "_convert_filters_to_backend_format"
        ) as mock_convert:
            mock_convert.return_value = {"discrete_memory_extracted": "t"}

            processed_results = await adapter.search_memories(
                query="",
                discrete_memory_extracted=DiscreteMemoryExtracted(eq="t"),
            )

            assert len(processed_results.memories) == 2
            for result in processed_results.memories:
                assert result.discrete_memory_extracted == "t"

        # Verify we can no longer find memories with discrete_memory_extracted='f'
        with patch.object(
            adapter, "_convert_filters_to_backend_format"
        ) as mock_convert:
            mock_convert.return_value = {"discrete_memory_extracted": "f"}

            unprocessed_results_after = await adapter.search_memories(
                query="",
                discrete_memory_extracted=DiscreteMemoryExtracted(eq="f"),
            )

            assert len(unprocessed_results_after.memories) == 0

    def test_redis_adapter_preserves_discrete_memory_extracted_flag(self):
        """Regression test: Ensure Redis adapter preserves discrete_memory_extracted='t' during search.

        This test catches the bug where MCP-created memories with discrete_memory_extracted='t'
        were being returned as 'f' because the Redis vector store adapter wasn't populating
        the field during document-to-memory conversion.
        """
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        # Create mock vectorstore and embeddings
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create Redis adapter
        adapter = RedisVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Mock document that simulates what Redis returns for an MCP-created memory
        mock_doc = MagicMock()
        mock_doc.page_content = "User likes green tea"
        mock_doc.metadata = {
            "id_": "memory_001",
            "session_id": None,
            "user_id": None,
            "namespace": "user_preferences",
            "created_at": datetime.now(UTC).timestamp(),
            "updated_at": datetime.now(UTC).timestamp(),
            "last_accessed": datetime.now(UTC).timestamp(),
            "topics": "preferences,beverages",
            "entities": "",
            "memory_hash": "abc123",
            "discrete_memory_extracted": "t",  # This should be preserved!
            "memory_type": "semantic",
            "persisted_at": None,
            "extracted_from": "",
            "event_date": None,
        }

        # Mock the search to return our test document
        mock_vectorstore.asimilarity_search_with_relevance_scores = AsyncMock(
            return_value=[(mock_doc, 0.9)]
        )

        # Perform search
        result = asyncio.run(
            adapter.search_memories(
                query="green tea",
                namespace=Namespace(field="namespace", eq="user_preferences"),
                limit=10,
            )
        )

        # Verify we got the memory back
        assert len(result.memories) == 1
        memory = result.memories[0]

        # REGRESSION TEST: This should be 't', not 'f'
        assert memory.discrete_memory_extracted == "t", (
            f"Regression: Expected discrete_memory_extracted='t', got '{memory.discrete_memory_extracted}'. "
            f"This indicates the Redis adapter is not preserving the flag during search."
        )

        # Also verify other expected properties
        assert memory.memory_type.value == "semantic"
        assert memory.namespace == "user_preferences"
        assert memory.text == "User likes green tea"


class TestCreateEmbeddings:
    """Test cases for the create_embeddings function."""

    def test_create_embeddings_aws_bedrock_success(self):
        """Test creating AWS Bedrock embeddings successfully."""
        mock_model_config = MagicMock()
        mock_model_config.provider = "aws-bedrock"

        # Create mock modules for langchain_aws
        mock_langchain_aws = MagicMock()
        mock_bedrock_embeddings_class = MagicMock()
        mock_langchain_aws.BedrockEmbeddings = mock_bedrock_embeddings_class

        mock_embeddings_instance = MagicMock()
        mock_bedrock_embeddings_class.return_value = mock_embeddings_instance

        # Mock the _aws modules
        mock_create_client = MagicMock()
        mock_model_exists = MagicMock(return_value=True)
        mock_aws_clients = MagicMock()
        mock_aws_utils = MagicMock()
        mock_aws_clients.create_bedrock_client = mock_create_client
        mock_aws_utils.bedrock_embedding_model_exists = mock_model_exists

        with (
            patch("agent_memory_server.vectorstore_factory.settings") as mock_settings,
            patch.dict(
                sys.modules,
                {
                    "langchain_aws": mock_langchain_aws,
                    "agent_memory_server._aws.clients": mock_aws_clients,
                    "agent_memory_server._aws.utils": mock_aws_utils,
                },
            ),
        ):
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "amazon.titan-embed-text-v2:0"
            mock_settings.aws_region = "us-east-1"

            mock_client = MagicMock()
            mock_create_client.return_value = mock_client

            result = create_embeddings()

            assert result == mock_embeddings_instance
            mock_create_client.assert_called_once()
            mock_model_exists.assert_called_once_with(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

    def test_create_embeddings_aws_bedrock_model_not_found(self):
        """Test error when Bedrock embedding model doesn't exist."""
        mock_model_config = MagicMock()
        mock_model_config.provider = "aws-bedrock"

        # Create mock module for langchain_aws (needed for import)
        mock_langchain_aws = MagicMock()
        mock_model_exists = MagicMock(return_value=False)
        mock_aws_clients = MagicMock()
        mock_aws_utils = MagicMock()
        mock_aws_utils.bedrock_embedding_model_exists = mock_model_exists

        with (
            patch("agent_memory_server.vectorstore_factory.settings") as mock_settings,
            patch.dict(
                sys.modules,
                {
                    "langchain_aws": mock_langchain_aws,
                    "agent_memory_server._aws.clients": mock_aws_clients,
                    "agent_memory_server._aws.utils": mock_aws_utils,
                },
            ),
        ):
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "invalid-model-id"
            mock_settings.aws_region = "us-east-1"

            with pytest.raises(ValueError) as exc_info:
                create_embeddings()

            assert "invalid-model-id" in str(exc_info.value)
            assert "not found" in str(exc_info.value)

    def test_create_embeddings_aws_bedrock_import_error(self):
        """Test error when AWS dependencies are not installed."""
        mock_model_config = MagicMock()
        mock_model_config.provider = "aws-bedrock"

        with (
            patch("agent_memory_server.vectorstore_factory.settings") as mock_settings,
            patch.dict(sys.modules, {"langchain_aws": None}),  # Simulate missing module
        ):
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "amazon.titan-embed-text-v2:0"

            with pytest.raises(ImportError):
                create_embeddings()
