"""
Integration tests for the memory vector database factory system.

Tests the real factory loading mechanism and Redis factory.
"""

import os
from unittest.mock import Mock, patch

import pytest

from agent_memory_server.config import ModelConfig, ModelProvider
from agent_memory_server.memory_vector_db_factory import (
    _import_and_call_factory,
    create_embeddings,
)


class MockEmbeddings:
    """Mock embeddings for testing."""

    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self._dimensions = dimensions
        self.model = "mock-embedding-model"

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]

    async def aembed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def aembed_query(self, text):
        return [0.1, 0.2, 0.3]


class TestFactoryLoading:
    """Test the factory loading mechanism."""

    def test_import_and_call_factory_import_error(self):
        """Test factory loading with import error."""

        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            embeddings = MockEmbeddings()

            with pytest.raises(ImportError):
                _import_and_call_factory("nonexistent.factory", embeddings)

    def test_import_and_call_factory_function_not_found(self):
        """Test factory loading when function doesn't exist."""

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            # Function doesn't exist on module
            if hasattr(mock_module, "nonexistent_function"):
                del mock_module.nonexistent_function
            mock_import.return_value = mock_module

            embeddings = MockEmbeddings()

            with pytest.raises(AttributeError):
                _import_and_call_factory("test_module.nonexistent_function", embeddings)

    def test_import_and_call_factory_invalid_return_type(self):
        """Test factory loading with invalid return type."""

        def invalid_factory(embeddings):
            return "not a memory vector database"  # Invalid return type

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.invalid_factory = invalid_factory
            mock_import.return_value = mock_module

            embeddings = MockEmbeddings()

            with pytest.raises(TypeError, match="must return MemoryVectorDatabase"):
                _import_and_call_factory("test_module.invalid_factory", embeddings)

    def test_import_and_call_factory_invalid_path(self):
        """Test factory loading with invalid module path."""

        embeddings = MockEmbeddings()

        with pytest.raises(ValueError, match="Invalid factory path"):
            _import_and_call_factory("invalid_path_no_dots", embeddings)


class TestEmbeddingsCreation:
    """Test embeddings creation."""

    @patch("agent_memory_server.config.settings")
    def test_create_openai_embeddings(self, mock_settings):
        """Test OpenAI embeddings creation returns LiteLLMEmbeddings."""
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        # Configure mock settings with ModelConfig object
        mock_settings.embedding_model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            name="text-embedding-3-small",
            max_tokens=8191,
            embedding_dimensions=1536,
        )
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_api_base = None

        result = create_embeddings()

        assert isinstance(result, LiteLLMEmbeddings)
        assert result.model == "text-embedding-3-small"
        assert result._dimensions == 1536

    @patch("agent_memory_server.config.settings")
    def test_create_embeddings_anthropic_raises_error(self, mock_settings):
        """Test embeddings creation with Anthropic raises error."""
        from agent_memory_server.llm import ModelValidationError

        # Create a mock model config with Anthropic provider
        mock_config = Mock()
        mock_config.provider = ModelProvider.ANTHROPIC
        mock_settings.embedding_model_config = mock_config
        mock_settings.embedding_model = "claude-embedding"

        with pytest.raises(
            ModelValidationError, match="Anthropic does not provide embedding"
        ):
            create_embeddings()


class TestDocumentationExamples:
    """Test that documentation examples work as expected."""

    def test_basic_factory_pattern(self):
        """Test the basic factory pattern from docs works."""

        def create_mock_backend(embeddings):
            """Factory function that creates a mock backend."""
            mock_store = Mock()
            mock_store.embeddings = embeddings
            mock_store.collection_name = "agent_memory"
            return mock_store

        embeddings = MockEmbeddings()
        result = create_mock_backend(embeddings)

        assert result.embeddings == embeddings
        assert result.collection_name == "agent_memory"

    def test_environment_config_pattern(self):
        """Test environment-based configuration pattern."""

        with patch.dict(
            os.environ,
            {
                "MEMORY_VECTOR_DB_CONFIG": '{"collection_name": "test_memories", "persist_directory": "./test_data"}',
                "BACKEND_TYPE": "mock",
            },
        ):

            def create_configured_backend(embeddings):
                """Factory that reads configuration from environment."""
                import json

                config = json.loads(os.getenv("MEMORY_VECTOR_DB_CONFIG", "{}"))
                backend_type = os.getenv("BACKEND_TYPE", "chroma")

                if backend_type == "mock":
                    mock_store = Mock()
                    mock_store.collection_name = config.get(
                        "collection_name", "default"
                    )
                    mock_store.persist_directory = config.get(
                        "persist_directory", "./default"
                    )
                    return mock_store
                raise ValueError(f"Unsupported backend: {backend_type}")

            embeddings = MockEmbeddings()
            result = create_configured_backend(embeddings)

            assert result.collection_name == "test_memories"
            assert result.persist_directory == "./test_data"


if __name__ == "__main__":
    pytest.main([__file__])
