"""
Tests for memory vector database factory patterns from documentation.

Focuses on testing the factory logic without external dependencies.
"""

import json
import os

import pytest


class MockEmbeddings:
    """Simple mock embeddings for testing."""

    def __init__(self):
        self._dimensions = 3
        self.model = "mock-embedding-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class MockBackend:
    """Mock backend for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestFactoryPatterns:
    """Test factory patterns without external dependencies."""

    def test_basic_factory_pattern(self):
        """Test the basic factory pattern."""

        def create_mock_backend(embeddings):
            """Factory function that creates a mock backend."""
            return MockBackend(
                collection_name="agent_memory",
                persist_directory="./data",
                embedding_function=embeddings,
            )

        embeddings = MockEmbeddings()
        result = create_mock_backend(embeddings)

        assert result.collection_name == "agent_memory"
        assert result.persist_directory == "./data"
        assert result.embedding_function == embeddings

    def test_environment_configuration_pattern(self):
        """Test environment-based configuration pattern."""

        config = {
            "collection_name": "test_memories",
            "persist_directory": "./test_data",
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("MEMORY_VECTOR_DB_CONFIG", json.dumps(config))
            mp.setenv("BACKEND_TYPE", "mock")

            def create_configured_backend(embeddings):
                """Factory that reads configuration from environment."""
                config = json.loads(os.getenv("MEMORY_VECTOR_DB_CONFIG", "{}"))
                backend_type = os.getenv("BACKEND_TYPE", "chroma")

                if backend_type == "mock":
                    return MockBackend(
                        collection_name=config.get("collection_name", "default"),
                        persist_directory=config.get("persist_directory", "./default"),
                        embedding_function=embeddings,
                    )
                raise ValueError(f"Unsupported backend: {backend_type}")

            embeddings = MockEmbeddings()
            result = create_configured_backend(embeddings)

            assert result.collection_name == "test_memories"
            assert result.persist_directory == "./test_data"
            assert result.embedding_function == embeddings

    def test_multi_environment_factory(self):
        """Test multi-environment factory pattern."""

        def create_adaptive_backend(embeddings):
            """Dynamically choose backend based on environment."""

            environment = os.getenv("ENVIRONMENT", "development")

            if environment == "production":
                return MockBackend(
                    backend_type="production",
                    index_name="prod-memories",
                    embeddings=embeddings,
                )
            if environment == "staging":
                return MockBackend(
                    backend_type="staging",
                    index_name="staging-memories",
                    embeddings=embeddings,
                )
            return MockBackend(
                backend_type="development",
                persist_directory="./dev_data",
                embeddings=embeddings,
            )

        embeddings = MockEmbeddings()

        # Test development environment (default)
        result_dev = create_adaptive_backend(embeddings)
        assert result_dev.backend_type == "development"
        assert hasattr(result_dev, "persist_directory")

        # Test staging environment
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("ENVIRONMENT", "staging")
            result_staging = create_adaptive_backend(embeddings)
            assert result_staging.backend_type == "staging"
            assert result_staging.index_name == "staging-memories"

        # Test production environment
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("ENVIRONMENT", "production")
            result_prod = create_adaptive_backend(embeddings)
            assert result_prod.backend_type == "production"
            assert result_prod.index_name == "prod-memories"

    def test_resilient_factory_pattern(self):
        """Test resilient factory with fallback pattern."""

        def create_resilient_backend(embeddings):
            """Create backend with built-in resilience patterns."""

            # Try multiple backends in order of preference
            backend_preferences = [
                ("primary", _create_primary_backend),
                ("secondary", _create_secondary_backend),
                ("fallback", _create_fallback_backend),
            ]

            last_error = None
            for backend_name, factory_func in backend_preferences:
                try:
                    backend = factory_func(embeddings)
                    backend.selected_backend = backend_name
                    return backend
                except Exception as e:
                    last_error = e
                    continue

            raise Exception(f"All backends failed. Last error: {last_error}")

        def _create_primary_backend(embeddings):
            """Primary backend that fails."""
            raise ConnectionError("Primary backend unavailable")

        def _create_secondary_backend(embeddings):
            """Secondary backend that works."""
            return MockBackend(backend_type="secondary", embeddings=embeddings)

        def _create_fallback_backend(embeddings):
            """Fallback backend."""
            return MockBackend(backend_type="fallback", embeddings=embeddings)

        embeddings = MockEmbeddings()

        # Should fall back to secondary when primary fails
        result = create_resilient_backend(embeddings)
        assert result.selected_backend == "secondary"
        assert result.backend_type == "secondary"

    def test_resilient_factory_all_fail(self):
        """Test resilient factory when all backends fail."""

        def create_failing_backend(embeddings):
            backend_preferences = [
                ("first", lambda e: _fail("First failed")),
                ("second", lambda e: _fail("Second failed")),
                ("third", lambda e: _fail("Third failed")),
            ]

            last_error = None
            for _backend_name, factory_func in backend_preferences:
                try:
                    return factory_func(embeddings)
                except Exception as e:
                    last_error = e
                    continue

            raise Exception(f"All backends failed. Last error: {last_error}")

        def _fail(message):
            raise RuntimeError(message)

        embeddings = MockEmbeddings()

        with pytest.raises(Exception, match="All backends failed"):
            create_failing_backend(embeddings)


class TestHybridPattern:
    """Test the hybrid backend pattern."""

    def test_hybrid_routing_logic(self):
        """Test the routing logic of the hybrid pattern."""

        class SimpleHybridStore:
            """Simplified hybrid store for testing routing logic."""

            def __init__(self, embeddings):
                self.embeddings = embeddings
                self.fast_store_items = []
                self.archive_store_items = []

            def add_texts(
                self, texts: list[str], metadatas: list[dict] = None, **kwargs
            ):
                """Route texts based on metadata."""
                if not metadatas:
                    metadatas = [{}] * len(texts)

                results = []
                for text, meta in zip(texts, metadatas, strict=True):
                    if self._should_use_fast_store(meta):
                        item_id = f"fast_{len(self.fast_store_items)}"
                        self.fast_store_items.append(
                            {"id": item_id, "text": text, "meta": meta}
                        )
                        results.append(item_id)
                    else:
                        item_id = f"archive_{len(self.archive_store_items)}"
                        self.archive_store_items.append(
                            {"id": item_id, "text": text, "meta": meta}
                        )
                        results.append(item_id)

                return results

            def _should_use_fast_store(self, metadata: dict) -> bool:
                """Determine routing based on access count."""
                access_count = metadata.get("access_count", 0)
                return access_count > 5

        embeddings = MockEmbeddings()
        hybrid_store = SimpleHybridStore(embeddings)

        # Test routing
        texts = ["high access text", "low access text"]
        metadatas = [
            {"access_count": 10},  # Should go to fast store
            {"access_count": 2},  # Should go to archive store
        ]

        results = hybrid_store.add_texts(texts, metadatas)

        # Verify routing worked
        assert len(hybrid_store.fast_store_items) == 1
        assert len(hybrid_store.archive_store_items) == 1
        assert hybrid_store.fast_store_items[0]["text"] == "high access text"
        assert hybrid_store.archive_store_items[0]["text"] == "low access text"
        assert "fast_" in results[0]
        assert "archive_" in results[1]


class TestErrorHandling:
    """Test error handling patterns."""

    def test_configuration_validation(self):
        """Test configuration validation patterns."""

        def create_validated_backend(embeddings):
            """Factory with configuration validation."""

            required_config = os.getenv("REQUIRED_CONFIG")
            if not required_config:
                raise ValueError("REQUIRED_CONFIG environment variable is required")

            try:
                config = json.loads(required_config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in REQUIRED_CONFIG: {e}") from e

            if "collection_name" not in config:
                raise ValueError("collection_name is required in configuration")

            # Mock successful creation
            return MockBackend(
                collection_name=config["collection_name"], embeddings=embeddings
            )

        embeddings = MockEmbeddings()

        # Test missing config
        with pytest.raises(
            ValueError, match="REQUIRED_CONFIG environment variable is required"
        ):
            create_validated_backend(embeddings)

        # Test invalid JSON
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("REQUIRED_CONFIG", "invalid json")
            with pytest.raises(ValueError, match="Invalid JSON in REQUIRED_CONFIG"):
                create_validated_backend(embeddings)

        # Test missing required field
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("REQUIRED_CONFIG", '{"other_field": "value"}')
            with pytest.raises(ValueError, match="collection_name is required"):
                create_validated_backend(embeddings)

        # Test valid config
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("REQUIRED_CONFIG", '{"collection_name": "test_collection"}')
            result = create_validated_backend(embeddings)
            assert result.collection_name == "test_collection"

    def test_dependency_handling(self):
        """Test handling of missing dependencies."""

        def create_backend_with_dependency_check(embeddings):
            """Factory that checks for dependencies."""

            try:
                # Try to import optional dependency
                import nonexistent_library  # This will fail

                return nonexistent_library.create_store(embeddings)
            except ImportError:
                # Fall back to a different implementation
                return MockBackend(fallback_used=True, embeddings=embeddings)

        embeddings = MockEmbeddings()
        result = create_backend_with_dependency_check(embeddings)

        # Should have fallen back successfully
        assert result.fallback_used is True

    def test_connection_validation(self):
        """Test connection validation patterns."""

        def create_backend_with_connection_test(embeddings):
            """Factory that validates connections."""

            def test_connection():
                # Simulate connection failure based on environment
                if os.getenv("SIMULATE_CONNECTION_FAILURE") == "true":
                    raise ConnectionError("Cannot connect to backend")
                return True

            # Test the connection during factory creation
            try:
                test_connection()
                return MockBackend(connection_tested=True, embeddings=embeddings)
            except ConnectionError as e:
                raise ConnectionError(f"Backend connection failed: {e}") from e

        embeddings = MockEmbeddings()

        # Test successful connection
        result = create_backend_with_connection_test(embeddings)
        assert result.connection_tested is True

        # Test connection failure
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("SIMULATE_CONNECTION_FAILURE", "true")
            with pytest.raises(ConnectionError, match="Backend connection failed"):
                create_backend_with_connection_test(embeddings)


class TestReturnTypes:
    """Test that factories return the correct types."""

    def test_factory_return_type(self):
        """Test factory returning a backend object."""

        def create_mock_db_factory(embeddings):
            """Factory returning a mock backend."""
            return MockBackend(embeddings=embeddings)

        embeddings = MockEmbeddings()
        result = create_mock_db_factory(embeddings)

        # Should have embeddings attribute
        assert result.embeddings == embeddings

    def test_invalid_return_type(self):
        """Test handling of invalid return types."""

        def create_invalid_factory(embeddings):
            """Factory returning invalid type."""
            return "this is not a database"

        embeddings = MockEmbeddings()
        result = create_invalid_factory(embeddings)

        # Should return string (invalid), which would be caught by factory system
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__])
