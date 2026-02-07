"""Memory vector database factory for creating backend instances.

This module provides a minimal, flexible factory approach where users can specify
their own memory vector database initialization function using Python dotted notation.

The factory function should have signature:
    (embeddings: LiteLLMEmbeddings) -> MemoryVectorDatabase

Examples:
    MEMORY_VECTOR_DB_FACTORY="my_module.create_custom_memory_db"
    MEMORY_VECTOR_DB_FACTORY="my_package.adapters.CustomDB.create"
    MEMORY_VECTOR_DB_FACTORY="agent_memory_server.memory_vector_db_factory.create_redis_memory_vector_db"

Benefits:
- No database-specific code in this codebase beyond the default Redis implementation
- Users have complete flexibility to configure any backend
- Dynamic imports avoid loading unnecessary dependencies
- All factories must return MemoryVectorDatabase instances
"""

import importlib
import logging

from redisvl.index import AsyncSearchIndex

from agent_memory_server.config import settings
from agent_memory_server.llm import LLMClient
from agent_memory_server.llm.embeddings import LiteLLMEmbeddings
from agent_memory_server.memory_vector_db import (
    MemoryVectorDatabase,
    RedisVLMemoryVectorDatabase,
)


logger = logging.getLogger(__name__)


def create_embeddings() -> LiteLLMEmbeddings:
    """Create an embeddings instance based on configuration.

    Delegates to LLMClient.create_embeddings() which centralizes all
    embedding provider configuration.

    Returns:
        A LiteLLMEmbeddings instance
    """
    return LLMClient.create_embeddings()


def _import_and_call_factory(
    factory_path: str, embeddings: LiteLLMEmbeddings
) -> MemoryVectorDatabase:
    """Import and call a user-specified factory function.

    Args:
        factory_path: Python dotted path to factory function
        embeddings: Embeddings instance to pass to factory

    Returns:
        MemoryVectorDatabase instance

    Raises:
        ImportError: If the module or function cannot be imported
        TypeError: If the factory returns an invalid type
        Exception: If the factory function fails
    """
    try:
        # Split the path into module and function parts
        if "." not in factory_path:
            raise ValueError(
                f"Invalid factory path: {factory_path}. Must be in format 'module.function'"
            )

        module_path, function_name = factory_path.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the function
        factory_function = getattr(module, function_name)

        # Call the function with embeddings
        result = factory_function(embeddings)

        # Validate return type
        if not isinstance(result, MemoryVectorDatabase):
            raise TypeError(
                f"Factory function {factory_path} must return MemoryVectorDatabase, "
                f"got {type(result)}"
            )

        return result

    except ImportError as e:
        logger.error(f"Failed to import factory function {factory_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Function {function_name} not found in module {module_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling factory function {factory_path}: {e}")
        raise


def _get_embedding_dimensions() -> int:
    """Get the embedding dimensions from the configured embedding model.

    Returns the dimensions from the model config if available, otherwise
    falls back to the REDISVL_VECTOR_DIMENSIONS setting.
    """
    embedding_config = settings.embedding_model_config
    if embedding_config and embedding_config.embedding_dimensions:
        logger.info(
            f"Using embedding dimensions {embedding_config.embedding_dimensions} "
            f"from model config for {settings.embedding_model}"
        )
        return embedding_config.embedding_dimensions

    # Fall back to explicit setting
    logger.info(
        f"Using embedding dimensions {settings.redisvl_vector_dimensions} "
        f"from REDISVL_VECTOR_DIMENSIONS setting"
    )
    return int(settings.redisvl_vector_dimensions)


def _build_redis_schema() -> dict:
    """Build a RedisVL index schema dictionary from settings.

    Returns:
        A dictionary suitable for AsyncSearchIndex.from_dict()
    """
    embedding_dimensions = _get_embedding_dimensions()

    return {
        "index": {
            "name": settings.redisvl_index_name,
            "prefix": settings.redisvl_index_prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "session_id", "type": "tag"},
            {"name": "user_id", "type": "tag"},
            {"name": "namespace", "type": "tag"},
            {"name": "memory_type", "type": "tag"},
            {"name": "topics", "type": "tag"},
            {"name": "entities", "type": "tag"},
            {"name": "memory_hash", "type": "tag"},
            {"name": "discrete_memory_extracted", "type": "tag"},
            {"name": "pinned", "type": "tag"},
            {"name": "extracted_from", "type": "tag"},
            {"name": "id_", "type": "tag"},
            {"name": "access_count", "type": "numeric"},
            {"name": "created_at", "type": "numeric"},
            {"name": "last_accessed", "type": "numeric"},
            {"name": "updated_at", "type": "numeric"},
            {"name": "persisted_at", "type": "numeric"},
            {"name": "event_date", "type": "numeric"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": embedding_dimensions,
                    "distance_metric": settings.redisvl_distance_metric.lower(),
                    "algorithm": settings.redisvl_indexing_algorithm.lower(),
                    "datatype": "float32",
                },
            },
        ],
    }


def create_redis_memory_vector_db(
    embeddings: LiteLLMEmbeddings,
) -> MemoryVectorDatabase:
    """Create a Redis memory vector database instance using RedisVL directly.

    This is the default factory function for Redis backends.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A RedisVLMemoryVectorDatabase instance
    """
    try:
        schema = _build_redis_schema()
        index = AsyncSearchIndex.from_dict(schema, redis_url=settings.redis_url)
        return RedisVLMemoryVectorDatabase(index, embeddings)
    except Exception as e:
        logger.error(f"Error creating Redis memory vector database: {e}")
        raise


def create_memory_vector_db() -> MemoryVectorDatabase:
    """Create a memory vector database using the configured factory function.

    Returns:
        A MemoryVectorDatabase instance configured for the selected backend
    """
    embeddings = create_embeddings()
    factory_path = settings.memory_vector_db_factory

    logger.info(f"Creating memory vector database using factory: {factory_path}")

    # Call user-specified factory function
    result = _import_and_call_factory(factory_path, embeddings)

    logger.info("Memory vector database created successfully")
    return result


# Global memory vector database instance
_memory_vector_db: MemoryVectorDatabase | None = None


async def get_memory_vector_db() -> MemoryVectorDatabase:
    """Get the global memory vector database instance.

    Returns:
        The global MemoryVectorDatabase instance
    """
    global _memory_vector_db

    if _memory_vector_db is None:
        _memory_vector_db = create_memory_vector_db()

    return _memory_vector_db
