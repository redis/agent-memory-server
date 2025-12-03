"""VectorStore factory for creating backend instances.

This module provides a minimal, flexible factory approach where users can specify
their own vectorstore initialization function using Python dotted notation.

The factory function should have signature:
    (embeddings: Embeddings) -> Union[VectorStore, VectorStoreAdapter]

Examples:
    VECTORSTORE_FACTORY="my_module.create_chroma_vectorstore"
    VECTORSTORE_FACTORY="my_package.adapters.CustomAdapter.create"
    VECTORSTORE_FACTORY="agent_memory_server.vectorstore_factory.create_redis_vectorstore"

Benefits:
- No database-specific code in this codebase
- Users have complete flexibility to configure any vectorstore
- Dynamic imports avoid loading unnecessary dependencies
- Supports both VectorStore and VectorStoreAdapter return types
"""

import importlib
import logging

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_redis.config import RedisConfig
from pydantic.types import SecretStr

# RedisVL uses the same python-ulid library as this project, so no patching needed
from agent_memory_server.config import ModelProvider, settings
from agent_memory_server.vectorstore_adapter import (
    LangChainVectorStoreAdapter,
    MemoryRedisVectorStore,
    RedisVectorStoreAdapter,
    VectorStoreAdapter,
)


logger = logging.getLogger(__name__)


def create_embeddings() -> Embeddings:
    """Create an embeddings instance based on configuration.

    Returns:
        An Embeddings instance
    """
    embedding_config = settings.embedding_model_config
    # Only support ModelConfig objects
    provider = embedding_config.provider if embedding_config else "openai"

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            if settings.openai_api_key is not None:
                api_key = SecretStr(settings.openai_api_key)
                return OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=api_key,
                )
            # Default: handle API key from environment
            return OpenAIEmbeddings(
                model=settings.embedding_model,
            )
        except ImportError:
            logger.error(
                "langchain-openai not installed. Install with: pip install langchain-openai"
            )
            raise
        except Exception as e:
            logger.error(f"Error creating OpenAI embeddings: {e}")
            raise

    elif provider == "anthropic":
        # Note: Anthropic doesn't currently provide embedding models
        # Fall back to OpenAI embeddings for now
        logger.warning(
            f"Anthropic embedding model '{settings.embedding_model}' specified, "
            "but Anthropic doesn't provide embedding models. Falling back to OpenAI text-embedding-3-small."
        )
        try:
            from langchain_openai import OpenAIEmbeddings

            if settings.openai_api_key is not None:
                api_key = SecretStr(settings.openai_api_key)
                return OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=api_key,
                )
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
            )
        except ImportError:
            logger.error(
                "langchain-openai not installed. Install with: pip install langchain-openai"
            )
            raise
        except Exception as e:
            logger.error(f"Error creating fallback OpenAI embeddings: {e}")
            raise

    elif provider == "aws-bedrock":
        try:
            from langchain_aws import BedrockEmbeddings

            from agent_memory_server._aws.clients import create_bedrock_runtime_client
            from agent_memory_server._aws.utils import bedrock_embedding_model_exists
        except ImportError:
            err_msg: str = (
                "AWS-related dependencies might be missing. "
                "Try to install with: pip install agent-memory-server[aws]."
            )
            logger.exception(err_msg)
            raise

        # Instantiation-time check to catch misconfigurations early
        bedrock_model_id: str = settings.embedding_model
        if not bedrock_embedding_model_exists(
            bedrock_model_id,
            region_name=settings.aws_region,
        ):
            err_msg = f"Bedrock embedding model {bedrock_model_id} not found in region {settings.aws_region}."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Create a bedrock-runtime client (not bedrock control plane)
        # BedrockEmbeddings uses bedrock-runtime for actual inference
        bedrock_runtime_client = create_bedrock_runtime_client()
        return BedrockEmbeddings(model_id=bedrock_model_id, client=bedrock_runtime_client)

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: {', '.join(ModelProvider.__members__.keys())}. "
            f"Provider '{ModelProvider.ANTHROPIC}' falls back to '{ModelProvider.OPENAI}'."
        )


def _import_and_call_factory(
    factory_path: str, embeddings: Embeddings
) -> VectorStore | VectorStoreAdapter:
    """Import and call a user-specified factory function.

    Args:
        factory_path: Python dotted path to factory function
        embeddings: Embeddings instance to pass to factory

    Returns:
        VectorStore or VectorStoreAdapter instance

    Raises:
        ImportError: If the module or function cannot be imported
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
        if not isinstance(result, VectorStore | VectorStoreAdapter):
            raise TypeError(
                f"Factory function {factory_path} must return VectorStore or VectorStoreAdapter, "
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


def create_redis_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Redis VectorStore instance using LangChain Redis.

    This is the default factory function for Redis backends.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Redis VectorStore instance
    """
    try:
        # Define metadata schema to match our existing schema
        metadata_schema = [
            {"name": "session_id", "type": "tag"},
            {"name": "user_id", "type": "tag"},
            {"name": "namespace", "type": "tag"},
            {"name": "memory_type", "type": "tag"},
            {"name": "topics", "type": "tag"},
            {"name": "entities", "type": "tag"},
            {"name": "memory_hash", "type": "tag"},
            {"name": "discrete_memory_extracted", "type": "tag"},
            {"name": "pinned", "type": "tag"},
            {"name": "access_count", "type": "numeric"},
            {"name": "created_at", "type": "numeric"},
            {"name": "last_accessed", "type": "numeric"},
            {"name": "updated_at", "type": "numeric"},
            {"name": "persisted_at", "type": "numeric"},
            {"name": "event_date", "type": "numeric"},
            {"name": "extracted_from", "type": "tag"},
            {"name": "id_", "type": "tag"},
        ]

        # Always use MemoryRedisVectorStore for consistency and to fix relevance score issues
        return MemoryRedisVectorStore(
            embeddings=embeddings,
            config=RedisConfig(
                redis_url=settings.redis_url,
                key_prefix=settings.redisvl_index_prefix,
                indexing_algorithm=settings.redisvl_indexing_algorithm,
                index_name=settings.redisvl_index_name,
                metadata_schema=metadata_schema,
                distance_metric=settings.redisvl_distance_metric,
                embedding_dimensions=int(settings.redisvl_vector_dimensions),
            ),
        )
    except ImportError:
        logger.error(
            "langchain-redis not installed. Install with: pip install langchain-redis"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating Redis VectorStore: {e}")
        raise


def create_vectorstore_adapter() -> VectorStoreAdapter:
    """Create a VectorStore adapter using the configured factory function.

    Returns:
        A VectorStoreAdapter instance configured for the selected backend
    """
    embeddings = create_embeddings()
    factory_path = settings.vectorstore_factory

    logger.info(f"Creating VectorStore using factory: {factory_path}")

    # Call user-specified factory function
    result = _import_and_call_factory(factory_path, embeddings)

    # If the result is already a VectorStoreAdapter, use it directly
    if isinstance(result, VectorStoreAdapter):
        logger.info("Factory returned VectorStoreAdapter directly")
        return result

    # If the result is a VectorStore, wrap it in appropriate adapter
    if isinstance(result, VectorStore):
        logger.info("Factory returned VectorStore, wrapping in adapter")

        # Special handling for Redis - use Redis-specific adapter
        if factory_path.endswith("create_redis_vectorstore"):
            # Use the actual Redis VectorStore returned by the factory
            adapter = RedisVectorStoreAdapter(result, embeddings)
        else:
            # For all other backends, use generic LangChain adapter
            adapter = LangChainVectorStoreAdapter(result, embeddings)

        logger.info("VectorStore adapter created successfully")
        return adapter

    # Should never reach here due to type validation in _import_and_call_factory
    raise TypeError(f"Unexpected return type from factory: {type(result)}")


# Global adapter instance
_adapter: VectorStoreAdapter | None = None


async def get_vectorstore_adapter() -> VectorStoreAdapter:
    """Get the global VectorStore adapter instance.

    Returns:
        The global VectorStoreAdapter instance
    """
    global _adapter

    if _adapter is None:
        _adapter = create_vectorstore_adapter()

    return _adapter
