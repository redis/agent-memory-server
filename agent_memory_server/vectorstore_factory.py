"""VectorStore factory for creating backend instances.

This module provides factory functions to create VectorStore and Embeddings
instances based on configuration settings.
"""

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


# Monkey patch RedisVL ULID issue before importing anything else
try:
    import redisvl.utils.utils
    from ulid import ULID

    def patched_create_ulid() -> str:
        """Patched ULID creation function that works with python-ulid."""
        return str(ULID())  # Use ulid.new() instead of ULID()

    # Replace the broken function with our working one
    redisvl.utils.utils.create_ulid = patched_create_ulid
    logging.info("Successfully patched RedisVL ULID function")
except Exception as e:
    logging.warning(f"Could not patch RedisVL ULID function: {e}")
    # Continue anyway - might work if ULID issue is fixed elsewhere

from agent_memory_server.config import settings
from agent_memory_server.vectorstore_adapter import (
    LangChainVectorStoreAdapter,
    RedisVectorStoreAdapter,
    VectorStoreAdapter,
)


logger = logging.getLogger(__name__)


def create_embeddings() -> Embeddings:
    """Create an embeddings instance based on configuration.

    Returns:
        An Embeddings instance
    """
    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    except ImportError:
        logger.error(
            "langchain-openai not installed. Install with: pip install langchain-openai"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise


def create_chroma_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Chroma VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Chroma VectorStore instance
    """
    try:
        from langchain_chroma import Chroma

        if settings.chroma_persist_directory:
            # Persistent storage
            return Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=settings.chroma_persist_directory,
            )
        # HTTP client
        import chromadb

        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        return Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            client=client,
        )
    except ImportError:
        logger.error("chromadb not installed. Install with: pip install chromadb")
        raise
    except Exception as e:
        logger.error(f"Error creating Chroma VectorStore: {e}")
        raise


def create_pinecone_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Pinecone VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Pinecone VectorStore instance
    """
    try:
        from langchain_pinecone import PineconeVectorStore

        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key,
        )
    except ImportError:
        logger.error(
            "pinecone-client not installed. Install with: pip install pinecone-client"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating Pinecone VectorStore: {e}")
        raise


def create_weaviate_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Weaviate VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Weaviate VectorStore instance
    """
    try:
        import weaviate
        from langchain_weaviate import WeaviateVectorStore

        # Create Weaviate client
        if settings.weaviate_api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key=settings.weaviate_api_key)
            client = weaviate.Client(
                url=settings.weaviate_url, auth_client_secret=auth_config
            )
        else:
            client = weaviate.Client(url=settings.weaviate_url)

        return WeaviateVectorStore(
            client=client,
            index_name=settings.weaviate_class_name,
            text_key="text",
            embedding=embeddings,
        )
    except ImportError:
        logger.error(
            "weaviate-client not installed. Install with: pip install weaviate-client"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating Weaviate VectorStore: {e}")
        raise


def create_qdrant_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Qdrant VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Qdrant VectorStore instance
    """
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        # Create Qdrant client
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        return QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection_name,
            embeddings=embeddings,
        )
    except ImportError:
        logger.error(
            "qdrant-client not installed. Install with: pip install qdrant-client"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating Qdrant VectorStore: {e}")
        raise


def create_milvus_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Milvus VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Milvus VectorStore instance
    """
    try:
        from langchain_milvus import Milvus

        connection_args = {
            "host": settings.milvus_host,
            "port": settings.milvus_port,
        }

        if settings.milvus_user and settings.milvus_password:
            connection_args.update(
                {
                    "user": settings.milvus_user,
                    "password": settings.milvus_password,
                }
            )

        return Milvus(
            embedding_function=embeddings,
            collection_name=settings.milvus_collection_name,
            connection_args=connection_args,
        )
    except ImportError:
        logger.error("pymilvus not installed. Install with: pip install pymilvus")
        raise
    except Exception as e:
        logger.error(f"Error creating Milvus VectorStore: {e}")
        raise


def create_pgvector_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a PostgreSQL/PGVector VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A PGVector VectorStore instance
    """
    try:
        from langchain_postgres import PGVector

        if not settings.postgres_url:
            raise ValueError("postgres_url must be set for PGVector backend")

        return PGVector(
            embeddings=embeddings,
            connection=settings.postgres_url,
            collection_name=settings.postgres_table_name,
        )
    except ImportError:
        logger.error(
            "langchain-postgres not installed. Install with: pip install langchain-postgres psycopg2-binary"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating PGVector VectorStore: {e}")
        raise


def create_lancedb_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a LanceDB VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A LanceDB VectorStore instance
    """
    try:
        import lancedb
        from langchain_community.vectorstores import LanceDB

        # Create LanceDB connection
        db = lancedb.connect(settings.lancedb_uri)

        return LanceDB(
            connection=db,
            table_name=settings.lancedb_table_name,
            embedding=embeddings,
        )
    except ImportError:
        logger.error("lancedb not installed. Install with: pip install lancedb")
        raise
    except Exception as e:
        logger.error(f"Error creating LanceDB VectorStore: {e}")
        raise


def create_opensearch_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create an OpenSearch VectorStore instance.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        An OpenSearch VectorStore instance
    """
    try:
        from langchain_community.vectorstores import OpenSearchVectorSearch

        opensearch_kwargs = {
            "opensearch_url": settings.opensearch_url,
            "index_name": settings.opensearch_index_name,
        }

        if settings.opensearch_username and settings.opensearch_password:
            opensearch_kwargs.update(
                {
                    "http_auth": (
                        settings.opensearch_username,
                        settings.opensearch_password,
                    ),
                }
            )

        return OpenSearchVectorSearch(
            embedding_function=embeddings,
            **opensearch_kwargs,
        )
    except ImportError:
        logger.error(
            "opensearch-py not installed. Install with: pip install opensearch-py"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating OpenSearch VectorStore: {e}")
        raise


def create_vectorstore(backend: str, embeddings: Embeddings) -> VectorStore:
    """Create a VectorStore instance based on the backend type.

    Note: Redis is handled separately in create_vectorstore_adapter()
    and does not use this function.

    Args:
        backend: Backend type (chroma, pinecone, weaviate, etc.) - Redis excluded
        embeddings: Embeddings instance to use

    Returns:
        A VectorStore instance

    Raises:
        ValueError: If the backend type is not supported
    """
    backend = backend.lower()

    if backend == "redis":
        raise ValueError("Redis backend should use RedisVectorStoreAdapter directly")
    if backend == "chroma":
        return create_chroma_vectorstore(embeddings)
    if backend == "pinecone":
        return create_pinecone_vectorstore(embeddings)
    if backend == "weaviate":
        return create_weaviate_vectorstore(embeddings)
    if backend == "qdrant":
        return create_qdrant_vectorstore(embeddings)
    if backend == "milvus":
        return create_milvus_vectorstore(embeddings)
    if backend == "pgvector" or backend == "postgres":
        return create_pgvector_vectorstore(embeddings)
    if backend == "lancedb":
        return create_lancedb_vectorstore(embeddings)
    if backend == "opensearch":
        return create_opensearch_vectorstore(embeddings)
    raise ValueError(f"Unsupported backend: {backend}")


def create_redis_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create a Redis VectorStore instance using LangChain Redis.

    Args:
        embeddings: Embeddings instance to use

    Returns:
        A Redis VectorStore instance
    """
    try:
        from langchain_redis import RedisVectorStore

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
            {"name": "created_at", "type": "numeric"},
            {"name": "last_accessed", "type": "numeric"},
            {"name": "updated_at", "type": "numeric"},
            {"name": "persisted_at", "type": "numeric"},
            {"name": "event_date", "type": "numeric"},
            {"name": "extracted_from", "type": "tag"},
            {"name": "id", "type": "tag"},
        ]

        # Try to connect to existing index first
        try:
            return RedisVectorStore.from_existing_index(
                index_name=settings.redisvl_index_name,
                embeddings=embeddings,
                redis_url=settings.redis_url,
            )
        except Exception:
            # If no existing index, create a new one with metadata schema
            return RedisVectorStore(
                embeddings=embeddings,
                redis_url=settings.redis_url,
                index_name=settings.redisvl_index_name,
                metadata_schema=metadata_schema,
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
    """Create a VectorStore adapter based on configuration.

    Returns:
        A VectorStoreAdapter instance configured for the selected backend
    """
    backend = settings.long_term_memory_backend.lower()
    embeddings = create_embeddings()

    logger.info(f"Creating VectorStore adapter with backend: {backend}")

    # For Redis, use Redis-specific adapter without LangChain's RedisVectorStore
    # since we use pure RedisVL for all operations
    if backend == "redis":
        # Create a dummy vectorstore for interface compatibility
        # The RedisVectorStoreAdapter doesn't actually use this
        from langchain_core.vectorstores import VectorStore

        class DummyVectorStore(VectorStore):
            def add_texts(self, texts, metadatas=None, **kwargs):
                return []

            def similarity_search(self, query, k=4, **kwargs):
                return []

            @classmethod
            def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
                return cls()

        dummy_vectorstore = DummyVectorStore()
        adapter = RedisVectorStoreAdapter(dummy_vectorstore, embeddings)
    else:
        # For all other backends, use generic LangChain adapter
        vectorstore = create_vectorstore(backend, embeddings)
        adapter = LangChainVectorStoreAdapter(vectorstore, embeddings)

    logger.info("VectorStore adapter created successfully")
    return adapter


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
