import logging

from redis.asyncio import ConnectionPool, Redis
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from config import settings
from models import OpenAIClientWrapper, ModelClientFactory, AnthropicClientWrapper
from typing import Union


REDIS_INDEX_NAME = "memory"

logger = logging.getLogger(__name__)
_redis_pool = None
_openai_client = None
_anthropic_client = None
_model_clients = {}


def get_redis_conn(url: str | None = settings.redis_url, **kwargs) -> Redis:
    """Get Redis connection"""
    global _redis_pool
    if _redis_pool is None:
        if url:
            _redis_pool = ConnectionPool.from_url(url, **kwargs)
        else:
            _redis_pool = ConnectionPool(**kwargs)
    return Redis(connection_pool=_redis_pool)


async def ensure_redisearch_index(
    redis: Redis,
    vector_dimensions: int,
    distance_metric: str = "COSINE",
    index_name: str = REDIS_INDEX_NAME,
) -> None:
    """
    Ensure that the RediSearch index exists, create it if it doesn't.
    
    TODO: Replace with RedisVL index.

    Args:
        vector_dimensions: Dimensions of the embedding vectors
        distance_metric: Distance metric to use (default: COSINE)
    """
    try:
        # Check if index exists
        try:
            await redis.ft(index_name).info()
            logger.info("RediSearch index already exists")
            return
        except Exception as e:
            # If error contains "unknown: index name", then index doesn't exist
            if "unknown index name" in str(e).lower():
                logger.info("RediSearch index doesn't exist, creating...")

                schema = [
                    TagField(name="session"),
                    TextField(name="content"),
                    TagField(name="role"),
                    VectorField(
                        name="vector",
                        algorithm="HNSW",
                        attributes={
                            "TYPE": "FLOAT32",
                            "DIM": vector_dimensions,
                            "DISTANCE_METRIC": distance_metric,
                        },
                    ),
                ]
                index_def = IndexDefinition(
                    prefix=[f"{index_name}:"], index_type=IndexType.HASH
                )
                await redis.ft(index_name).create_index(
                    fields=schema,
                    definition=index_def,
                )
                logger.info(
                    f"Created RediSearch index with {vector_dimensions} dimensions and {distance_metric} metric"
                )
                return
            else:
                # This is an unexpected error
                raise
    except Exception as e:
        logger.error(f"Error ensuring RediSearch index: {e}")
        raise


async def get_openai_client(**kwargs) -> OpenAIClientWrapper:
    """Get OpenAI client (legacy function, use get_model_client instead)"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClientWrapper(api_key=settings.openai_api_key, **kwargs)
    return _openai_client


async def get_model_client(
    model_name: str,
) -> Union[OpenAIClientWrapper, AnthropicClientWrapper]:
    """Get the appropriate client for a model using the factory"""
    global _model_clients

    if model_name not in _model_clients:
        _model_clients[model_name] = await ModelClientFactory.get_client(model_name)

    return _model_clients[model_name]


class Keys:
    """Keys for Redis"""

    @staticmethod
    def session_key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def context_key(session_id: str) -> str:
        return f"context:{session_id}"

    @staticmethod
    def token_count_key(session_id: str) -> str:
        return f"tokens:{session_id}"

    @staticmethod
    def messages_key(session_id: str) -> str:
        return f"messages:{session_id}"

    @staticmethod
    def sessions_key() -> str:
        return "sessions"

    @staticmethod
    def memory_key(id: str) -> str:
        return f"memory:{id}"
