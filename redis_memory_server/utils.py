import logging
import re

from redis.asyncio import ConnectionPool, Redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from regex import Pattern

from redis_memory_server.config import settings
from redis_memory_server.llms import (
    AnthropicClientWrapper,
    ModelClientFactory,
    OpenAIClientWrapper,
)


REDIS_INDEX_NAME = "memory"

logger = logging.getLogger(__name__)
_redis_pool = None
_openai_client = None
_anthropic_client = None
_model_clients = {}  # TODO: Use WeakRefDict


def get_redis_conn(url: str | None = settings.redis_url, **kwargs) -> Redis:
    """Get Redis connection"""
    global _redis_pool
    if _redis_pool is None:
        if url:
            _redis_pool = ConnectionPool.from_url(url, **kwargs)
            return Redis(connection_pool=_redis_pool)
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
                    TagField(name="topics", separator=","),
                    TagField(name="entities", separator=","),
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
) -> OpenAIClientWrapper | AnthropicClientWrapper:
    """Get the appropriate client for a model using the factory"""
    global _model_clients

    if model_name not in _model_clients:
        _model_clients[model_name] = await ModelClientFactory.get_client(model_name)

    return _model_clients[model_name]


class Keys:
    """Keys for Redis"""

    @staticmethod
    def context_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"context:{namespace}:{session_id}"
            if namespace
            else f"context:{session_id}"
        )

    @staticmethod
    def token_count_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"tokens:{namespace}:{session_id}" if namespace else f"tokens:{session_id}"
        )

    @staticmethod
    def messages_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"messages:{namespace}:{session_id}"
            if namespace
            else f"messages:{session_id}"
        )

    @staticmethod
    def sessions_key(namespace: str | None = None) -> str:
        return f"sessions:{namespace}" if namespace else "sessions"

    @staticmethod
    def memory_key(id: str, namespace: str | None = None) -> str:
        return f"memory:{namespace}:{id}" if namespace else f"memory:{id}"

    @staticmethod
    def metadata_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"metadata:{namespace}:{session_id}"
            if namespace
            else f"metadata:{session_id}"
        )


class TokenEscaper:
    """Escape punctuation within an input string.

    Adapted from RedisOM Python.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Pattern | None = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"Value must be a string object for token escaping, got type {type(value)}"
            )

        def escape_symbol(match):
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)
