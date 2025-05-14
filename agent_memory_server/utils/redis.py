"""Redis utility functions."""

import logging
from typing import Any

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)
_redis_pool: Redis | None = None
_index: AsyncSearchIndex | None = None


async def get_redis_conn(url: str = settings.redis_url, **kwargs) -> Redis:
    """Get a Redis connection.

    Args:
        url: Redis connection URL, or None to use settings.redis_url
        **kwargs: Additional arguments to pass to Redis.from_url

    Returns:
        A Redis client instance
    """
    global _redis_pool

    # Always use the existing _redis_pool if it's not None, regardless of the URL parameter
    # This ensures that the patched _redis_pool from the test fixture is used
    if _redis_pool is None:
        _redis_pool = Redis.from_url(url, **kwargs)
    return _redis_pool


def get_search_index(
    redis: Redis,
    index_name: str = settings.redisvl_index_name,
    vector_dimensions: str = settings.redisvl_vector_dimensions,
    distance_metric: str = settings.redisvl_distance_metric,
) -> AsyncSearchIndex:
    global _index
    if _index is None:
        schema = {
            "index": {
                "name": index_name,
                "prefix": f"{index_name}:",
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "memory_hash", "type": "tag"},
                {"name": "id_", "type": "tag"},
                {"name": "session_id", "type": "tag"},
                {"name": "user_id", "type": "tag"},
                {"name": "namespace", "type": "tag"},
                {"name": "topics", "type": "tag"},
                {"name": "entities", "type": "tag"},
                {"name": "created_at", "type": "numeric"},
                {"name": "last_accessed", "type": "numeric"},
                {"name": "memory_type", "type": "tag"},
                {"name": "discrete_memory_extracted", "type": "tag"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "HNSW",
                        "dims": vector_dimensions,
                        "distance_metric": distance_metric,
                        "datatype": "float32",
                    },
                },
            ],
        }
        index_schema = IndexSchema.from_dict(schema)
        _index = AsyncSearchIndex(index_schema, redis_client=redis)
    return _index


async def ensure_search_index_exists(
    redis: Redis,
    index_name: str = settings.redisvl_index_name,
    vector_dimensions: str = settings.redisvl_vector_dimensions,
    distance_metric: str = settings.redisvl_distance_metric,
    overwrite: bool = False,
) -> None:
    """
    Ensure that the async search index exists, create it if it doesn't.
    Uses RedisVL's AsyncSearchIndex.

    Args:
        redis: A Redis client instance
        vector_dimensions: Dimensions of the embedding vectors
        distance_metric: Distance metric to use (default: COSINE)
        index_name: The name of the index
    """
    index = get_search_index(redis, index_name, vector_dimensions, distance_metric)
    if await index.exists():
        logger.info("Async search index already exists")
        if overwrite:
            logger.info("Overwriting existing index")
            await redis.execute_command("FT.DROPINDEX", index.name)
        else:
            return
    else:
        logger.info("Async search index doesn't exist, creating...")

    await index.create()

    logger.info(
        f"Created async search index with {vector_dimensions} dimensions and {distance_metric} metric"
    )


def safe_get(doc: Any, key: str, default: Any | None = None) -> Any:
    """Get a value from a Document, returning a default if the key is not present.

    Args:
        doc: Document or object to get a value from
        key: Key to get
        default: Default value to return if key is not found

    Returns:
        The value if found, or the default
    """
    if isinstance(doc, dict):
        return doc.get(key, default)
    try:
        return getattr(doc, key)
    except (AttributeError, KeyError):
        return default
