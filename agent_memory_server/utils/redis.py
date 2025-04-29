"""Redis utility functions."""

import logging
from typing import Any

from redis.asyncio import Redis
from redis.commands.search.commands import SearchCommands
from redis.commands.search.field import (
    Field,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from agent_memory_server.config import settings
from agent_memory_server.utils.keys import Keys


logger = logging.getLogger(__name__)
_redis_pool: Redis | None = None


async def get_redis_conn(url: str = settings.redis_url, **kwargs) -> Redis:
    """Get a Redis connection.

    Args:
        url: Redis connection URL, or None to use settings.redis_url
        **kwargs: Additional arguments to pass to Redis.from_url

    Returns:
        A Redis client instance
    """
    global _redis_pool

    print(f"DEBUG: get_redis_conn called with url={url}")
    print(f"DEBUG: _redis_pool before check: {_redis_pool}")

    # Always use the existing _redis_pool if it's not None, regardless of the URL parameter
    # This ensures that the patched _redis_pool from the test fixture is used
    if _redis_pool is None:
        print(f"DEBUG: Creating new Redis connection with url={url}")
        _redis_pool = Redis.from_url(url, **kwargs)
        print(f"DEBUG: _redis_pool after creation: {_redis_pool}")
    else:
        print(f"DEBUG: Using existing _redis_pool: {_redis_pool}")

    print(f"DEBUG: Returning _redis_pool: {_redis_pool}")
    return _redis_pool


async def get_search_index(redis_client: Redis | None = None) -> SearchCommands:
    """Get the Redis search index.

    Args:
        redis_client: Redis client to use, or None to create a new one

    Returns:
        A Redis search index instance
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    return redis_client.ft(Keys.search_index_name())


async def ensure_search_index_exists(redis: Redis) -> None:
    """Ensure the search index exists

    Args:
        redis: Redis client to use

    Raises:
        Exception: If creating the index fails for a reason other than it already existing
    """
    index_name = Keys.search_index_name()
    print(f"DEBUG: ensure_search_index_exists called with index_name={index_name}")
    print(f"DEBUG: redis connection: {redis}")

    try:
        # First check if the index already exists
        try:
            print(f"DEBUG: Checking if index {index_name} exists")
            info = await redis.execute_command(f"FT.INFO {index_name}")
            print(f"DEBUG: Index {index_name} exists: {info}")
            return
        except Exception as e:
            if "unknown index name" not in str(e).lower():
                print(f"DEBUG: Error checking if index exists: {e}")
                raise
            print(f"DEBUG: Index {index_name} does not exist, creating it")

        schema: list[Field] = [
            TextField("text", weight=5.0),
            TextField("id_"),
            TextField("user_id"),
            TextField("session_id"),
            TextField("namespace"),
            NumericField("created_at", sortable=True),
            NumericField("last_accessed", sortable=True),
            TagField("topics", separator=","),
            TagField("entities", separator=","),
            TagField(
                "memory_hash"
            ),  # Add hash as a tag field for efficient search/aggregation
            VectorField(
                "vector",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"},
            ),
        ]

        definition = IndexDefinition(prefix=["memory:"], index_type=IndexType.HASH)
        print(f"DEBUG: Creating index {index_name} with prefix=['memory:']")
        await redis.ft(index_name).create_index(schema, definition=definition)
        print(f"DEBUG: Created index {index_name}")
        logger.info("Created search index")
    except Exception as e:
        if "Index already exists" in str(e):
            print(f"DEBUG: Index {index_name} already exists")
            logger.info("Search index already exists")
        else:
            print(f"DEBUG: Failed to create index {index_name}: {e}")
            logger.error(f"Failed to create search index: {e}")
            raise


def safe_get(doc: Any, key: str, default: Any | None = None) -> Any:
    """Get a value from a Document, returning a default if the key is not present.

    Args:
        doc: Document or object to get a value from
        key: Key to get
        default: Default value to return if key is not found

    Returns:
        The value if found, or the default
    """
    try:
        return getattr(doc, key)
    except (AttributeError, KeyError):
        return default
