"""Redis utility functions."""

import logging
from typing import Any

from redis.asyncio import Redis

from agent_memory_server.config import settings


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

    # Always use the existing _redis_pool if it's not None, regardless of the URL parameter
    # This ensures connection reuse and prevents multiple Redis connections
    if _redis_pool is None:
        _redis_pool = Redis.from_url(url, **kwargs)
    return _redis_pool


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
