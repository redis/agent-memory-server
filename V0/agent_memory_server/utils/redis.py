"""Redis utility functions."""

import logging
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)
_redis_pool: Redis | RedisCluster | None = None
_CLUSTER_SCHEMES = {"redis+cluster", "rediss+cluster"}


def _netloc_has_multiple_hosts(netloc: str) -> bool:
    host_part = netloc.rsplit("@", 1)[-1]
    return "," in host_part


def is_redis_cluster_url(url: str) -> bool:
    """Return True when the URL targets a Redis Cluster deployment."""
    parsed = urlparse(url)
    if parsed.scheme in _CLUSTER_SCHEMES:
        return True

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get("cluster", "").lower() == "true":
        return True

    return _netloc_has_multiple_hosts(parsed.netloc)


def _strip_cluster_query(url: str) -> str:
    parsed = urlparse(url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() != "cluster"
    ]
    return urlunparse(parsed._replace(query=urlencode(query_items, doseq=True)))


def redis_url_for_docket(url: str) -> str:
    """Normalize a Redis URL for Docket's cluster-aware URL scheme."""
    if not is_redis_cluster_url(url):
        return url

    parsed = urlparse(_strip_cluster_query(url))
    if parsed.scheme in _CLUSTER_SCHEMES:
        return urlunparse(parsed)

    return urlunparse(parsed._replace(scheme=f"{parsed.scheme}+cluster"))


def redis_url_for_redisvl(url: str) -> str:
    """Normalize a Redis URL for RedisVL's cluster detection."""
    if not is_redis_cluster_url(url):
        return url

    parsed = urlparse(_strip_cluster_query(url))
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    query_items.append(("cluster", "true"))
    return urlunparse(
        parsed._replace(
            scheme=parsed.scheme.replace("+cluster", ""),
            query=urlencode(query_items, doseq=True),
        )
    )


def redis_url_for_async_redis(url: str) -> str:
    """Normalize a Redis URL for redis-py's async standalone/cluster clients."""
    if not is_redis_cluster_url(url):
        return url

    parsed = urlparse(_strip_cluster_query(url))
    return urlunparse(parsed._replace(scheme=parsed.scheme.replace("+cluster", "")))


def docket_prefix(name: str, redis_url: str) -> str:
    """Return the Docket key prefix for the given deployment mode."""
    if is_redis_cluster_url(redis_url):
        return f"{{{name}}}"
    return name


def docket_stream_key(name: str, redis_url: str) -> str:
    """Return the Docket stream key with cluster-safe hashing when needed."""
    return f"{docket_prefix(name, redis_url)}:stream"


async def get_redis_conn(
    url: str = settings.redis_url, **kwargs
) -> Redis | RedisCluster:
    """Get a Redis connection.

    Args:
        url: Redis connection URL, or None to use settings.redis_url
        **kwargs: Additional arguments to pass to the Redis client

    Returns:
        A Redis or RedisCluster client instance
    """
    global _redis_pool

    # Always use the existing _redis_pool if it's not None, regardless of the URL parameter
    # This ensures connection reuse and prevents multiple Redis connections
    if _redis_pool is None:
        normalized_url = redis_url_for_async_redis(url)
        if is_redis_cluster_url(url):
            _redis_pool = RedisCluster.from_url(normalized_url, **kwargs)
        else:
            _redis_pool = Redis.from_url(normalized_url, **kwargs)
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
