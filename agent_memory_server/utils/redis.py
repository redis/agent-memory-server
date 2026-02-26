"""Redis utility functions."""

import logging
import ssl
from typing import Any

from redis.asyncio import Redis

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)
_redis_pool: Redis | None = None


def build_redis_tls_kwargs(url: str, **kwargs) -> dict:
    """Build TLS connection kwargs based on URL scheme and settings.

    Applies TLS settings when using rediss:// URLs or when
    redis_ssl_ca_certs is explicitly configured.

    Args:
        url: Redis connection URL
        **kwargs: Base kwargs to extend

    Returns:
        Connection kwargs dict with TLS settings applied if needed
    """
    connection_kwargs = dict(kwargs)
    if url.startswith("rediss://") or settings.redis_ssl_ca_certs:
        if settings.redis_ssl_ca_certs:
            connection_kwargs.setdefault("ssl_ca_certs", settings.redis_ssl_ca_certs)
        if settings.redis_ssl_certfile:
            connection_kwargs.setdefault("ssl_certfile", settings.redis_ssl_certfile)
        if settings.redis_ssl_keyfile:
            connection_kwargs.setdefault("ssl_keyfile", settings.redis_ssl_keyfile)
        # Map string cert_reqs to ssl enum values
        cert_reqs_str = settings.redis_ssl_cert_reqs
        cert_reqs_map = {
            "required": ssl.CERT_REQUIRED,
            "optional": ssl.CERT_OPTIONAL,
            "none": ssl.CERT_NONE,
        }
        cert_reqs = cert_reqs_map.get(cert_reqs_str.lower(), ssl.CERT_REQUIRED)
        connection_kwargs.setdefault("ssl_cert_reqs", cert_reqs)
        min_ver = getattr(
            ssl.TLSVersion,
            settings.redis_ssl_min_version,
            ssl.TLSVersion.TLSv1_2,
        )
        connection_kwargs.setdefault("ssl_min_version", min_ver)
    return connection_kwargs


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
        connection_kwargs = build_redis_tls_kwargs(url, **kwargs)
        _redis_pool = Redis.from_url(url, **connection_kwargs)
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
