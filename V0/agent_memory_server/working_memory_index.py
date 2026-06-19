"""Working memory search index for session listing.

This module provides Redis Search index creation and management for working memory
JSON documents. Using a search index instead of sorted sets ensures that when
working memory expires via TTL, the session is automatically removed from the index.
"""

import logging

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)


def _get_working_memory_index_schema() -> IndexSchema:
    """
    Get the IndexSchema for the working memory search index.

    Returns:
        IndexSchema configured for working memory JSON documents
    """
    return IndexSchema.from_dict(
        {
            "index": {
                "name": settings.working_memory_index_name,
                "prefix": settings.working_memory_index_prefix,
                "storage_type": "json",
            },
            "fields": [
                {
                    "name": "session_id",
                    "type": "tag",
                    "path": "$.session_id",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "namespace",
                    "type": "tag",
                    "path": "$.namespace",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "user_id",
                    "type": "tag",
                    "path": "$.user_id",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "created_at",
                    "type": "numeric",
                    "path": "$.created_at",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "updated_at",
                    "type": "numeric",
                    "path": "$.updated_at",
                    "attrs": {"sortable": True},
                },
            ],
        }
    )


async def get_working_memory_index(redis_client: Redis) -> AsyncSearchIndex:
    """
    Get an AsyncSearchIndex instance for working memory.

    Args:
        redis_client: Redis client instance

    Returns:
        AsyncSearchIndex configured for working memory
    """
    schema = _get_working_memory_index_schema()
    return AsyncSearchIndex(schema=schema, redis_client=redis_client)


async def ensure_working_memory_index(redis_client: Redis) -> bool:
    """
    Ensure the working memory search index exists.

    Creates a Redis Search index on JSON documents with prefix 'working_memory:'
    if it doesn't already exist. The index enables efficient session listing
    with filtering by namespace and user_id.

    Args:
        redis_client: Redis client instance

    Returns:
        True if index was created, False if it already existed
    """
    index = await get_working_memory_index(redis_client)
    index_name = settings.working_memory_index_name
    prefix = settings.working_memory_index_prefix

    try:
        # Check if index already exists
        if await index.exists():
            logger.info(f"Working memory index '{index_name}' already exists")
            return False

        # Create the index
        await index.create(overwrite=False)
        logger.info(
            f"Created working memory index '{index_name}' with prefix '{prefix}'"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to create working memory index: {e}")
        raise


async def drop_working_memory_index(redis_client: Redis) -> bool:
    """
    Drop the working memory search index.

    Args:
        redis_client: Redis client instance

    Returns:
        True if index was dropped, False if it didn't exist
    """
    index = await get_working_memory_index(redis_client)
    index_name = settings.working_memory_index_name

    try:
        if not await index.exists():
            logger.info(f"Working memory index '{index_name}' does not exist")
            return False

        await index.delete(drop=False)
        logger.info(f"Dropped working memory index '{index_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to drop working memory index: {e}")
        raise


async def rebuild_working_memory_index(redis_client: Redis) -> bool:
    """
    Rebuild the working memory search index by dropping and recreating it.

    Args:
        redis_client: Redis client instance

    Returns:
        True if index was rebuilt successfully
    """
    await drop_working_memory_index(redis_client)
    return await ensure_working_memory_index(redis_client)
