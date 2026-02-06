"""Working memory search index for session listing.

This module provides Redis Search index creation and management for working memory
JSON documents. Using a search index instead of sorted sets ensures that when
working memory expires via TTL, the session is automatically removed from the index.
"""

import logging

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)

# Index name constant
WORKING_MEMORY_INDEX_NAME = settings.working_memory_index_name
WORKING_MEMORY_INDEX_PREFIX = settings.working_memory_index_prefix


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
    index_name = WORKING_MEMORY_INDEX_NAME
    prefix = WORKING_MEMORY_INDEX_PREFIX

    try:
        # Check if index already exists
        await redis_client.execute_command("FT.INFO", index_name)
        logger.info(f"Working memory index '{index_name}' already exists")
        return False
    except ResponseError as e:
        error_msg = str(e).lower()
        # Handle both "unknown index name" and "no such index" error messages
        if "unknown index name" not in error_msg and "no such index" not in error_msg:
            # Some other error occurred
            raise

    # Create the index
    # Schema indexes the JSON fields we need for filtering and sorting
    try:
        await redis_client.execute_command(
            "FT.CREATE",
            index_name,
            "ON",
            "JSON",
            "PREFIX",
            "1",
            prefix,
            "SCHEMA",
            "$.session_id",
            "AS",
            "session_id",
            "TAG",
            "SORTABLE",
            "$.namespace",
            "AS",
            "namespace",
            "TAG",
            "SORTABLE",
            "$.user_id",
            "AS",
            "user_id",
            "TAG",
            "SORTABLE",
            "$.created_at",
            "AS",
            "created_at",
            "NUMERIC",
            "SORTABLE",
            "$.updated_at",
            "AS",
            "updated_at",
            "NUMERIC",
            "SORTABLE",
        )
        logger.info(
            f"Created working memory index '{index_name}' with prefix '{prefix}'"
        )
        return True
    except ResponseError as e:
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
    index_name = WORKING_MEMORY_INDEX_NAME

    try:
        await redis_client.execute_command("FT.DROPINDEX", index_name)
        logger.info(f"Dropped working memory index '{index_name}'")
        return True
    except ResponseError as e:
        error_msg = str(e).lower()
        # Handle both "unknown index name" and "no such index" error messages
        if "unknown index name" in error_msg or "no such index" in error_msg:
            logger.info(f"Working memory index '{index_name}' does not exist")
            return False
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
