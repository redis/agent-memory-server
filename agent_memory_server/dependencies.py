from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks

from agent_memory_server.config import settings
from agent_memory_server.logging import get_logger


logger = get_logger(__name__)


class DocketBackgroundTasks(BackgroundTasks):
    """A BackgroundTasks implementation that uses Docket."""

    async def add_task(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Run tasks either directly or through Docket"""
        from docket import Docket

        from agent_memory_server.utils.redis import get_redis_conn

        logger.info("Adding task to background tasks...")

        if settings.use_docket:
            logger.info("Scheduling task through Docket")
            # Get the Redis connection that's already configured (will use testcontainer in tests)
            redis_conn = await get_redis_conn()

            # Extract Redis URL from the connection pool
            connection_kwargs = redis_conn.connection_pool.connection_kwargs
            if "host" in connection_kwargs and "port" in connection_kwargs:
                redis_url = (
                    f"redis://{connection_kwargs['host']}:{connection_kwargs['port']}"
                )
                if "db" in connection_kwargs:
                    redis_url += f"/{connection_kwargs['db']}"
            else:
                # Fallback to settings if we can't extract from connection
                redis_url = settings.redis_url

            logger.info("redis_url: %s", redis_url)
            logger.info("docket_name: %s", settings.docket_name)
            async with Docket(
                name=settings.docket_name,
                url=redis_url,
            ) as docket:
                # Schedule task through Docket
                await docket.add(func)(*args, **kwargs)
        else:
            logger.info("Running task directly")
            await func(*args, **kwargs)


def get_background_tasks() -> DocketBackgroundTasks:
    """
    Dependency function that returns a DocketBackgroundTasks instance.

    This is used by API endpoints to inject a consistent background tasks object.
    """
    logger.info("Getting background tasks class")
    return DocketBackgroundTasks()
