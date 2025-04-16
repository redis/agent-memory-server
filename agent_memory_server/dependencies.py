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

        logger.info("Adding task to background tasks...")

        if settings.use_docket:
            logger.info("Scheduling task through Docket")
            logger.info("redis_url: %s", settings.redis_url)
            logger.info("docket_name: %s", settings.docket_name)
            async with Docket(
                name=settings.docket_name,
                url=settings.redis_url,
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
