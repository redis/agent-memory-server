from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks

from agent_memory_server.config import settings
from agent_memory_server.logging import get_logger


logger = get_logger(__name__)


class HybridBackgroundTasks(BackgroundTasks):
    """A BackgroundTasks implementation that can use either Docket or FastAPI background tasks."""

    async def add_task(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Run tasks either directly, through Docket, or through FastAPI background tasks"""
        logger.info("Adding task to background tasks...")

        if settings.use_docket:
            logger.info("Scheduling task through Docket")
            from docket import Docket

            async with Docket(
                name=settings.docket_name,
                url=settings.redis_url,
            ) as docket:
                # Schedule task through Docket
                await docket.add(func)(*args, **kwargs)
        else:
            logger.info("Using FastAPI background tasks")
            # Use FastAPI's background tasks
            super().add_task(func, *args, **kwargs)


# Backwards compatibility alias
DocketBackgroundTasks = HybridBackgroundTasks


def get_background_tasks() -> HybridBackgroundTasks:
    """
    Dependency function that returns a HybridBackgroundTasks instance.

    This is used by API endpoints to inject a consistent background tasks object.
    """
    logger.info("Getting background tasks class")
    return HybridBackgroundTasks()
