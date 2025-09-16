import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks

from agent_memory_server.config import settings
from agent_memory_server.logging import get_logger


logger = get_logger(__name__)


class HybridBackgroundTasks(BackgroundTasks):
    """A BackgroundTasks implementation that can use either Docket or FastAPI background tasks."""

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run tasks either directly, through Docket, or through FastAPI background tasks"""
        logger.info("Adding task to background tasks...")

        if settings.use_docket:
            logger.info("Scheduling task through Docket")
            # Schedule task directly in Docket's Redis queue
            self._schedule_docket_task(func, *args, **kwargs)
        else:
            logger.info("Using FastAPI background tasks")
            # Use FastAPI's background tasks directly
            super().add_task(func, *args, **kwargs)

    def _schedule_docket_task(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Schedule a task in Docket's Redis queue"""

        async def schedule_task():
            from docket import Docket

            async with Docket(
                name=settings.docket_name,
                url=settings.redis_url,
            ) as docket:
                # Schedule task in Docket's queue
                await docket.add(func)(*args, **kwargs)

        # Run the async scheduling operation
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create a task
            loop.create_task(schedule_task())
        except RuntimeError:
            # If no event loop is running, run it synchronously
            asyncio.run(schedule_task())


# Backwards compatibility alias
DocketBackgroundTasks = HybridBackgroundTasks


def get_background_tasks() -> HybridBackgroundTasks:
    """
    Dependency function that returns a HybridBackgroundTasks instance.

    This is used by API endpoints to inject a consistent background tasks object.
    """
    logger.info("Getting background tasks class")
    return HybridBackgroundTasks()
