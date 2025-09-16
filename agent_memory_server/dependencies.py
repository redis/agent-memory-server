import concurrent.futures
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

            # Import Docket here to avoid import issues in tests
            from docket import Docket

            # Schedule task directly in Docket without using FastAPI background tasks
            # This runs in a thread to avoid event loop conflicts
            def run_in_thread():
                """Run the async Docket operations in a separate thread"""
                import asyncio

                async def schedule_task():
                    async with Docket(
                        name=settings.docket_name,
                        url=settings.redis_url,
                    ) as docket:
                        # Schedule task in Docket's queue
                        await docket.add(func)(*args, **kwargs)

                # Run in a new event loop in this thread
                asyncio.run(schedule_task())

            # Execute in a thread pool to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                future.result()  # Wait for completion

            # When using Docket, we don't add anything to FastAPI background tasks
        else:
            logger.info("Using FastAPI background tasks")
            # Use FastAPI's background tasks directly
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
