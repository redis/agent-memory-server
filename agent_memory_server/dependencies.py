import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks
from starlette.concurrency import run_in_threadpool

from agent_memory_server.config import settings
from agent_memory_server.logging import get_logger


logger = get_logger(__name__)


class HybridBackgroundTasks(BackgroundTasks):
    """A BackgroundTasks implementation that can use either Docket or asyncio tasks.

    When use_docket=True, tasks are scheduled through Docket's Redis-based queue
    for processing by a separate worker process.

    When use_docket=False, tasks are scheduled using asyncio.create_task() to run
    in the current event loop. This works in both FastAPI and MCP contexts, unlike
    the parent class's approach which relies on Starlette's response lifecycle
    (which doesn't exist in MCP's stdio/SSE modes).
    """

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Schedule a background task for execution.

        Args:
            func: The function to run (can be sync or async)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        logger.info(f"Adding background task: {func.__name__}")

        if settings.use_docket:
            logger.info("Scheduling task through Docket")

            # Import Docket here to avoid import issues in tests
            from docket import Docket

            # Schedule task directly in Docket without using FastAPI background tasks
            # This runs in a thread to avoid event loop conflicts
            def run_in_thread():
                """Run the async Docket operations in a separate thread"""

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
            logger.info("Scheduling task with asyncio.create_task")
            # Use asyncio.create_task to schedule the task in the event loop.
            # This works universally in both FastAPI and MCP contexts.
            #
            # Note: We don't use super().add_task() because Starlette's BackgroundTasks
            # relies on being attached to a response object and run after the response
            # is sent. In MCP mode (stdio/SSE), there's no Starlette response lifecycle,
            # so tasks added via super().add_task() would never execute.
            asyncio.create_task(self._run_task(func, *args, **kwargs))

    async def _run_task(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Execute a background task, handling both sync and async functions.

        Args:
            func: The function to run (can be sync or async)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        try:
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                # Run sync functions in a thread pool to avoid blocking the event loop
                await run_in_threadpool(func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Background task {func.__name__} failed: {e}", exc_info=True)


# Backwards compatibility alias
DocketBackgroundTasks = HybridBackgroundTasks


def get_background_tasks() -> HybridBackgroundTasks:
    """
    Dependency function that returns a HybridBackgroundTasks instance.

    NOTE: This function is deprecated. Use HybridBackgroundTasks directly as a type
    annotation in your endpoint instead of Depends(get_background_tasks).

    Example:
        # Old way (deprecated):
        async def endpoint(background_tasks=Depends(get_background_tasks)):
            ...

        # New way (correct):
        async def endpoint(background_tasks: HybridBackgroundTasks):
            ...

    FastAPI will automatically inject the correct instance when you use
    HybridBackgroundTasks as a type annotation.
    """
    logger.info("Getting background tasks class")
    return HybridBackgroundTasks()
