#!/usr/bin/env python
"""
Command-line interface for agent-memory-server.
"""

import datetime
import importlib
import logging
import sys

import click
import uvicorn

from agent_memory_server.config import settings
from agent_memory_server.logging import configure_logging, get_logger
from agent_memory_server.migrations import (
    migrate_add_discrete_memory_extracted_2,
    migrate_add_memory_hashes_1,
    migrate_add_memory_type_3,
)
from agent_memory_server.utils.redis import ensure_search_index_exists, get_redis_conn


configure_logging()
logger = get_logger(__name__)

VERSION = "0.2.0"


@click.group()
def cli():
    """Command-line interface for agent-memory-server."""
    pass


@cli.command()
def version():
    """Show the version of agent-memory-server."""
    click.echo(f"agent-memory-server version {VERSION}")


@cli.command()
def rebuild_index():
    """Rebuild the search index."""
    import asyncio

    async def setup_and_run():
        redis = await get_redis_conn()
        await ensure_search_index_exists(redis, overwrite=True)

    asyncio.run(setup_and_run())


@cli.command()
def migrate_memories():
    """Migrate memories from the old format to the new format."""
    import asyncio

    click.echo("Starting memory migrations...")

    async def run_migrations():
        redis = await get_redis_conn()
        migrations = [
            migrate_add_memory_hashes_1,
            migrate_add_discrete_memory_extracted_2,
            migrate_add_memory_type_3,
        ]
        for migration in migrations:
            await migration(redis=redis)

    asyncio.run(run_migrations())

    click.echo("Memory migrations completed successfully.")


@cli.command()
@click.option("--port", default=settings.port, help="Port to run the server on")
@click.option("--host", default="0.0.0.0", help="Host to run the server on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def api(port: int, host: str, reload: bool):
    """Run the REST API server."""
    from agent_memory_server.main import on_start_logger

    on_start_logger(port)
    uvicorn.run(
        "agent_memory_server.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--port", default=settings.mcp_port, help="Port to run the MCP server on")
@click.option(
    "--mode",
    default="stdio",
    help="Run the MCP server in SSE or stdio mode",
    type=click.Choice(["stdio", "sse"]),
)
def mcp(port: int, mode: str):
    """Run the MCP server."""
    import asyncio

    # Update the port in settings FIRST
    settings.mcp_port = port

    # Import mcp_app AFTER settings have been updated
    from agent_memory_server.mcp import mcp_app

    async def setup_and_run():
        # Redis setup is handled by the MCP app before it starts

        # Run the MCP server
        if mode == "sse":
            logger.info(f"Starting MCP server on port {port}\n")
            await mcp_app.run_sse_async()
        elif mode == "stdio":
            # Try to force all logging to stderr because stdio-mode MCP servers
            # use standard output for the protocol.
            logging.basicConfig(
                level=settings.log_level,
                stream=sys.stderr,
                force=True,  # remove any existing handlers
                format="%(asctime)s %(name)s %(levelname)s %(message)s",
            )
            await mcp_app.run_stdio_async()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # Update the port in settings
    settings.mcp_port = port

    asyncio.run(setup_and_run())


@cli.command()
@click.argument("task_path")
@click.option(
    "--args",
    "-a",
    multiple=True,
    help="Arguments to pass to the task in the format key=value",
)
def schedule_task(task_path: str, args: list[str]):
    """
    Schedule a background task by path.

    TASK_PATH is the import path to the task function, e.g.,
    "agent_memory_server.long_term_memory.compact_long_term_memories"
    """
    import asyncio

    from docket import Docket

    # Parse the arguments
    task_args = {}
    for arg in args:
        try:
            key, value = arg.split("=", 1)
            # Try to convert to appropriate type
            if value.lower() == "true":
                task_args[key] = True
            elif value.lower() == "false":
                task_args[key] = False
            elif value.isdigit():
                task_args[key] = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") <= 1:
                task_args[key] = float(value)
            else:
                task_args[key] = value
        except ValueError:
            click.echo(f"Invalid argument format: {arg}. Use key=value format.")
            sys.exit(1)

    async def setup_and_run_task():
        redis = await get_redis_conn()
        await ensure_search_index_exists(redis)

        # Import the task function
        module_path, function_name = task_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            task_func = getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            click.echo(f"Error importing task: {e}")
            sys.exit(1)

        # Initialize Docket client
        async with Docket(
            name=settings.docket_name,
            url=settings.redis_url,
        ) as docket:
            click.echo(f"Scheduling task {task_path} with arguments: {task_args}")
            await docket.add(task_func)(**task_args)
            click.echo("Task scheduled successfully")

    asyncio.run(setup_and_run_task())


@cli.command()
@click.option(
    "--concurrency", default=10, help="Number of tasks to process concurrently"
)
@click.option(
    "--redelivery-timeout",
    default=30,
    help="Seconds to wait before redelivering a task to another worker",
)
def task_worker(concurrency: int, redelivery_timeout: int):
    """
    Start a Docket worker using the Docket name from settings.

    This command starts a worker that processes background tasks registered
    with Docket. The worker uses the Docket name from settings.
    """
    import asyncio

    from docket import Worker

    if not settings.use_docket:
        click.echo("Docket is disabled in settings. Cannot run worker.")
        sys.exit(1)

    asyncio.run(
        Worker.run(
            docket_name=settings.docket_name,
            url=settings.redis_url,
            concurrency=concurrency,
            redelivery_timeout=datetime.timedelta(seconds=redelivery_timeout),
            tasks=["agent_memory_server.docket_tasks:task_collection"],
        )
    )


if __name__ == "__main__":
    cli()
