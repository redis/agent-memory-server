"""
Command-line interface for agent-memory-server.
"""

import importlib
import json
import sys
from datetime import UTC, datetime, timedelta

import click
import uvicorn

from agent_memory_server import __version__
from agent_memory_server.config import settings
from agent_memory_server.logging import (
    configure_logging,
    configure_mcp_logging,
    get_logger,
)
from agent_memory_server.migrations import (
    migrate_add_discrete_memory_extracted_2,
    migrate_add_memory_hashes_1,
    migrate_add_memory_type_3,
)
from agent_memory_server.utils.redis import get_redis_conn


logger = get_logger(__name__)

VERSION = __version__


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

    from agent_memory_server.vectorstore_adapter import RedisVectorStoreAdapter
    from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

    configure_logging()

    async def setup_and_run():
        # Get the vectorstore adapter
        adapter = await get_vectorstore_adapter()

        # Only Redis adapter supports index rebuilding
        if isinstance(adapter, RedisVectorStoreAdapter):
            index = adapter.vectorstore.index
            logger.info(f"Dropping and recreating index '{index.name}'")
            index.create(overwrite=True)
            logger.info("Index rebuilt successfully")
        else:
            logger.error(
                "Index rebuilding is only supported for Redis vectorstore. "
                "Current vectorstore does not support this operation."
            )

    asyncio.run(setup_and_run())


@cli.command()
def migrate_memories():
    """Migrate memories from the old format to the new format."""
    import asyncio

    configure_logging()
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
@click.option(
    "--batch-size",
    default=1000,
    help="Number of keys to process in each batch",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only count keys without migrating",
)
def migrate_working_memory(batch_size: int, dry_run: bool):
    """
    Migrate working memory keys from string format to JSON format.

    This command migrates all working memory keys stored in the old string
    format (JSON serialized as a string) to the new native Redis JSON format.

    Use --dry-run to see how many keys need migration without making changes.
    """
    import asyncio
    import time

    from agent_memory_server.utils.keys import Keys
    from agent_memory_server.working_memory import (
        set_migration_complete,
    )

    configure_logging()

    async def run_migration():
        import json as json_module

        redis = await get_redis_conn()

        # Scan for string keys only using _type filter (much faster)
        string_keys = []
        cursor = 0
        pattern = Keys.working_memory_key("*")

        click.echo("Scanning for working memory keys (string type only)...")
        scan_start = time.time()

        while True:
            # Use _type="string" to only get string keys directly
            cursor, keys = await redis.scan(
                cursor, match=pattern, count=batch_size, _type="string"
            )

            if keys:
                string_keys.extend(keys)

            if cursor == 0:
                break

        scan_time = time.time() - scan_start

        click.echo(f"Scan completed in {scan_time:.2f}s")
        click.echo(f"  String format (need migration): {len(string_keys)}")

        if not string_keys:
            click.echo("\nNo keys need migration. All done!")
            # Mark migration as complete
            await set_migration_complete(redis)
            return

        if dry_run:
            click.echo("\n--dry-run specified, no changes made.")
            return

        # Migrate keys in batches using pipeline
        click.echo(f"\nMigrating {len(string_keys)} keys...")
        migrate_start = time.time()
        migrated = 0
        errors = 0

        # Process in batches
        for batch_start in range(0, len(string_keys), batch_size):
            batch_keys = string_keys[batch_start : batch_start + batch_size]

            # Read all string data and TTLs in a pipeline
            read_pipe = redis.pipeline()
            for key in batch_keys:
                read_pipe.get(key)
                read_pipe.ttl(key)
            results = await read_pipe.execute()

            # Parse results (alternating: data, ttl, data, ttl, ...)
            migrations = []  # List of (key, data, ttl) tuples
            for i, key in enumerate(batch_keys):
                string_data = results[i * 2]
                ttl = results[i * 2 + 1]

                if string_data is None:
                    continue

                try:
                    if isinstance(string_data, bytes):
                        string_data = string_data.decode("utf-8")
                    data = json_module.loads(string_data)
                    migrations.append((key, data, ttl))
                except Exception as e:
                    errors += 1
                    logger.error(f"Failed to parse key {key}: {e}")

            # Execute migrations in a pipeline (delete + json.set + expire if needed)
            if migrations:
                write_pipe = redis.pipeline()
                for key, data, ttl in migrations:
                    write_pipe.delete(key)
                    write_pipe.json().set(key, "$", data)
                    if ttl > 0:
                        write_pipe.expire(key, ttl)

                try:
                    await write_pipe.execute()
                    migrated += len(migrations)
                except Exception as e:
                    # If batch fails, try one by one
                    logger.warning(
                        f"Batch migration failed, retrying individually: {e}"
                    )
                    for key, data, ttl in migrations:
                        try:
                            await redis.delete(key)
                            await redis.json().set(key, "$", data)
                            if ttl > 0:
                                await redis.expire(key, ttl)
                            migrated += 1
                        except Exception as e2:
                            errors += 1
                            logger.error(f"Failed to migrate key {key}: {e2}")

            # Progress update
            total_processed = batch_start + len(batch_keys)
            if total_processed % 10000 == 0 or total_processed == len(string_keys):
                elapsed = time.time() - migrate_start
                rate = migrated / elapsed if elapsed > 0 else 0
                remaining = len(string_keys) - total_processed
                eta = remaining / rate if rate > 0 else 0
                click.echo(
                    f"  Migrated {migrated}/{len(string_keys)} "
                    f"({rate:.0f} keys/sec, ETA: {eta:.0f}s)"
                )

        migrate_time = time.time() - migrate_start
        rate = migrated / migrate_time if migrate_time > 0 else 0

        click.echo(f"\nMigration completed in {migrate_time:.2f}s")
        click.echo(f"  Migrated: {migrated}")
        click.echo(f"  Errors: {errors}")
        click.echo(f"  Rate: {rate:.0f} keys/sec")

        if errors == 0:
            # Mark migration as complete
            await set_migration_complete(redis)
            click.echo("\nMigration status set to complete.")
            click.echo(
                "\n💡 Tip: Set WORKING_MEMORY_MIGRATION_COMPLETE=true to skip "
                "startup checks permanently."
            )
        else:
            click.echo(
                "\nMigration completed with errors. Run again to retry failed keys."
            )

    asyncio.run(run_migration())


@cli.command()
@click.option("--port", default=settings.port, help="Port to run the server on")
@click.option("--host", default="0.0.0.0", help="Host to run the server on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option(
    "--no-worker",
    is_flag=True,
    help=(
        "(DEPRECATED) Use --task-backend=asyncio instead. "
        "If present, force FastAPI/asyncio background tasks instead of Docket."
    ),
    deprecated=True,
)
@click.option(
    "--task-backend",
    default="docket",
    type=click.Choice(["asyncio", "docket"]),
    help=(
        "Background task backend (asyncio, docket). "
        "Default is 'docket' to preserve existing behavior using Docket-based "
        "workers (requires a running `agent-memory task-worker` for "
        "non-blocking background tasks). Use 'asyncio' (or deprecated "
        "--no-worker) for single-process development without a worker."
    ),
)
def api(port: int, host: str, reload: bool, no_worker: bool, task_backend: str):
    """Run the REST API server."""
    from agent_memory_server.main import on_start_logger

    configure_logging()

    # Determine effective backend.
    # - Default is 'docket' to preserve prior behavior (Docket workers).
    # - --task-backend=asyncio opts into single-process asyncio background tasks.
    # - Deprecated --no-worker flag forces asyncio for backward compatibility.
    effective_backend = "asyncio" if no_worker else task_backend

    if effective_backend == "docket":
        settings.use_docket = True
    else:  # "asyncio"
        settings.use_docket = False

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
@click.option(
    "--task-backend",
    default="asyncio",
    type=click.Choice(["asyncio", "docket"]),
    help=(
        "Background task backend (asyncio, docket). "
        "Default is 'asyncio' (no separate worker needed). "
        "Use 'docket' for production setups with a running task worker "
        "(see `agent-memory task-worker`)."
    ),
)
def mcp(port: int, mode: str, task_backend: str):
    """Run the MCP server."""
    import asyncio

    # Configure MCP-specific logging BEFORE any imports to avoid stdout contamination
    if mode == "stdio":
        configure_mcp_logging()
    else:
        configure_logging()

    # Update the port in settings FIRST
    settings.mcp_port = port

    # Import mcp_app AFTER settings have been updated
    from agent_memory_server.mcp import mcp_app

    async def setup_and_run():
        # Configure background task backend for MCP.
        # Default is asyncio (no separate worker required). Use 'docket' to
        # send tasks to a separate worker process.
        if task_backend == "docket":
            settings.use_docket = True
        else:  # "asyncio"
            settings.use_docket = False

        # Run the MCP server
        if mode == "sse":
            logger.info(f"Starting MCP server on port {port}\n")
            await mcp_app.run_sse_async()
        elif mode == "stdio":
            await mcp_app.run_stdio_async()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # TODO: Do we really need to update the port again?
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

    configure_logging()

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
        await get_redis_conn()

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

    configure_logging()

    if not settings.use_docket:
        click.echo("Docket is disabled in settings. Cannot run worker.")
        sys.exit(1)

    async def _ensure_stream_and_group():
        """Ensure the Docket stream and consumer group exist to avoid NOGROUP errors."""
        from redis.exceptions import ResponseError

        redis = await get_redis_conn()
        stream_key = f"{settings.docket_name}:stream"
        group_name = "docket-workers"

        try:
            # Create consumer group, auto-create stream if missing
            await redis.xgroup_create(
                name=stream_key, groupname=group_name, id="$", mkstream=True
            )
        except ResponseError as e:
            # BUSYGROUP means it already exists; safe to ignore
            if "BUSYGROUP" not in str(e).upper():
                raise

    async def _run_worker():
        await _ensure_stream_and_group()
        await get_redis_conn()
        await Worker.run(
            docket_name=settings.docket_name,
            url=settings.redis_url,
            concurrency=concurrency,
            redelivery_timeout=timedelta(seconds=redelivery_timeout),
            tasks=["agent_memory_server.docket_tasks:task_collection"],
        )

    asyncio.run(_run_worker())


@cli.group()
def token():
    """Manage authentication tokens."""
    pass


@token.command()
@click.option("--description", "-d", required=True, help="Token description")
@click.option("--expires-days", "-e", type=int, help="Token expiration in days")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--token",
    "provided_token",
    type=str,
    help="Use a pre-generated token instead of generating a new one.",
)
def add(
    description: str,
    expires_days: int | None,
    output_format: str,
    provided_token: str | None,
) -> None:
    """Add a new authentication token."""
    import asyncio

    from agent_memory_server.auth import TokenInfo, generate_token, hash_token
    from agent_memory_server.utils.keys import Keys

    async def create_token():
        redis = await get_redis_conn()

        # Determine token value
        token_value = provided_token or generate_token()
        token_hash = hash_token(token_value)

        # Calculate expiration
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=expires_days) if expires_days else None

        # Create token info
        token_info = TokenInfo(
            description=description,
            created_at=now,
            expires_at=expires_at,
            token_hash=token_hash,
        )

        # Store in Redis
        key = Keys.auth_token_key(token_hash)
        await redis.set(key, token_info.model_dump_json())

        # Set TTL if expiration is set
        if expires_at:
            ttl_seconds = int((expires_at - now).total_seconds())
            await redis.expire(key, ttl_seconds)

        # Add to tokens list (for listing purposes)
        list_key = Keys.auth_tokens_list_key()
        await redis.sadd(list_key, token_hash)

        return token_value, token_info

    token, token_info = asyncio.run(create_token())

    if output_format == "json":
        data = {
            "token": token,
            "description": token_info.description,
            "created_at": token_info.created_at.isoformat(),
            "expires_at": token_info.expires_at.isoformat()
            if token_info.expires_at
            else None,
            "hash": token_info.token_hash,
        }
        click.echo(json.dumps(data))
    else:
        expires_at = token_info.expires_at
        click.echo("Token created successfully!")
        click.echo(f"Token: {token}")
        click.echo(f"Description: {token_info.description}")
        if expires_at:
            click.echo(f"Expires: {expires_at.isoformat()}")
        else:
            click.echo("Expires: Never")
        click.echo("\nWARNING: Save this token securely. It will not be shown again.")


@token.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def list(output_format: str):
    """List all authentication tokens."""
    import asyncio

    from agent_memory_server.auth import TokenInfo
    from agent_memory_server.utils.keys import Keys

    async def list_tokens():
        redis = await get_redis_conn()

        # Get all token hashes
        list_key = Keys.auth_tokens_list_key()
        token_hashes = await redis.smembers(list_key)

        tokens_data = []

        if not token_hashes:
            if output_format == "text":
                click.echo("No tokens found.")
            return tokens_data

        if output_format == "text":
            click.echo("Authentication Tokens:")
            click.echo("=" * 50)

        for token_hash in token_hashes:
            key = Keys.auth_token_key(token_hash)
            token_data = await redis.get(key)

            if not token_data:
                # Token expired or deleted, remove from list
                await redis.srem(list_key, token_hash)
                continue

            try:
                token_info = TokenInfo.model_validate_json(token_data)

                tokens_data.append(
                    {
                        "hash": token_hash,
                        "description": token_info.description,
                        "created_at": token_info.created_at.isoformat(),
                        "expires_at": token_info.expires_at.isoformat()
                        if token_info.expires_at
                        else None,
                        "status": (
                            "Never Expires"
                            if not token_info.expires_at
                            else (
                                "EXPIRED"
                                if datetime.now(UTC) > token_info.expires_at
                                else "Active"
                            )
                        ),
                    }
                )

                if output_format == "text":
                    # Mask the token hash for display
                    masked_hash = token_hash[:8] + "..." + token_hash[-8:]

                    click.echo(f"Token: {masked_hash}")
                    click.echo(f"Description: {token_info.description}")
                    click.echo(f"Created: {token_info.created_at.isoformat()}")
                    if token_info.expires_at:
                        click.echo(f"Expires: {token_info.expires_at.isoformat()}")
                    else:
                        click.echo("Expires: Never")
                    click.echo("-" * 30)

            except Exception as e:
                if output_format == "text":
                    click.echo(f"Error processing token {token_hash}: {e}")

        return tokens_data

    tokens_data = asyncio.run(list_tokens())

    if output_format == "json":
        click.echo(json.dumps(tokens_data))


@token.command()
@click.argument("token_hash")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def show(token_hash: str, output_format: str):
    """Show details for a specific token."""
    import asyncio

    from agent_memory_server.auth import TokenInfo
    from agent_memory_server.utils.keys import Keys

    async def show_token():
        nonlocal token_hash
        redis = await get_redis_conn()

        # Try to find the token by partial hash
        if len(token_hash) < 16:
            # If partial hash provided, find the full hash
            list_key = Keys.auth_tokens_list_key()
            token_hashes = await redis.smembers(list_key)

            matching_hashes = [h for h in token_hashes if h.startswith(token_hash)]

            if not matching_hashes:
                if output_format == "json":
                    click.echo(
                        json.dumps({"error": f"No token found matching '{token_hash}'"})
                    )
                    sys.exit(1)
                else:
                    click.echo(f"No token found matching '{token_hash}'")
                    sys.exit(1)
            if len(matching_hashes) > 1:
                if output_format == "json":
                    click.echo(
                        json.dumps(
                            {
                                "error": f"Multiple tokens match '{token_hash}'",
                                "matches": [
                                    f"{h[:8]}...{h[-8:]}" for h in matching_hashes
                                ],
                            }
                        )
                    )
                    sys.exit(1)
                else:
                    click.echo(f"Multiple tokens match '{token_hash}':")
                    for h in matching_hashes:
                        click.echo(f"  {h[:8]}...{h[-8:]}")
                    sys.exit(1)
            token_hash = matching_hashes[0]

        key = Keys.auth_token_key(token_hash)
        token_data = await redis.get(key)

        if not token_data:
            if output_format == "json":
                click.echo(json.dumps({"error": f"Token not found: {token_hash}"}))
                sys.exit(1)
            else:
                click.echo(f"Token not found: {token_hash}")
                sys.exit(1)

        try:
            token_info = TokenInfo.model_validate_json(token_data)

            if token_info.expires_at and datetime.now(UTC) > token_info.expires_at:
                status = "EXPIRED"
            else:
                status = "Active"

            return token_hash, token_info, status

        except Exception as e:
            if output_format == "json":
                click.echo(
                    json.dumps({"error": f"Failed to parse token data: {str(e)}"})
                )
                sys.exit(1)
            else:
                click.echo(f"Error processing token: {e}", err=True)
                sys.exit(1)

    result = asyncio.run(show_token())
    token_hash, token_info, status = result

    if output_format == "json":
        data = {
            "hash": token_hash,
            "description": token_info.description,
            "created_at": token_info.created_at.isoformat(),
            "expires_at": token_info.expires_at.isoformat()
            if token_info.expires_at
            else None,
            "status": status,
        }
        click.echo(json.dumps(data))
    else:
        click.echo("Token Details:")
        click.echo("=" * 30)
        click.echo(f"Hash: {token_hash}")
        click.echo(f"Description: {token_info.description}")
        click.echo(f"Created: {token_info.created_at.isoformat()}")
        if token_info.expires_at:
            click.echo(f"Expires: {token_info.expires_at.isoformat()}")
        else:
            click.echo("Expires: Never")
        click.echo(f"Status: {status}")


@token.command()
@click.argument("token_hash")
@click.option("--force", "-f", is_flag=True, help="Force removal without confirmation")
def remove(token_hash: str, force: bool):
    """Remove an authentication token."""
    import asyncio

    from agent_memory_server.auth import TokenInfo
    from agent_memory_server.utils.keys import Keys

    async def remove_token():
        nonlocal token_hash
        redis = await get_redis_conn()

        # Try to find the token by partial hash
        if len(token_hash) < 16:
            # If partial hash provided, find the full hash
            list_key = Keys.auth_tokens_list_key()
            token_hashes = await redis.smembers(list_key)

            matching_hashes = [h for h in token_hashes if h.startswith(token_hash)]

            if not matching_hashes:
                click.echo(f"No token found matching '{token_hash}'")
                return
            if len(matching_hashes) > 1:
                click.echo(f"Multiple tokens match '{token_hash}':")
                for h in matching_hashes:
                    click.echo(f"  {h[:8]}...{h[-8:]}")
                return
            token_hash = matching_hashes[0]

        key = Keys.auth_token_key(token_hash)
        token_data = await redis.get(key)

        if not token_data:
            click.echo(f"Token not found: {token_hash}")
            return

        try:
            token_info = TokenInfo.model_validate_json(token_data)

            # Show token info before removal
            click.echo("Token to remove:")
            click.echo(f"  Description: {token_info.description}")
            click.echo(f"  Created: {token_info.created_at.isoformat()}")

            # Confirm removal
            if not force and not click.confirm(
                "Are you sure you want to remove this token?"
            ):
                click.echo("Token removal cancelled.")
                return

            # Remove from Redis
            await redis.delete(key)

            # Remove from tokens list
            list_key = Keys.auth_tokens_list_key()
            await redis.srem(list_key, token_hash)

            click.echo("Token removed successfully.")

        except Exception as e:
            click.echo(f"Error processing token: {e}")

    asyncio.run(remove_token())


if __name__ == "__main__":
    cli()
