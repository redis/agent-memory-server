"""
Simplest possible migrations you could have.
"""

import ulid
from redis.asyncio import Redis

from agent_memory_server.logging import get_logger
from agent_memory_server.long_term_memory import generate_memory_hash
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import get_redis_conn


logger = get_logger(__name__)


async def migrate_add_memory_hashes_1(redis: Redis | None = None) -> None:
    """
    Migration 1: Add memory_hash to all existing memories in Redis
    """
    logger.info("Starting memory hash migration")
    redis = redis or await get_redis_conn()

    # 1. Scan Redis for all memory keys
    memory_keys = []
    cursor = 0

    pattern = Keys.memory_key("*")

    while True:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
        memory_keys.extend(keys)
        if cursor == 0:
            break

    if not memory_keys:
        logger.info("No memories found to migrate")
        return

    # 2. Process memories in batches
    batch_size = 50
    migrated_count = 0

    for i in range(0, len(memory_keys), batch_size):
        batch_keys = memory_keys[i : i + batch_size]
        pipeline = redis.pipeline()

        # First get the data
        for key in batch_keys:
            pipeline.hgetall(key)

        results = await pipeline.execute()

        # Now update with hashes
        update_pipeline = redis.pipeline()
        for j, result in enumerate(results):
            if not result:
                continue

            # Convert bytes to strings
            try:
                memory = {
                    k.decode() if isinstance(k, bytes) else k: v.decode()
                    if isinstance(v, bytes)
                    else v
                    for k, v in result.items()
                    if k in (b"text", b"user_id", b"session_id", b"memory_hash")
                }
            except Exception as e:
                logger.error(f"Error decoding memory: {result}, {e}")
                continue

            if not memory or "memory_hash" in memory:
                continue

            memory_hash = generate_memory_hash(memory)

            update_pipeline.hset(batch_keys[j], "memory_hash", memory_hash)
            migrated_count += 1

        await update_pipeline.execute()
        logger.info(f"Migrated {migrated_count} memories so far")

    logger.info(f"Migration completed. Added hashes to {migrated_count} memories")


async def migrate_add_discrete_memory_extracted_2(redis: Redis | None = None) -> None:
    """
    Migration 2: Add discrete_memory_extracted to all existing memories in Redis
    """
    logger.info("Starting discrete_memory_extracted migration")
    redis = redis or await get_redis_conn()

    keys = await redis.keys(Keys.memory_key("*"))

    migrated_count = 0
    for key in keys:
        id_ = await redis.hget(name=key, key="id_")  # type: ignore
        if not id_:
            logger.info("Updating memory with no ID to set ID")
            await redis.hset(name=key, key="id_", value=str(ulid.ULID()))  # type: ignore
        # extracted: bytes | None = await redis.hget(
        #     name=key, key="discrete_memory_extracted"
        # )  # type: ignore
        # if extracted and extracted.decode() == "t":
        #     continue
        await redis.hset(name=key, key="discrete_memory_extracted", value="f")  # type: ignore
        migrated_count += 1

    logger.info(
        f"Migration completed. Added discrete_memory_extracted (f) to {migrated_count} memories"
    )


async def migrate_add_memory_type_3(redis: Redis | None = None) -> None:
    """
    Migration 3: Add memory_type to all existing memories in Redis
    """
    logger.info("Starting memory_type migration")
    redis = redis or await get_redis_conn()

    keys = await redis.keys(Keys.memory_key("*"))

    migrated_count = 0
    for key in keys:
        id_ = await redis.hget(name=key, key="id_")  # type: ignore
        if not id_:
            logger.info("Updating memory with no ID to set ID")
            await redis.hset(name=key, key="id_", value=str(ulid.ULID()))  # type: ignore
        memory_type: bytes | None = await redis.hget(name=key, key="memory_type")  # type: ignore
        if not memory_type:
            await redis.hset(name=key, key="memory_type", value="message")  # type: ignore
        migrated_count += 1

    logger.info(f"Migration completed. Added memory_type to {migrated_count} memories")


async def migrate_redis_key_naming_4(
    redis: Redis | None = None,
    batch_size: int = 50,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Migration 4: Rename Redis keys and drop old indexes to use dash convention.

    Renames:
      - memory_idx:* → memory-idx:*
      - working_memory:* → working-memory:*
      - auth_token:* → auth-token:*
      - auth_tokens:list → auth-tokens:list

    Drops old indexes (without deleting data):
      - memory_records
      - working_memory_idx

    Args:
        redis: Optional Redis client
        batch_size: Number of keys to rename per pipeline batch
        dry_run: If True, only count keys without renaming

    Returns:
        Dict with counts per category
    """
    logger.info("Starting Redis key naming migration (underscore → dash)")
    redis = redis or await get_redis_conn()

    counts: dict[str, int] = {
        "memory_idx": 0,
        "working_memory": 0,
        "auth_token": 0,
        "auth_tokens_list": 0,
        "indexes_dropped": 0,
    }

    # Migration status keys to skip (they are themselves being renamed)
    migration_status_keys = {
        b"working_memory:migration:complete",
        b"working-memory:migration:complete",
        b"working_memory:migration:remaining",
        b"working-memory:migration:remaining",
    }

    async def _scan_and_rename(
        pattern: str, old_prefix: str, new_prefix: str, category: str
    ) -> int:
        """Scan for keys matching pattern and rename old_prefix to new_prefix."""
        renamed = 0
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
            if not keys:
                if cursor == 0:
                    break
                continue

            # Filter out migration status keys for working_memory category
            if category == "working_memory":
                keys = [k for k in keys if k not in migration_status_keys]

            if not keys:
                if cursor == 0:
                    break
                continue

            if dry_run:
                renamed += len(keys)
            else:
                # Batch rename using pipeline with RENAMENX for idempotency
                for i in range(0, len(keys), batch_size):
                    batch = keys[i : i + batch_size]
                    pipe = redis.pipeline()
                    for key in batch:
                        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                        new_key = key_str.replace(old_prefix, new_prefix, 1)
                        pipe.renamenx(key_str, new_key)
                    results = await pipe.execute()
                    renamed += sum(1 for r in results if r)

            if cursor == 0:
                break

        return renamed

    # 1. Rename memory_idx:* → memory-idx:*
    counts["memory_idx"] = await _scan_and_rename(
        "memory_idx:*", "memory_idx:", "memory-idx:", "memory_idx"
    )
    logger.info(
        f"{'Would rename' if dry_run else 'Renamed'} {counts['memory_idx']} memory_idx keys"
    )

    # 2. Rename working_memory:* → working-memory:*
    counts["working_memory"] = await _scan_and_rename(
        "working_memory:*", "working_memory:", "working-memory:", "working_memory"
    )
    logger.info(
        f"{'Would rename' if dry_run else 'Renamed'} {counts['working_memory']} working_memory keys"
    )

    # 3. Rename auth_token:* → auth-token:*
    counts["auth_token"] = await _scan_and_rename(
        "auth_token:*", "auth_token:", "auth-token:", "auth_token"
    )
    logger.info(
        f"{'Would rename' if dry_run else 'Renamed'} {counts['auth_token']} auth_token keys"
    )

    # 4. Rename auth_tokens:list → auth-tokens:list (single key, idempotent)
    if not dry_run:
        exists = await redis.exists("auth_tokens:list")
        if exists:
            result = await redis.renamenx("auth_tokens:list", "auth-tokens:list")
            if result:
                counts["auth_tokens_list"] = 1
                logger.info("Renamed auth_tokens:list → auth-tokens:list")
            else:
                logger.info(
                    "auth-tokens:list already exists, skipping auth_tokens:list rename"
                )
    else:
        exists = await redis.exists("auth_tokens:list")
        if exists:
            counts["auth_tokens_list"] = 1
            logger.info("Would rename auth_tokens:list → auth-tokens:list")

    # 5. Drop old indexes (FT.DROPINDEX without DD flag preserves data)
    if not dry_run:
        for old_index_name in ("memory_records", "working_memory_idx"):
            try:
                await redis.execute_command("FT.DROPINDEX", old_index_name)
                counts["indexes_dropped"] += 1
                logger.info(f"Dropped old index '{old_index_name}'")
            except Exception as e:
                # Index may not exist, which is fine
                if "Unknown index name" in str(e) or "Unknown Index name" in str(e):
                    logger.info(
                        f"Old index '{old_index_name}' does not exist, skipping"
                    )
                else:
                    logger.warning(f"Failed to drop index '{old_index_name}': {e}")
    else:
        for old_index_name in ("memory_records", "working_memory_idx"):
            try:
                await redis.execute_command("FT.INFO", old_index_name)
                counts["indexes_dropped"] += 1
                logger.info(f"Would drop old index '{old_index_name}'")
            except Exception:
                logger.info(f"Old index '{old_index_name}' does not exist, skipping")

    total = sum(counts.values())
    logger.info(
        f"Redis key naming migration {'(dry run) ' if dry_run else ''}complete. "
        f"Total: {total} operations ({counts})"
    )

    return counts
