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
