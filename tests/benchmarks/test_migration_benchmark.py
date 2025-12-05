"""
Benchmark tests for working memory migration from string to JSON format.

Run with:
    uv run pytest tests/benchmarks/test_migration_benchmark.py -v -s --benchmark

Use environment variables to control scale:
    BENCHMARK_KEY_COUNT=10000 uv run pytest tests/benchmarks/test_migration_benchmark.py -v -s
"""

import asyncio
import json
import os
import time

import pytest

from agent_memory_server.utils.keys import Keys
from agent_memory_server.working_memory import (
    check_and_set_migration_status,
    get_working_memory,
    reset_migration_status,
)


# Default to 1000 keys for CI, can override with env var
DEFAULT_KEY_COUNT = 1000
KEY_COUNT = int(os.environ.get("BENCHMARK_KEY_COUNT", DEFAULT_KEY_COUNT))


def create_old_format_data(session_id: str, namespace: str) -> dict:
    """Create old-format working memory data."""
    return {
        "messages": [
            {
                "id": f"msg-{session_id}",
                "role": "user",
                "content": f"Hello from session {session_id}",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ],
        "memories": [],
        "session_id": session_id,
        "namespace": namespace,
        "context": None,
        "user_id": None,
        "tokens": 10,
        "ttl_seconds": None,
        "data": {},
        "long_term_memory_strategy": {"strategy": "discrete"},
        "last_accessed": 1704067200,
        "created_at": 1704067200,
        "updated_at": 1704067200,
    }


@pytest.fixture
async def cleanup_working_memory_keys(async_redis_client):
    """Clean up all working memory keys before and after test."""
    async def cleanup():
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await async_redis_client.scan(
                cursor=cursor, match="working_memory:*", count=1000
            )
            if keys:
                await async_redis_client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        return deleted

    # Clean before
    await cleanup()
    reset_migration_status()

    yield

    # Clean after
    await cleanup()
    reset_migration_status()


@pytest.mark.benchmark
class TestMigrationBenchmark:
    """Benchmark tests for migration performance."""

    @pytest.mark.asyncio
    async def test_startup_scan_performance(
        self, async_redis_client, cleanup_working_memory_keys
    ):
        """Benchmark: How long does startup scan take with many string keys?

        With early-exit optimization, this should be very fast - it stops
        as soon as it finds the first string key.
        """
        namespace = "benchmark"

        # Create string keys in batches using pipeline
        print(f"\n📊 Creating {KEY_COUNT:,} string keys...")
        start = time.perf_counter()

        batch_size = 1000
        for batch_start in range(0, KEY_COUNT, batch_size):
            pipe = async_redis_client.pipeline()
            for i in range(batch_start, min(batch_start + batch_size, KEY_COUNT)):
                key = Keys.working_memory_key(
                    session_id=f"bench-session-{i}", namespace=namespace
                )
                data = create_old_format_data(f"bench-session-{i}", namespace)
                pipe.set(key, json.dumps(data))
            await pipe.execute()

            if (batch_start + batch_size) % 10000 == 0:
                print(f"  Created {batch_start + batch_size:,} keys...")

        creation_time = time.perf_counter() - start
        print(f"✅ Created {KEY_COUNT:,} keys in {creation_time:.2f}s")

        # Benchmark startup scan (with early exit)
        print(f"\n📊 Benchmarking startup scan (early exit on first string key)...")
        reset_migration_status()

        start = time.perf_counter()
        result = await check_and_set_migration_status(async_redis_client)
        scan_time = time.perf_counter() - start

        print(f"✅ Startup scan completed in {scan_time:.4f}s (early exit)")
        print(f"   Result: migration_complete={result}")

        assert result is False  # Should find string keys

    @pytest.mark.asyncio
    async def test_lazy_migration_performance(
        self, async_redis_client, cleanup_working_memory_keys
    ):
        """Benchmark: How long does lazy migration take per key?

        This tests the CURRENT implementation which re-scans after each migration.
        With N keys, this is O(N²) - very slow for large N.
        """
        namespace = "benchmark"
        # Use smaller count for this test since it's O(N²)
        key_count = min(KEY_COUNT, 100)

        # Create string keys
        print(f"\n📊 Creating {key_count} string keys for lazy migration test...")
        for i in range(key_count):
            key = Keys.working_memory_key(
                session_id=f"lazy-session-{i}", namespace=namespace
            )
            data = create_old_format_data(f"lazy-session-{i}", namespace)
            await async_redis_client.set(key, json.dumps(data))

        # Set migration status
        reset_migration_status()
        await check_and_set_migration_status(async_redis_client)

        # Benchmark lazy migration (read each key, triggering migration)
        print(f"\n📊 Benchmarking lazy migration of {key_count} keys...")
        start = time.perf_counter()

        for i in range(key_count):
            await get_working_memory(
                session_id=f"lazy-session-{i}",
                namespace=namespace,
                redis_client=async_redis_client,
            )

        migration_time = time.perf_counter() - start
        print(f"✅ Lazy migration completed in {migration_time:.2f}s")
        print(f"   Average per key: {migration_time / key_count * 1000:.2f}ms")
        print(f"   Keys/second: {key_count / migration_time:,.0f}")

    @pytest.mark.asyncio
    async def test_post_migration_read_performance(
        self, async_redis_client, cleanup_working_memory_keys
    ):
        """Benchmark: Read performance after migration is complete (fast path)."""
        namespace = "benchmark"
        key_count = min(KEY_COUNT, 1000)

        # Create JSON keys directly (simulating post-migration state)
        print(f"\n📊 Creating {key_count} JSON keys...")
        batch_size = 100
        for batch_start in range(0, key_count, batch_size):
            pipe = async_redis_client.pipeline()
            for i in range(batch_start, min(batch_start + batch_size, key_count)):
                key = Keys.working_memory_key(
                    session_id=f"json-session-{i}", namespace=namespace
                )
                data = create_old_format_data(f"json-session-{i}", namespace)
                pipe.json().set(key, "$", data)
            await pipe.execute()

        # Set migration as complete
        reset_migration_status()
        await check_and_set_migration_status(async_redis_client)

        # Benchmark reads (should use fast path)
        print(f"\n📊 Benchmarking fast-path reads of {key_count} keys...")
        start = time.perf_counter()

        for i in range(key_count):
            await get_working_memory(
                session_id=f"json-session-{i}",
                namespace=namespace,
                redis_client=async_redis_client,
            )

        read_time = time.perf_counter() - start
        print(f"✅ Fast-path reads completed in {read_time:.2f}s")
        print(f"   Average per key: {read_time / key_count * 1000:.2f}ms")
        print(f"   Keys/second: {key_count / read_time:,.0f}")

    @pytest.mark.asyncio
    async def test_worst_case_single_string_key_at_end(
        self, async_redis_client, cleanup_working_memory_keys
    ):
        """Benchmark: Worst case - 1M JSON keys with 1 string key (scanned last).

        This tests the scenario where early-exit doesn't help because
        the string key is found at the very end of the scan.
        """
        namespace = "benchmark"

        # Create JSON keys in batches using pipeline
        print(f"\n📊 Creating {KEY_COUNT:,} JSON keys + 1 string key...")
        start = time.perf_counter()

        batch_size = 1000
        for batch_start in range(0, KEY_COUNT, batch_size):
            pipe = async_redis_client.pipeline()
            for i in range(batch_start, min(batch_start + batch_size, KEY_COUNT)):
                key = Keys.working_memory_key(
                    session_id=f"json-session-{i}", namespace=namespace
                )
                data = create_old_format_data(f"json-session-{i}", namespace)
                pipe.json().set(key, "$", data)
            await pipe.execute()

            if (batch_start + batch_size) % 100000 == 0:
                print(f"  Created {batch_start + batch_size:,} JSON keys...")

        # Add ONE string key with a session ID that sorts last alphabetically
        # Using 'zzz' prefix to make it likely to be scanned last
        string_key = Keys.working_memory_key(
            session_id="zzz-string-key-last", namespace=namespace
        )
        string_data = create_old_format_data("zzz-string-key-last", namespace)
        await async_redis_client.set(string_key, json.dumps(string_data))

        creation_time = time.perf_counter() - start
        print(f"✅ Created {KEY_COUNT:,} JSON keys + 1 string key in {creation_time:.2f}s")

        # Benchmark startup scan - must scan all keys to find the string one
        print(f"\n📊 Benchmarking startup scan (worst case - string key at end)...")
        reset_migration_status()

        start = time.perf_counter()
        result = await check_and_set_migration_status(async_redis_client)
        scan_time = time.perf_counter() - start

        print(f"✅ Startup scan completed in {scan_time:.2f}s")
        print(f"   Result: migration_complete={result}")
        print(f"   Keys scanned per second: {KEY_COUNT / scan_time:,.0f}")

        # Should find the string key
        assert result is False

