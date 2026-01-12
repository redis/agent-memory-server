#!/usr/bin/env python
"""
Simulate an assistant using agent memory server to store coffee preferences.
Check Redis after each memory creation to see what's actually stored.
"""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ulid
from redis.asyncio import Redis

from agent_memory_server.config import settings
from agent_memory_server.filters import Namespace, UserId
from agent_memory_server.long_term_memory import (
    count_long_term_memories,
    index_long_term_memories,
    search_long_term_memories,
)
from agent_memory_server.models import MemoryRecord
from agent_memory_server.utils.redis import get_redis_conn


# Test configuration
NAMESPACE = f"test-simulation-{ulid.ULID()}"
USER_ID = f"user-{ulid.ULID()}"

# The coffee preference memories (paraphrased versions of the same fact)
COFFEE_MEMORIES = [
    "User likes coffee, flat white usually",
    "They are a coffee enthusiast, favorite coffee is flatwhite",
    "User loves coffee, especially flat white",
    "The user prefers flat white as their go-to coffee drink",
]


async def get_all_memories_from_redis(redis: Redis, namespace: str, user_id: str) -> list[dict]:
    """Directly query Redis to see all stored memories."""
    # Search for all memories in our namespace
    results = await search_long_term_memories(
        text="coffee",  # Search term
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        limit=100,
    )
    return results.memories if results else []


async def print_redis_state(redis: Redis, namespace: str, user_id: str, step: str):
    """Print the current state of memories in Redis."""
    memories = await get_all_memories_from_redis(redis, namespace, user_id)
    count = await count_long_term_memories(
        namespace=namespace,
        user_id=user_id,
        redis_client=redis,
    )
    
    print(f"\n{'='*60}")
    print(f"REDIS STATE AFTER: {step}")
    print(f"{'='*60}")
    print(f"Total memory count: {count}")
    print(f"Memories found in search: {len(memories)}")
    
    for i, mem in enumerate(memories, 1):
        print(f"\n  Memory {i}:")
        print(f"    ID: {mem.id}")
        print(f"    Text: {mem.text[:100]}..." if len(mem.text) > 100 else f"    Text: {mem.text}")
        print(f"    Created: {mem.created_at}")
        if hasattr(mem, 'dist') and mem.dist is not None:
            print(f"    Distance: {mem.dist:.4f}")
    
    return count, memories


async def simulate_assistant_usage():
    """Simulate an assistant storing coffee preference memories."""
    print("\n" + "="*60)
    print("SIMULATION: Assistant storing coffee preferences")
    print("="*60)
    print(f"Namespace: {NAMESPACE}")
    print(f"User ID: {USER_ID}")
    print(f"Deduplication threshold: {settings.deduplication_distance_threshold}")
    
    redis = await get_redis_conn()
    
    # Initial state
    await print_redis_state(redis, NAMESPACE, USER_ID, "Initial (empty)")
    
    memory_counts = []
    
    for i, text in enumerate(COFFEE_MEMORIES, 1):
        print(f"\n{'#'*60}")
        print(f"STEP {i}: Creating memory")
        print(f"{'#'*60}")
        print(f"Text: \"{text}\"")
        
        memory = MemoryRecord(
            id=str(ulid.ULID()),
            text=text,
            namespace=NAMESPACE,
            user_id=USER_ID,
            memory_type="semantic",
        )
        
        # First memory: no deduplication needed
        # Subsequent memories: deduplicate=True to test merging
        deduplicate = i > 1
        print(f"Deduplicate flag: {deduplicate}")
        
        try:
            await index_long_term_memories(
                [memory],
                redis_client=redis,
                deduplicate=deduplicate,
            )
            print("Memory indexed successfully")
        except Exception as e:
            print(f"ERROR indexing memory: {e}")
            continue
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Check Redis state
        count, memories = await print_redis_state(
            redis, NAMESPACE, USER_ID, f"Memory {i} created"
        )
        memory_counts.append(count)
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(f"Memories created: {len(COFFEE_MEMORIES)}")
    print(f"Final count in Redis: {memory_counts[-1] if memory_counts else 0}")
    print(f"Memory count progression: {memory_counts}")
    
    if memory_counts and memory_counts[-1] == 1:
        print("\n✅ SUCCESS: All paraphrased memories were merged into 1")
    elif memory_counts and memory_counts[-1] < len(COFFEE_MEMORIES):
        print(f"\n⚠️  PARTIAL: Some merging occurred ({memory_counts[-1]} memories remain)")
    else:
        print(f"\n❌ FAILURE: Duplicates stored! Expected 1, got {memory_counts[-1] if memory_counts else 'unknown'}")
    
    return memory_counts


if __name__ == "__main__":
    asyncio.run(simulate_assistant_usage())

