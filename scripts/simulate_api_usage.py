#!/usr/bin/env python
"""
Simulate an assistant using the API endpoint (like MCP tool would).
This tests the real-world scenario where create_long_term_memories is called.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ulid
from redis.asyncio import Redis

from agent_memory_server.config import settings
from agent_memory_server.filters import Namespace, UserId
from agent_memory_server.long_term_memory import (
    count_long_term_memories,
    search_long_term_memories,
)
from agent_memory_server.models import CreateMemoryRecordRequest, ExtractedMemoryRecord
from agent_memory_server.utils.redis import get_redis_conn


NAMESPACE = f"api-test-{ulid.ULID()}"
USER_ID = f"user-{ulid.ULID()}"

COFFEE_MEMORIES = [
    "User likes coffee, flat white usually",
    "They are a coffee enthusiast, favorite coffee is flatwhite", 
    "User loves coffee, especially flat white",
]


async def create_memory_via_api_path(memory_text: str, namespace: str, user_id: str):
    """Simulate what happens when MCP tool or API creates a memory."""
    from agent_memory_server import long_term_memory
    
    memory = ExtractedMemoryRecord(
        id=str(ulid.ULID()),
        text=memory_text,
        namespace=namespace,
        user_id=user_id,
        memory_type="semantic",
    )
    
    # This is what CreateMemoryRecordRequest does - deduplicate=True by default
    payload = CreateMemoryRecordRequest(
        memories=[memory],
        deduplicate=True,  # Default is True
    )
    
    # This is what the API endpoint does
    await long_term_memory.index_long_term_memories(
        memories=payload.memories,
        deduplicate=payload.deduplicate,
    )


async def print_state(redis: Redis, namespace: str, user_id: str, step: str):
    """Print current Redis state."""
    count = await count_long_term_memories(
        namespace=namespace, user_id=user_id, redis_client=redis
    )
    results = await search_long_term_memories(
        text="coffee flat white",
        namespace=Namespace(eq=namespace),
        user_id=UserId(eq=user_id),
        limit=100,
    )
    
    print(f"\n{'='*60}")
    print(f"STATE: {step}")
    print(f"{'='*60}")
    print(f"Memory count: {count}")
    
    for i, mem in enumerate(results.memories, 1):
        print(f"\n  [{i}] ID: {mem.id}")
        print(f"      Text: {mem.text}")
        if hasattr(mem, 'dist') and mem.dist:
            print(f"      Distance: {mem.dist:.4f}")
    
    return count


async def main():
    print("\n" + "#"*60)
    print("SIMULATION: API/MCP Tool Usage Pattern")
    print("#"*60)
    print(f"Namespace: {NAMESPACE}")
    print(f"User ID: {USER_ID}")
    print(f"Settings:")
    print(f"  - deduplication_distance_threshold: {settings.deduplication_distance_threshold}")
    
    redis = await get_redis_conn()
    counts = []
    
    for i, text in enumerate(COFFEE_MEMORIES, 1):
        print(f"\n{'#'*60}")
        print(f"STEP {i}: Assistant creates memory via API")
        print(f"{'#'*60}")
        print(f"Memory text: \"{text}\"")
        
        await create_memory_via_api_path(text, NAMESPACE, USER_ID)
        await asyncio.sleep(2)
        
        count = await print_state(redis, NAMESPACE, USER_ID, f"After memory {i}")
        counts.append(count)
    
    print("\n" + "="*60)
    print("REPORT")
    print("="*60)
    print(f"Memories submitted: {len(COFFEE_MEMORIES)}")
    print(f"Final count: {counts[-1]}")
    print(f"Progression: {counts}")
    
    if counts[-1] == 1:
        print("\n✅ DEDUPLICATION WORKING: All merged into 1 memory")
    else:
        print(f"\n❌ DUPLICATES STORED: {counts[-1]} memories exist")


if __name__ == "__main__":
    asyncio.run(main())

