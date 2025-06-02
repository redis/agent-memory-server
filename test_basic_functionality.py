#!/usr/bin/env python3
"""Test script to validate basic pluggable long-term memory functionality."""

import asyncio
import logging

from agent_memory_server.models import MemoryRecord, MemoryTypeEnum
from agent_memory_server.vectorstore_factory import (
    create_vectorstore_adapter,
    get_vectorstore_adapter,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    print("Testing basic adapter functionality...")

    # Test factory
    try:
        adapter = create_vectorstore_adapter()
        print(f"✓ Created adapter: {type(adapter).__name__}")
    except Exception as e:
        print(f"✗ Error creating adapter: {e}")
        return

    # Test global adapter
    try:
        global_adapter = await get_vectorstore_adapter()
        print(f"✓ Got global adapter: {type(global_adapter).__name__}")
    except Exception as e:
        print(f"✗ Error getting global adapter: {e}")
        return

    # Test memory creation and hashing
    try:
        memory = MemoryRecord(
            text="Test memory",
            memory_type=MemoryTypeEnum.SEMANTIC,
            user_id="test-user",
            session_id="test-session",
        )
        hash_value = adapter.generate_memory_hash(memory)
        print(f"✓ Generated memory hash: {hash_value[:16]}...")
    except Exception as e:
        print(f"✗ Error creating memory: {e}")
        return

    print("✓ Basic functionality test passed!")


async def test_basic_crud_operations():
    """Test basic CRUD operations with the vectorstore adapter."""
    print("\n=== Testing Basic CRUD Operations ===")

    # Create adapter
    adapter = create_vectorstore_adapter()

    # Get backend name safely
    if hasattr(adapter, "vectorstore"):
        backend_name = type(adapter.vectorstore).__name__
    else:
        backend_name = type(adapter).__name__

    print(f"✅ Created adapter with backend: {backend_name}")

    # Create test memories
    test_memories = [
        MemoryRecord(
            text="User prefers dark mode theme",
            session_id="test_session_1",
            user_id="test_user_1",
            namespace="preferences",
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=["ui", "preferences"],
            entities=["dark_mode", "theme"],
        ),
        MemoryRecord(
            text="User discussed vacation plans to Japan",
            session_id="test_session_1",
            user_id="test_user_1",
            namespace="conversation",
            memory_type=MemoryTypeEnum.EPISODIC,
            topics=["travel", "vacation"],
            entities=["Japan", "vacation"],
        ),
        MemoryRecord(
            text="Meeting scheduled for tomorrow at 3pm",
            session_id="test_session_2",
            user_id="test_user_1",
            namespace="calendar",
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=["meetings", "schedule"],
            entities=["meeting", "3pm"],
        ),
    ]

    print(f"📝 Creating {len(test_memories)} test memories...")

    # Test adding memories
    try:
        memory_ids = await adapter.add_memories(test_memories)
        print(f"✅ Added {len(memory_ids)} memories successfully")
        print(f"   Memory IDs: {memory_ids[:2]}...")  # Show first 2 IDs
    except Exception as e:
        print(f"❌ Error adding memories: {e}")
        return False

    # Test searching memories
    print("\n📍 Testing search functionality...")

    try:
        # Simple text search
        results = await adapter.search_memories(query="dark mode preferences", limit=5)
        print(f"✅ Text search returned {len(results.memories)} results")
        if results.memories:
            print(f"   Top result: '{results.memories[0].text[:50]}...'")
            print(f"   Score: {results.memories[0].dist}")

        # Search with filters
        from agent_memory_server.filters import SessionId, Topics

        filtered_results = await adapter.search_memories(
            query="vacation",
            session_id=SessionId(eq="test_session_1"),
            topics=Topics(any=["travel", "vacation"]),
            limit=5,
        )
        print(f"✅ Filtered search returned {len(filtered_results.memories)} results")

    except Exception as e:
        print(f"❌ Error searching memories: {e}")
        return False

    # Test counting memories
    print("\n🔢 Testing count functionality...")

    try:
        total_count = await adapter.count_memories()
        user_count = await adapter.count_memories(user_id="test_user_1")
        session_count = await adapter.count_memories(session_id="test_session_1")

        print(f"✅ Total memories: {total_count}")
        print(f"✅ User test_user_1 memories: {user_count}")
        print(f"✅ Session test_session_1 memories: {session_count}")

    except Exception as e:
        print(f"❌ Error counting memories: {e}")
        return False

    # Test deletion (optional - only if we want to clean up)
    if memory_ids:
        print(f"\n🗑️  Testing deletion of {len(memory_ids)} memories...")
        try:
            deleted_count = await adapter.delete_memories(memory_ids)
            print(f"✅ Deleted {deleted_count} memories")
        except Exception as e:
            print(f"❌ Error deleting memories: {e}")
            return False

    return True


async def test_different_backends():
    """Test multiple backends if available."""
    print("\n=== Testing Different Backends ===")

    # Test Redis (default)
    print("🔍 Testing Redis backend...")
    redis_success = await test_basic_crud_operations()

    if redis_success:
        print("✅ Redis backend test passed!")
    else:
        print("❌ Redis backend test failed!")

    return redis_success


async def main():
    """Run all tests."""
    print("🚀 Starting Pluggable Long-Term Memory Tests...")
    print("=" * 50)

    try:
        # Test basic functionality
        basic_success = await test_basic_functionality()

        # Test different backends
        backend_success = await test_different_backends()

        print("\n" + "=" * 50)
        if basic_success and backend_success:
            print(
                "🎉 All tests passed! Pluggable long-term memory is working correctly."
            )
        else:
            print("❌ Some tests failed. Please check the output above.")

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
