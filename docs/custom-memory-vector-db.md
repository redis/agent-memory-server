# Custom Memory Vector Databases

The Agent Memory Server uses a flexible factory system that allows you to plug in custom memory vector database implementations. By default, the server uses Redis for vector indexing and search over long term memories, but you can override this by providing your own factory function.

## How It Works

The server uses Redis (via RedisVL) by default, but you can override this by setting the `MEMORY_VECTOR_DB_FACTORY` environment variable to point to your own factory function:

```bash
# Default Redis/RedisVL (no configuration needed)
# MEMORY_VECTOR_DB_FACTORY="agent_memory_server.memory_vector_db_factory.create_redis_memory_vector_db"

# Use your own factory
MEMORY_VECTOR_DB_FACTORY="my_backends.create_my_backend"
```

## Factory Function Requirements

Your factory function must:

1. **Accept an `embeddings` parameter**: The server passes a `LiteLLMEmbeddings` instance
2. **Return a `MemoryVectorDatabase` instance**: Your implementation must subclass the `MemoryVectorDatabase` abstract base class

```python
from agent_memory_server.memory_vector_db import MemoryVectorDatabase

def create_my_backend(embeddings) -> MemoryVectorDatabase:
    """Factory function that creates a custom memory vector database."""
    return MyCustomMemoryVectorDatabase(embeddings)
```

## The MemoryVectorDatabase Interface

All backends must implement the `MemoryVectorDatabase` abstract base class:

```python
from agent_memory_server.memory_vector_db import MemoryVectorDatabase
from agent_memory_server.models import MemoryRecord, MemoryRecordResult, MemoryRecordResults

class MyCustomBackend(MemoryVectorDatabase):
    """Custom backend implementation."""

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memories and return their IDs."""
        ...

    async def search_memories(self, query: str, limit: int = 10, **kwargs) -> MemoryRecordResults:
        """Search memories by semantic similarity."""
        ...

    async def delete_memories(self, memory_ids: list[str]) -> None:
        """Delete memories by ID."""
        ...

    async def update_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Update existing memories."""
        ...

    async def count_memories(self, **filter_kwargs) -> int:
        """Count memories matching filters."""
        ...

    async def list_memories(self, offset: int = 0, limit: int = 100, **filter_kwargs) -> MemoryRecordResults:
        """List memories with optional filters."""
        ...
```

## Basic Example

Here's a simple custom backend example:

```python
# my_backends.py
from agent_memory_server.memory_vector_db import MemoryVectorDatabase
from agent_memory_server.models import MemoryRecord, MemoryRecordResult, MemoryRecordResults

class MyCustomMemoryVectorDatabase(MemoryVectorDatabase):
    """Custom memory vector database implementation."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        # Initialize your backend connection here

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        # Your implementation
        ...

    async def search_memories(self, query: str, **kwargs) -> MemoryRecordResults:
        # Your implementation
        ...

    # ... implement remaining abstract methods

def create_my_backend(embeddings) -> MyCustomMemoryVectorDatabase:
    """Factory function that creates a custom backend."""
    return MyCustomMemoryVectorDatabase(embeddings)
```

Then configure it:
```bash
MEMORY_VECTOR_DB_FACTORY="my_backends.create_my_backend"
```

## Error Handling

The factory system provides clear error messages:

- **Import errors**: Missing dependencies or incorrect module paths
- **Function not found**: Function doesn't exist in the specified module
- **Invalid return type**: Function must return a `MemoryVectorDatabase` instance
- **Runtime errors**: Issues during backend creation

## Redis (Default Backend)

The server comes with Redis as the default backend, powered by RedisVL:

```bash
# Redis configuration (optional - uses defaults)
REDIS_URL=redis://localhost:6379
REDISVL_DISTANCE_METRIC=COSINE
REDISVL_VECTOR_DIMENSIONS=1536
REDISVL_INDEX_NAME=memory
```

**Requirements:**
- Redis 8 with vector search support
- No additional configuration needed for basic usage

## Benefits of the Factory System

- **Customizable** - implement the `MemoryVectorDatabase` interface for any backend
- **Dynamic loading** - only install what you need
- **Full control** - implement memory operations exactly how you want
- **Environment-based config** - no code changes to switch backends
