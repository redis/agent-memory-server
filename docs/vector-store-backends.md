# Vector Store Backends

The Redis Agent Memory Server uses a flexible factory system that allows you to plug in any vector store backend. Instead of maintaining database-specific code in the core system, you simply specify a Python function that creates and returns your vectorstore.

## How It Works

The server uses Redis by default, but you can override this by setting the `VECTORSTORE_FACTORY` environment variable to point to your own factory function:

```bash
# Default Redis (no configuration needed)
# VECTORSTORE_FACTORY="agent_memory_server.vectorstore_factory.create_redis_vectorstore"

# Use your own factory
VECTORSTORE_FACTORY="my_vectorstores.create_my_backend"
```

## Factory Function Requirements

Your factory function must:

1. **Accept an `embeddings` parameter**: `(embeddings: Embeddings) -> Union[VectorStore, VectorStoreAdapter]`
2. **Return either**:
   - A `VectorStore` instance (will be wrapped in `LangChainVectorStoreAdapter`)
   - A `VectorStoreAdapter` instance (used directly for full customization)

## Basic Example

Here's a simple example using Chroma:

```python
# my_vectorstores.py
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

def create_chroma_backend(embeddings: Embeddings) -> Chroma:
    """Factory function that creates a Chroma vectorstore."""
    return Chroma(
        collection_name="agent_memory",
        persist_directory="./chroma_data",
        embedding_function=embeddings
    )
```

Then configure it:
```bash
VECTORSTORE_FACTORY="my_vectorstores.create_chroma_backend"
```

## Advanced Patterns

### Environment-Based Configuration

```python
# my_vectorstores.py
import os
import json
from langchain_core.embeddings import Embeddings

def create_configured_backend(embeddings: Embeddings):
    """Factory that reads configuration from environment."""

    config = json.loads(os.getenv("VECTORSTORE_CONFIG", "{}"))
    backend_type = os.getenv("BACKEND_TYPE", "chroma")

    if backend_type == "chroma":
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=config.get("collection_name", "agent_memory"),
            persist_directory=config.get("persist_directory", "./data"),
            embedding_function=embeddings
        )
    else:
        # Add other backends as needed
        raise ValueError(f"Unsupported backend: {backend_type}")
```

### Custom Adapter

For full control over memory operations, return a custom VectorStoreAdapter:

```python
# my_adapters.py
from langchain_core.embeddings import Embeddings
from agent_memory_server.vectorstore_adapter import VectorStoreAdapter

class MyCustomAdapter(VectorStoreAdapter):
    """Custom adapter with specialized behavior."""

    def __init__(self, vectorstore, embeddings: Embeddings):
        super().__init__(vectorstore, embeddings)
        # Custom initialization

    # Override methods for custom behavior
    async def add_memories(self, memories):
        # Custom memory addition logic
        return await super().add_memories(memories)

def create_custom_adapter(embeddings: Embeddings) -> MyCustomAdapter:
    # Initialize your vectorstore however you want
    from langchain_chroma import Chroma
    vectorstore = Chroma(
        collection_name="custom_memories",
        embedding_function=embeddings
    )
    return MyCustomAdapter(vectorstore, embeddings)
```

## Error Handling

The factory system provides clear error messages:

- **Import errors**: Missing dependencies or incorrect module paths
- **Function not found**: Function doesn't exist in the specified module
- **Invalid return type**: Function must return `VectorStore` or `VectorStoreAdapter`
- **Runtime errors**: Issues during vectorstore creation

## Redis (Default Backend)

The server comes with Redis as the default backend:

```bash
# Redis configuration (optional - uses defaults)
REDIS_URL=redis://localhost:6379
REDISVL_DISTANCE_METRIC=COSINE
REDISVL_VECTOR_DIMENSIONS=1536
REDISVL_INDEX_NAME=memory
```

**Requirements:**
- Redis with RediSearch module (RedisStack recommended)
- No additional configuration needed for basic usage

## Benefits of the Factory System

✅ **Zero vendor lock-in** - plug in any vectorstore
✅ **Dynamic loading** - only install what you need
✅ **Custom adapters** - full control over memory operations
✅ **Environment-based config** - no code changes to switch backends

## How to Use Other Backends

The factory system is completely generic - any LangChain-compatible vectorstore will work. Simply:

1. Install the vectorstore library you want (`pip install langchain-chroma`, etc.)
2. Write a factory function that returns your configured vectorstore
3. Set `VECTORSTORE_FACTORY` to point to your function

The server doesn't include specific support for other backends, but the factory pattern makes it trivial to plug in whatever you need.
