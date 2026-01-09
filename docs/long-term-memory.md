# Long-term Memory

Long-term memory is **persistent**, **cross-session** storage designed for knowledge that should be retained and searchable across all interactions. It's the "knowledge base" of facts, preferences, and experiences the agent learns at runtime.

## Overview

Long-term memory provides persistent storage that survives server restarts and session expiration. It's optimized for semantic search, deduplication, and rich metadata to enable intelligent retrieval of relevant information.

| Feature | Details |
|---------|---------|
| **Scope** | Cross-session, persistent |
| **Lifespan** | Permanent until manually deleted |
| **Storage** | Redis with vector indexing |
| **Search** | Semantic vector search |
| **Capacity** | Unlimited (with compaction) |
| **Use Case** | Knowledge base, user preferences |
| **Indexing** | Vector embeddings + metadata |
| **Deduplication** | Hash-based and semantic |

## Characteristics

- **Cross-Session**: Accessible from any session
- **Persistent**: Survives server restarts and session expiration
- **Vector Indexed**: Semantic search with configurable embeddings (OpenAI, Bedrock, Ollama, and more via [LiteLLM](llm-providers.md))
- **Deduplication**: Automatic hash-based and semantic deduplication
- **Rich Metadata**: Topics, entities, timestamps, memory types
- **Compaction**: Automatic cleanup and merging of duplicates

## Memory Types

Long-term memory supports three types of memories:

### 1. Semantic Memory
Facts, preferences, general knowledge

```json
{
  "text": "User prefers dark mode interfaces",
  "memory_type": "semantic",
  "topics": ["preferences", "ui"],
  "entities": ["dark mode"]
}
```

### 2. Episodic Memory
Events with temporal context

```json
{
  "text": "User visited Paris in March 2024",
  "memory_type": "episodic",
  "event_date": "2024-03-15T10:00:00Z",
  "topics": ["travel"],
  "entities": ["Paris"]
}
```

### 3. Message Memory
Conversation records (auto-generated)

```json
{
  "text": "user: What's the weather like?",
  "memory_type": "message",
  "session_id": "chat_123"
}
```

## When to Use Long-Term Memory

### 1. User Preferences and Profile

```python
# Store lasting user preferences
memories = [
    MemoryRecord(
        text="User prefers metric units for temperature",
        id="pref_metric_temp",
        memory_type="semantic",
        topics=["preferences", "units"],
        user_id="user_123"
    )
]
```

### 2. Important Facts and Knowledge

```python
# Store domain knowledge
memories = [
    MemoryRecord(
        text="Customer's subscription expires on 2024-06-15",
        id="sub_expiry_customer_456",
        memory_type="episodic",
        event_date=datetime(2024, 6, 15),
        entities=["customer_456", "subscription"],
        user_id="user_123"
    )
]
```

### 3. Cross-Session Context

```python
# Store context that spans conversations
memories = [
    MemoryRecord(
        text="User is working on a Python machine learning project",
        id="context_ml_project",
        memory_type="semantic",
        topics=["programming", "machine-learning", "python"],
        namespace="work_context"
    )
]
```

## API Endpoints

```http
# Create long-term memories
POST /v1/long-term-memory/

# Search long-term memories
POST /v1/long-term-memory/search
```

## Search Capabilities

Long-term memory provides powerful search features:

### Semantic Vector Search
```json
{
  "text": "python programming help",
  "limit": 10,
  "distance_threshold": 0.8
}
```

### Advanced Filtering
```json
{
  "text": "user preferences",
  "filters": {
    "user_id": {"eq": "user_123"},
    "memory_type": {"eq": "semantic"},
    "topics": {"any": ["preferences", "settings"]},
    "created_at": {"gte": "2024-01-01T00:00:00Z"}
  }
}
```

## Deduplication and Compaction

Long-term memory automatically manages duplicates through:

### Hash-Based Deduplication
- Identical text content is automatically deduplicated
- Preserves the most recent version with complete metadata

### Semantic Deduplication
- Uses vector similarity to identify semantically similar memories
- LLM-powered merging of related memories
- Configurable similarity thresholds

### Automatic Compaction
```python
# Server automatically:
# - Identifies hash-based duplicates
# - Finds semantically similar memories
# - Merges related memories using LLM
# - Removes obsolete duplicates
```

## Memory Prompt Integration

The memory system integrates with AI prompts through the `/v1/memory/prompt` endpoint:

```python
# Get memory-enriched prompt
response = await memory_prompt({
    "query": "Help me plan dinner",
    "session": {
        "session_id": "current_chat",
        "model_name": "gpt-4o",
        "context_window_max": 4000
    },
    "long_term_search": {
        "text": "food preferences dietary restrictions",
        "filters": {"user_id": {"eq": "user_123"}},
        "limit": 5
    }
})

# Returns ready-to-use messages with:
# - Conversation context from working memory
# - Relevant memories from long-term storage
# - User's query as final message
```

## Creating Long-Term Memories

There are two main ways to create long-term memories:

### 1. Automatic Promotion from Working Memory

The most common approach is to let the system automatically promote memories from working memory to long-term storage. This handles extraction strategies, background processing, and batch optimization.

!!! info "Working Memory Integration"
    For automatic memory promotion from conversations, see the [Working Memory documentation](working-memory.md). This covers extraction strategies, background processing, and how to configure the memory server to automatically create long-term memories from conversation content.

### 2. Manual Creation via API

For immediate storage of important facts, you can create long-term memories directly using the API or LLM tools.

#### Direct API Calls

```python
# Create memories directly via Python client
await client.create_long_term_memories([
    {
        "text": "User prefers dark mode interfaces",
        "memory_type": "semantic",
        "topics": ["preferences", "ui"],
        "entities": ["dark mode"],
        "user_id": "user_123"
    },
    {
        "text": "User completed Python certification on January 15, 2024",
        "memory_type": "episodic",
        "event_date": "2024-01-15T10:00:00Z",
        "topics": ["education", "certification"],
        "entities": ["Python certification"],
        "user_id": "user_123"
    }
])
```

#### LLM Tool Usage (Eager Creation)

Your LLM can use the `create_long_term_memory` tool for immediate storage:

```python
# LLM tool call for eager memory creation
tools = [client.create_long_term_memory_tool_schema()]

# LLM can call:
# create_long_term_memory(
#     memories=[
#         {
#             "text": "User works as a software engineer at TechCorp",
#             "memory_type": "semantic",
#             "topics": ["career", "work"],
#             "entities": ["software engineer", "TechCorp"]
#         }
#     ]
# )
```

This approach is ideal when:
- You need memories to be immediately searchable
- You're processing batch data or imports
- You want to bypass working memory entirely
- You have structured data that doesn't need extraction

## Configuration

Some long-term memory behavior can be configured through environment variables:

```bash
# Long-term memory settings
LONG_TERM_MEMORY=true                         # Enable long-term memory features
ENABLE_DISCRETE_MEMORY_EXTRACTION=true        # Extract memories from messages
INDEX_ALL_MESSAGES_IN_LONG_TERM_MEMORY=false  # Index messages in long-term memory (default: false)
GENERATION_MODEL=gpt-4o-mini                  # Model for summarization/extraction

# Vector search settings
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model (see LLM Providers for options)
DISTANCE_THRESHOLD=0.8                  # Similarity threshold for search
```

For complete configuration options, see the [Configuration Guide](configuration.md).

## Related Documentation

- [Working Memory](working-memory.md) - Session-scoped memory storage for conversations
- [Memory Integration Patterns](memory-integration-patterns.md) - How to integrate memory with your applications
- [Memory Extraction Strategies](memory-extraction-strategies.md) - Different approaches to memory extraction and storage
- [LLM Providers](llm-providers.md) - Configure OpenAI, Anthropic, AWS Bedrock, Ollama, and more
- [Vector Store Backends](vector-store-backends.md) - Configuring different vector storage backends
