# Working Memory

Working memory is **session-scoped**, **durable** storage designed for active conversation state and session data. It's the "scratch pad" where an AI agent keeps track of the current conversation context for a particular session.

## Overview

Working memory provides durable storage for a single conversation session. It's optimized for storing conversation messages, session-specific data, and structured memories that may later be promoted to long-term storage. By default, working memory persists to maintain conversation history, but you can set TTL expiration if your application doesn't need persistent conversation history.

| Feature | Details |
|---------|---------|
| **Scope** | Session-scoped |
| **Lifespan** | Durable by default, optional TTL |
| **Storage** | Redis key-value with JSON |
| **Search** | Simple text matching |
| **Capacity** | Limited by window size |
| **Use Case** | Active conversation state |
| **Indexing** | None |
| **Deduplication** | None |

## Characteristics

- **Session Scoped**: Each session has its own isolated working memory
- **Durable by Default**: Persists conversation history unless TTL is explicitly set
- **Optional TTL**: Can be configured to expire if conversation history isn't needed
- **Window Management**: Automatically summarizes when message count exceeds limits
- **Mixed Content**: Stores both conversation messages and structured memory records
- **No Indexing**: Simple JSON storage in Redis
- **Promotion**: Structured memories can be promoted to long-term storage

## Data Structure

Working memory contains:

- **Messages**: Conversation history (role/content pairs)
- **Memories**: Structured memory records awaiting promotion
- **Context**: Summary of past conversation when truncated
- **Data**: Arbitrary JSON key-value storage
- **Metadata**: User ID, timestamps, TTL settings

## When to Use Working Memory

### 1. Conversation Messages

The primary use of working memory is storing conversation messages to maintain context across turns:

```python
import ulid
from datetime import datetime, UTC

# Store conversation messages for context continuity
# IMPORTANT: Provide created_at timestamps that reflect actual message creation times
# In real usage, each message would have its own timestamp from when it was created
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[
        MemoryMessage(
            role="user",
            content="I'm planning a trip to Paris next month",
            id=ulid.ULID(),
            created_at=datetime.fromisoformat("2024-01-15T10:30:00+00:00")
        ),
        MemoryMessage(
            role="assistant",
            content="That sounds exciting! What type of activities are you interested in?",
            id=ulid.ULID(),
            created_at=datetime.fromisoformat("2024-01-15T10:30:05+00:00")
        ),
        MemoryMessage(
            role="user",
            content="I love museums and good food",
            id=ulid.ULID(),
            created_at=datetime.fromisoformat("2024-01-15T10:30:30+00:00")
        )
    ]
)

# On the next turn, the assistant can access this context:
# - User is planning a Paris trip
# - Trip is next month
# - User likes museums and food
# This enables coherent, context-aware responses
```

> **⚠️ Important: Message Timestamps**
>
> Always provide `created_at` timestamps for your messages. This ensures:
> - Accurate message ordering by recency
> - Correct temporal context when promoting to long-term memory
> - Proper recency scoring in semantic search
>
> If you omit `created_at`, the server will auto-generate it at deserialization time and log a deprecation warning. In a future major version, `created_at` will become required.

### 2. Session-Specific Data

Use the `data` field for temporary session information that doesn't need to persist across conversations:

```python
# Store session-specific facts and configuration
working_memory = WorkingMemory(
    session_id="chat_123",
    data={
        "temp_trip_info": {
            "destination": "Paris",
            "travel_month": "next month",
            "planning_stage": "initial"
        },
        "user_preferences": {"temperature_unit": "celsius"},
        "conversation_mode": "casual"
    }
)
```

### 3. Structured Memories for Long-Term Storage

Use the `memories` field for important facts that should be remembered across all future conversations:

```python
# Important facts that should persist beyond this session
working_memory = WorkingMemory(
    session_id="chat_123",
    memories=[
        MemoryRecord(
            text="User is planning a trip to Paris next month",
            id="trip_planning_paris",
            memory_type="episodic",
            topics=["travel", "planning"],
            entities=["Paris"]
        )
    ]
)
# This memory will be automatically promoted to long-term storage
```

> **🔑 Key Distinction**:
> - Use `data` field for **session-specific** facts that stay only in the session
> - Use `memories` field for **important** facts that should be promoted to long-term storage
> - Anything in the `memories` field will automatically become persistent and searchable across all future sessions

## Memory Promotion to Long-Term Storage

Working memory can automatically promote important information to long-term storage using configurable extraction strategies.

**Two approaches:**

1. **Background extraction** (server-side): The memory server automatically analyzes conversation content and extracts memories. Configure this using the `long_term_memory_strategy` field on working memory.

2. **Client-side extraction** (LLM tools): Your LLM uses tools to add memories to the `memories` field of working memory. These are batched and promoted to long-term storage efficiently.

```python
# Background extraction with a custom strategy
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[...],
    long_term_memory_strategy=MemoryStrategyConfig(
        strategy="discrete",  # or "summary", "preferences", "custom"
        config={}
    ),
    user_id="alice"
)
```

For detailed guidance on when to use each approach, see [Memory Integration Patterns](memory-integration-patterns.md).

For configuration options for each extraction strategy (discrete, summary, preferences, custom), see [Memory Extraction Strategies](memory-extraction-strategies.md).

## API Endpoints

```http
# Get working memory for a session
GET /v1/working-memory/{session_id}?namespace=demo&model_name=gpt-4o

# Set working memory (replaces existing, with optional TTL)
PUT /v1/working-memory/{session_id}?ttl_seconds=3600

# Delete working memory
DELETE /v1/working-memory/{session_id}?namespace=demo
```

## Automatic Promotion

When structured memories in working memory are stored, they are automatically promoted to long-term storage in the background:

1. Memories with `persisted_at=null` are identified
2. Server assigns unique IDs and timestamps
3. Memories are indexed in long-term storage with vector embeddings
4. Working memory is updated with `persisted_at` timestamps



## Memory Lifecycle

### 1. Creation in Working Memory
```python
# Client creates structured memory
memory = MemoryRecord(
    text="User likes Italian food",
    id="client_generated_id",
    memory_type="semantic"
)

# Add to working memory
working_memory = WorkingMemory(
    session_id="current_session",
    memories=[memory]
)
```

### 2. Automatic Promotion
```python
# Server promotes to long-term storage (background)
# - Assigns persisted_at timestamp
# - Generates vector embeddings
# - Indexes for search
# - Updates working memory with timestamps
```

## Best Practices

### Working Memory Usage
- Keep conversation state and session-specific data
- Use for session-specific configuration and context
- Store structured memories that should become long-term
- Set TTL only if conversation history doesn't need to persist
- Let automatic promotion handle long-term memory persistence

### Memory Design
- Use `data` field for session-specific facts that stay only in the session
- Use `memories` field for important facts that should be promoted to long-term storage
- Design memory text for LLM consumption
- Include relevant topics and entities for better search

## TTL and Persistence

Working memory is **durable by default** to preserve conversation history. However, you can configure TTL (time-to-live) expiration if your application doesn't need persistent conversation history:

```python
# Durable working memory (default behavior)
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[...],
    # No TTL - memory persists until explicitly deleted
)

# Working memory with TTL expiration
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[...],
    ttl_seconds=3600  # Expires after 1 hour
)
```

```http
# Set working memory with TTL via REST API
PUT /v1/working-memory/chat_123?ttl_seconds=3600
```

**When to use TTL:**
- Temporary chat sessions that don't need history
- Privacy-sensitive applications requiring automatic cleanup
- Resource-constrained environments

**When to keep durable (default):**
- Applications that need conversation history
- Multi-turn conversations that reference past context
- Customer support or assistant applications

## Transparent Reconstruction from Long-Term Memory

When `index_all_messages_in_long_term_memory` is enabled, working memory can be transparently reconstructed from long-term storage. This allows you to use TTL expiration while still maintaining conversation continuity.

**How it works:**
1. Set `index_all_messages_in_long_term_memory=true` in configuration
2. Messages are automatically indexed in long-term memory as they flow through working memory
3. When working memory expires (TTL), the messages remain in long-term storage
4. If you request a session that doesn't exist in working memory, the system automatically searches long-term memory for messages from that session and reconstructs the working memory

**Example workflow:**
```python
# 1. Store working memory with TTL (expires after 1 hour)
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[
        MemoryMessage(role="user", content="Hello"),
        MemoryMessage(role="assistant", content="Hi there!"),
    ],
    ttl_seconds=3600  # 1 hour expiration
)

# 2. Messages are automatically indexed in long-term memory
# 3. After 1 hour, working memory expires and is deleted
# 4. Later, when you request the session:

# GET /v1/working-memory/chat_123
# System automatically reconstructs from long-term memory
# Returns working memory with original messages
```

This feature is perfect for applications that want to:
- Reduce Redis memory usage with TTL expiration
- Maintain conversation continuity across sessions
- Automatically handle session restoration without manual intervention

## Configuration

Working memory behavior can be configured through environment variables:

```bash
# Working memory settings
WINDOW_SIZE=50                    # Message window before summarization
LONG_TERM_MEMORY=true            # Enable long-term memory features

# Long-term memory settings
ENABLE_DISCRETE_MEMORY_EXTRACTION=true  # Extract memories from messages
GENERATION_MODEL=gpt-4o-mini     # Model for summarization/extraction
```

For complete configuration options, see the [Configuration Guide](configuration.md).

## Related Documentation

- [Long-term Memory](long-term-memory.md) - Persistent, cross-session memory storage
- [Memory Integration Patterns](memory-integration-patterns.md) - How to integrate memory with your applications
- [Memory Extraction Strategies](memory-extraction-strategies.md) - Different approaches to memory extraction and storage
- [LLM Providers](llm-providers.md) - Configure OpenAI, Anthropic, AWS Bedrock, Ollama, and more
