# Working Memory

Working memory is **session-scoped**, **durable** storage designed for active conversation state and session data. It's the "scratch pad" where an AI agent keeps track of the current conversation context for a particular session.

## Overview

Working memory provides durable storage for a single conversation session. It's optimized for storing active conversation state, session-specific data, and structured memories that may later be promoted to long-term storage. By default, working memory persists to maintain conversation history, but you can set TTL expiration if your application doesn't need persistent conversation history.

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

### 1. Active Conversation State

```python
import ulid

# Store current conversation messages
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[
        MemoryMessage(role="user", content="What's the weather like?", id=ulid.ULID()),
        MemoryMessage(role="assistant", content="I'll check that for you...", id=ulid.ULID())
    ]
)
```

### 2. Session-Specific Structured Data

```python
# Store session-specific facts during conversation (using data field)
working_memory = WorkingMemory(
    session_id="chat_123",
    data={
        "temp_trip_info": {
            "destination": "Paris",
            "travel_month": "next month",
            "planning_stage": "initial"
        },
        "conversation_context": "travel planning"
    }
)
```

### 3. Session-Specific Settings

```python
# Store session-specific configuration
working_memory = WorkingMemory(
    session_id="chat_123",
    data={
        "user_preferences": {"temperature_unit": "celsius"},
        "conversation_mode": "casual",
        "current_task": "trip_planning"
    }
)
```

### 4. Promoting Memories to Long-Term Storage

```python
# Memories in working memory are automatically promoted to long-term storage
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
# This memory will become permanent in long-term storage
```

> **🔑 Key Distinction**:
> - Use `data` field for **session-specific** facts that stay only in the session
> - Use `memories` field for **important** facts that should be promoted to long-term storage
> - Anything in the `memories` field will automatically become persistent and searchable across all future sessions

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

## Three Ways to Create Long-Term Memories

Long-term memories are typically created by LLMs (either yours or the memory server's) based on conversations. There are three pathways:

### 1. 🤖 **Automatic Extraction from Conversations**
The server automatically extracts memories from conversation messages using an LLM in the background:

```python
# Server analyzes messages and creates memories automatically
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[
        {"role": "user", "content": "I love Italian food, especially carbonara"},
        {"role": "assistant", "content": "Great! I'll remember your preference for Italian cuisine."}
    ]
    # Server will extract: "User enjoys Italian food, particularly carbonara pasta"
)
```

### 2. ⚡ **LLM-Identified Memories via Working Memory** (Performance Optimization)
Your LLM can pre-identify memories and add them to working memory for batch storage:

```python
# LLM identifies important facts and adds to memories field
working_memory = WorkingMemory(
    session_id="chat_123",
    memories=[
        MemoryRecord(
            text="User prefers morning meetings and dislikes calls after 4 PM",
            memory_type="semantic",
            topics=["preferences", "scheduling"],
            entities=["morning meetings", "4 PM"]
        )
    ]
    # Automatically promoted to long-term storage when saving working memory
)
```

### 3. 🎯 **Direct Long-Term Memory Creation**
Create memories directly via API or LLM tool calls:

```python
# Direct API call or LLM using create_long_term_memory tool
await client.create_long_term_memories([
    {
        "text": "User works as a software engineer at TechCorp",
        "memory_type": "semantic",
        "topics": ["career", "work"],
        "entities": ["software engineer", "TechCorp"]
    }
])
```

> **💡 LLM-Driven Design**: The system is designed for LLMs to make memory decisions. Your LLM can use memory tools to search existing memories, decide what's important to remember, and choose the most efficient storage method.

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
