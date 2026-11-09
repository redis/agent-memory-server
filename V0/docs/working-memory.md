# Working Memory

Working memory stores the **current conversation** for a session. It holds messages, tracks context, and automatically summarizes old messages when the conversation gets too long.

## What Working Memory Does

1. **Stores conversation messages** — The chat history for a session
2. **Tracks session data** — Arbitrary key-value data that lives only in this session
3. **Automatically summarizes** — When messages exceed the token limit, older messages are summarized and removed
4. **Promotes memories** — Structured memories added here get indexed in long-term storage

## Quick Reference

| Feature | Details |
|---------|---------|
| **Scope** | One session |
| **Lifespan** | Persistent (default) or TTL-based |
| **Storage** | Redis JSON |
| **Key Feature** | Automatic summarization |
| **Search** | None (use long-term memory for search) |

## Data Structure

Working memory contains:

| Field | Description |
|-------|-------------|
| `messages` | Conversation history (role/content pairs) |
| `context` | **Summary of older messages** (populated by auto-summarization) |
| `memories` | Structured memory records that get promoted to long-term storage |
| `data` | Arbitrary JSON key-value storage for the session |
| `user_id` | Owner of this session |
| `namespace` | Logical grouping |
| `ttl_seconds` | Optional expiration time |

## Automatic Summarization

When your conversation exceeds the model's context window, working memory automatically:

1. **Summarizes older messages** into a compact summary
2. **Stores the summary** in the `context` field
3. **Removes the summarized messages** to free space
4. **Keeps recent messages** intact

This happens transparently—you don't need to trigger it.

### How It Works

The server tracks token usage against your model's context window. When messages exceed a threshold (default: 70% of the context window), summarization kicks in:

```
Messages: [msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9, msg10]
                  ↓ (exceeds threshold)
                  ↓ summarize older messages
Context:  "User discussed trip planning to Paris, preferences for museums..."
Messages: [msg8, msg9, msg10]  ← recent messages preserved
```

### Finding the Summary

The summary is stored in the `context` field of working memory:

```python
# After summarization has occurred
working_memory = await get_working_memory("session_123")

print(working_memory.context)
# "User discussed trip planning to Paris, preferences for museums and food,
#  budget constraints around $3000, and interest in Impressionist art..."

print(working_memory.messages)
# [recent messages only]
```

### Monitoring Summarization

The `WorkingMemoryResponse` includes fields to track context usage:

```python
response = await get_working_memory("session_123")

# How much of the total context window is used (0-100%)
print(response.context_percentage_total_used)  # e.g., 45.2

# How close to triggering summarization (0-100%)
print(response.context_percentage_until_summarization)  # e.g., 64.5
# When this hits 100%, summarization triggers
```

### Configuring Summarization

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SUMMARIZATION_THRESHOLD` | `0.7` | Fraction of context window that triggers summarization |
| `GENERATION_MODEL` | `gpt-4o-mini` | Model used for summarization |
| `PROGRESSIVE_SUMMARIZATION_PROMPT` | (see below) | Custom prompt for summarization |

The summarization prompt can be customized. It must include `{prev_summary}` and `{messages_joined}` placeholders:

```bash
PROGRESSIVE_SUMMARIZATION_PROMPT="Your custom prompt with {prev_summary} and {messages_joined}..."
```

## Storing Messages

The primary use of working memory is storing conversation messages:

```python
from datetime import datetime, UTC
import ulid

working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[
        MemoryMessage(
            role="user",
            content="I'm planning a trip to Paris next month",
            id=ulid.ULID(),
            created_at=datetime.now(UTC)
        ),
        MemoryMessage(
            role="assistant",
            content="What type of activities interest you?",
            id=ulid.ULID(),
            created_at=datetime.now(UTC)
        ),
    ]
)
```

> **⚠️ Always provide `created_at` timestamps**
>
> This ensures correct message ordering and proper temporal context when promoting to long-term memory. Omitting `created_at` triggers a deprecation warning—it will become required in a future version.

## Session-Specific Data

Use the `data` field for temporary information that doesn't need to persist across conversations:

```python
working_memory = WorkingMemory(
    session_id="chat_123",
    data={
        "current_topic": "trip_planning",
        "user_timezone": "America/New_York",
    }
)
```

## Structured Memories

Use the `memories` field for facts that should persist beyond this session:

```python
working_memory = WorkingMemory(
    session_id="chat_123",
    memories=[
        MemoryRecord(
            text="User is planning a trip to Paris next month",
            id="trip_planning_paris",
            memory_type="episodic",
            topics=["travel"],
            entities=["Paris"]
        )
    ]
)
```

These are automatically promoted to long-term storage and become searchable across all sessions.

> **Key distinction:**
> - `data` → session-only, not searchable, not persisted beyond session
> - `memories` → promoted to long-term storage, searchable, persistent

## Memory Promotion to Long-Term Storage

Memories added to the `memories` field are automatically promoted to long-term storage:

1. Server identifies memories with `persisted_at=null`
2. Generates vector embeddings
3. Indexes in long-term storage
4. Updates working memory with `persisted_at` timestamps

You can also configure **background extraction** to automatically extract memories from conversation messages:

```python
working_memory = WorkingMemory(
    session_id="chat_123",
    messages=[...],
    long_term_memory_strategy=MemoryStrategyConfig(
        strategy="discrete",  # or "summary", "preferences", "custom"
        config={}
    ),
)
```

See [Memory Extraction Strategies](memory-extraction-strategies.md) for configuration options.

## API Reference

```http
# Get working memory
GET /v1/working-memory/{session_id}?namespace=demo&model_name=gpt-4o

# Set working memory (replaces existing)
PUT /v1/working-memory/{session_id}?ttl_seconds=3600

# Delete working memory
DELETE /v1/working-memory/{session_id}?namespace=demo
```

## TTL and Persistence

Working memory is **persistent by default**. Set `ttl_seconds` to auto-expire:

```python
# Persistent (default)
working_memory = WorkingMemory(session_id="chat_123", messages=[...])

# Expires after 1 hour
working_memory = WorkingMemory(session_id="chat_123", messages=[...], ttl_seconds=3600)
```

**Use TTL for:** temporary sessions, privacy requirements, resource constraints.

**Keep persistent for:** conversation history, multi-turn context, support applications.

## Reconstruction from Long-Term Memory

With `INDEX_ALL_MESSAGES_IN_LONG_TERM_MEMORY=true`, working memory can be reconstructed after TTL expiration:

1. Messages are indexed in long-term memory as they flow through
2. When working memory expires, messages remain in long-term storage
3. Requesting an expired session reconstructs it from long-term memory

This lets you use TTL to save Redis memory while maintaining conversation continuity.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARIZATION_THRESHOLD` | `0.7` | Fraction of context window that triggers summarization |
| `GENERATION_MODEL` | `gpt-4o-mini` | Model for summarization |
| `PROGRESSIVE_SUMMARIZATION_PROMPT` | (built-in) | Custom summarization prompt |
| `LONG_TERM_MEMORY` | `true` | Enable long-term memory features |
| `INDEX_ALL_MESSAGES_IN_LONG_TERM_MEMORY` | `false` | Index messages for reconstruction |

See the [Configuration Guide](configuration.md) for all options.

## Related Documentation

- [Long-term Memory](long-term-memory.md) — Persistent, cross-session storage
- [Memory Integration Patterns](memory-integration-patterns.md) — How to integrate memory
- [Memory Extraction Strategies](memory-extraction-strategies.md) — Automatic memory extraction
- [LLM Providers](llm-providers.md) — Configure OpenAI, Anthropic, Bedrock, Ollama
