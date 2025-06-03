# Memory Types

The Redis Agent Memory Server provides two distinct types of memory storage, each optimized for different use cases and access patterns: **Working Memory** and **Long-Term Memory**.

## Overview

| Feature | Working Memory | Long-Term Memory |
|---------|----------------|------------------|
| **Scope** | Session-scoped | Cross-session, persistent |
| **Lifespan** | TTL-based (1 hour default) | Permanent until manually deleted |
| **Storage** | Redis key-value with JSON | Redis with vector indexing |
| **Search** | Simple text matching | Semantic vector search |
| **Capacity** | Limited by window size | Unlimited (with compaction) |
| **Use Case** | Active conversation state | Knowledge base, user preferences |
| **Indexing** | None | Vector embeddings + metadata |
| **Deduplication** | None | Hash-based and semantic |

## Working Memory

Working memory is **session-scoped**, **ephemeral** storage designed for active conversation state and temporary data. It's the "scratch pad" where an AI agent keeps track of the current conversation context.

### Characteristics

- **Session Scoped**: Each session has its own isolated working memory
- **TTL-Based**: Automatically expires (default: 1 hour)
- **Window Management**: Automatically summarizes when message count exceeds limits
- **Mixed Content**: Stores both conversation messages and structured memory records
- **No Indexing**: Simple JSON storage in Redis
- **Promotion**: Structured memories can be promoted to long-term storage

### Data Structure

Working memory contains:

- **Messages**: Conversation history (role/content pairs)
- **Memories**: Structured memory records awaiting promotion
- **Context**: Summary of past conversation when truncated
- **Data**: Arbitrary JSON key-value storage
- **Metadata**: User ID, timestamps, TTL settings

### When to Use Working Memory

1. **Active Conversation State**
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

2. **Temporary Structured Data**
   ```python
   # Store temporary facts during conversation
   working_memory = WorkingMemory(
       session_id="chat_123",
       memories=[
           MemoryRecord(
               text="User is planning a trip to Paris next month",
               id="temp_trip_info",
               memory_type="episodic"
           )
       ]
   )
   ```

3. **Session-Specific Settings**
   ```python
   # Store ephemeral configuration
   working_memory = WorkingMemory(
       session_id="chat_123",
       data={
           "user_preferences": {"temperature_unit": "celsius"},
           "conversation_mode": "casual",
           "current_task": "trip_planning"
       }
   )
   ```

### API Endpoints

```http
# Get working memory for a session
GET /sessions/{session_id}/memory?namespace=demo&window_size=50

# Set working memory (replaces existing)
PUT /sessions/{session_id}/memory

# Delete working memory
DELETE /sessions/{session_id}/memory?namespace=demo
```

### Automatic Promotion

When structured memories in working memory are stored, they are automatically promoted to long-term storage in the background:

1. Memories with `persisted_at=null` are identified
2. Server assigns unique IDs and timestamps
3. Memories are indexed in long-term storage with vector embeddings
4. Working memory is updated with `persisted_at` timestamps

## Long-Term Memory

Long-term memory is **persistent**, **cross-session** storage designed for knowledge that should be retained and searchable across all interactions. It's the "knowledge base" where important facts, preferences, and experiences are stored.

### Characteristics

- **Cross-Session**: Accessible from any session
- **Persistent**: Survives server restarts and session expiration
- **Vector Indexed**: Semantic search with OpenAI embeddings
- **Deduplication**: Automatic hash-based and semantic deduplication
- **Rich Metadata**: Topics, entities, timestamps, memory types
- **Compaction**: Automatic cleanup and merging of duplicates

### Memory Types

Long-term memory supports three types of memories:

1. **Semantic**: Facts, preferences, general knowledge
   ```json
   {
     "text": "User prefers dark mode interfaces",
     "memory_type": "semantic",
     "topics": ["preferences", "ui"],
     "entities": ["dark mode"]
   }
   ```

2. **Episodic**: Events with temporal context
   ```json
   {
     "text": "User visited Paris in March 2024",
     "memory_type": "episodic",
     "event_date": "2024-03-15T10:00:00Z",
     "topics": ["travel"],
     "entities": ["Paris"]
   }
   ```

3. **Message**: Conversation records (auto-generated)
   ```json
   {
     "text": "user: What's the weather like?",
     "memory_type": "message",
     "session_id": "chat_123"
   }
   ```

### When to Use Long-Term Memory

1. **User Preferences and Profile**
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

2. **Important Facts and Knowledge**
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

3. **Cross-Session Context**
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

### API Endpoints

```http
# Create long-term memories
POST /long-term-memory

# Search long-term memories only
POST /long-term-memory/search

# Search across all memory types
POST /memory/search
```

### Search Capabilities

Long-term memory provides powerful search features:

#### Semantic Vector Search
```json
{
  "text": "python programming help",
  "limit": 10,
  "distance_threshold": 0.8
}
```

#### Advanced Filtering
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

#### Hybrid Search
```json
{
  "text": "travel plans",
  "filters": {
    "namespace": {"eq": "personal"},
    "event_date": {"gte": "2024-03-01T00:00:00Z"}
  },
  "include_working_memory": true,
  "include_long_term_memory": true
}
```

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

### 3. Deduplication and Compaction
```python
# Server automatically:
# - Identifies hash-based duplicates
# - Finds semantically similar memories
# - Merges related memories using LLM
# - Removes obsolete duplicates
```

### 4. Retrieval and Search
```python
# Client searches across all memory
results = await search_memories(
    text="food preferences",
    filters={"user_id": {"eq": "user_123"}}
)
```

## Memory Prompt Integration

The memory system integrates with AI prompts through the `/memory-prompt` endpoint:

```python
# Get memory-enriched prompt
response = await memory_prompt({
    "query": "Help me plan dinner",
    "session": {
        "session_id": "current_chat",
        "window_size": 20
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

## Best Practices

### Working Memory
- Keep conversation state and temporary data
- Use for session-specific configuration
- Store structured memories that might become long-term
- Let automatic promotion handle persistence

### Long-Term Memory
- Store user preferences and lasting facts
- Include rich metadata (topics, entities, timestamps)
- Use meaningful IDs for easier retrieval
- Leverage semantic search for discovery

### Memory Design
- Use semantic memory for timeless facts
- Use episodic memory for time-bound events
- Include relevant topics and entities for better search
- Design memory text for LLM consumption

### Search Strategy
- Start with semantic search for discovery
- Add filters for precision
- Use unified search for comprehensive results
- Consider both working and long-term contexts

## Configuration

Memory behavior can be configured through environment variables:

```bash
# Working memory settings
WINDOW_SIZE=50                    # Message window before summarization
LONG_TERM_MEMORY=true            # Enable long-term memory features

# Long-term memory settings
ENABLE_DISCRETE_MEMORY_EXTRACTION=true  # Extract memories from messages
GENERATION_MODEL=gpt-4o-mini     # Model for summarization/extraction

# Search settings
DEFAULT_MEMORY_LIMIT=1000        # Default search result limit
```

For complete configuration options, see the [Configuration Guide](configuration.md).
