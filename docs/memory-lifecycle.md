# Memory Lifecycle Management

Redis Agent Memory Server provides sophisticated memory lifecycle management to prevent unlimited growth and maintain optimal performance. This includes automatic background forgetting processes, memory compaction strategies, and server-controlled cleanup operations.

## Overview

Memory lifecycle in the system follows these stages:

1. **Creation** - Memories are created in working memory or directly as long-term memories
2. **Promotion** - Working memories are automatically promoted to long-term storage
3. **Access** - Memories are tracked for access patterns and recency
4. **Aging** - Memories accumulate age and inactivity metrics  
5. **Forgetting** - Memories are deleted by background server processes based on configuration
6. **Compaction** - Background processes optimize storage and indexes

## Key Architectural Principle

Memory forgetting operates through **server-controlled background processes**. The system automatically manages memory cleanup based on server configuration, ensuring consistent resource management and optimal performance.

## Memory Creation Patterns

The memory server is designed for **LLM-driven memory management**, where AI agents make intelligent decisions about what to remember and when. There are three primary patterns for creating long-term memories:

### 1. Automatic Background Extraction
The server continuously analyzes conversation messages using an LLM to automatically extract important facts:

```python
# Conversations are analyzed in the background
working_memory = WorkingMemory(
    session_id="user_session",
    messages=[
        {"role": "user", "content": "My name is Sarah, I'm a data scientist at Google"},
        {"role": "assistant", "content": "Nice to meet you Sarah! How long have you been at Google?"},
        {"role": "user", "content": "About 2 years now. I work primarily with machine learning models"}
    ]
)

# Server automatically extracts and creates:
# - "User's name is Sarah, works as data scientist at Google for 2 years"
# - "Sarah specializes in machine learning models"
```

**Benefits**:
- Zero extra API calls required
- No LLM token usage from your application
- Continuous learning from natural conversations
- Handles implicit information extraction

### 2. LLM-Optimized Batch Storage
Your LLM pre-identifies important information and batches it with working memory updates:

```python
# Your LLM analyzes conversation and identifies memories
working_memory = WorkingMemory(
    session_id="user_session",
    messages=conversation_messages,
    memories=[
        MemoryRecord(
            text="User Sarah prefers Python over R for data analysis",
            memory_type="semantic",
            topics=["preferences", "programming", "data_science"],
            entities=["Sarah", "Python", "R", "data analysis"]
        )
    ]
)

# Single API call stores both conversation and memories
await client.set_working_memory("user_session", working_memory)
```

**Benefits**:
- Performance optimization - no separate API calls
- LLM has full conversation context for better memory decisions
- Structured metadata (topics, entities) for better search
- Immediate availability for search

### 3. Direct Long-Term Memory API
For real-time memory creation or when working without sessions:

```python
# LLM can use create_long_term_memory tool directly
await client.create_long_term_memories([
    {
        "text": "User completed advanced Python certification course",
        "memory_type": "episodic",
        "event_date": "2024-01-15T10:00:00Z",
        "topics": ["education", "certification", "python"],
        "entities": ["Python certification"],
        "user_id": "sarah_123"
    }
])
```

**Benefits**:
- Immediate storage without working memory
- Perfect for event-driven memory creation
- Fine-grained control over memory attributes
- Cross-session memory creation

> **🎯 Recommended Pattern**: Use method #2 (LLM-optimized batch storage) for most applications as it provides the best balance of performance, control, and automatic background processing.

## Memory Forgetting

### How Forgetting Works

Memory forgetting operates as an **automatic background process** using Docket (a Redis-based task scheduler). The system periodically evaluates and deletes memories based on server configuration thresholds and policies.

### Server Configuration

Forgetting behavior is controlled entirely through server-side environment variables and configuration:

```bash
# Enable/disable automatic forgetting
FORGETTING_ENABLED=true

# How often to run the forgetting process (in minutes)  
FORGETTING_EVERY_MINUTES=60

# Maximum age before memories are eligible for deletion
FORGETTING_MAX_AGE_DAYS=90.0

# Maximum inactive period before memories are eligible for deletion
FORGETTING_MAX_INACTIVE_DAYS=30.0

# Keep only the top N most recently accessed memories
FORGETTING_BUDGET_KEEP_TOP_N=10000
```

### Forgetting Policies

The system supports these automated forgetting strategies:

#### 1. Age-Based Deletion
Memories older than `FORGETTING_MAX_AGE_DAYS` are eligible for deletion.

#### 2. Inactivity-Based Deletion  
Memories not accessed within `FORGETTING_MAX_INACTIVE_DAYS` are eligible for deletion.

#### 3. Combined Age + Inactivity
When both thresholds are set, memories must be both old AND inactive to be deleted, unless they exceed a "hard age limit" (calculated as `max_age_days * hard_age_multiplier`).

#### 4. Budget-Based Cleanup
When `FORGETTING_BUDGET_KEEP_TOP_N` is set, only the most recently accessed N memories are kept, regardless of age.

### Client Capabilities

Clients can perform direct memory management operations:

#### Delete Specific Memories
```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Delete specific long-term memories by ID
memory_ids = ["memory-id-1", "memory-id-2"] 
await client.delete_long_term_memories(memory_ids)

# Delete working memory for a session
await client.delete_working_memory("session-id")
```

#### Search and Manual Cleanup
```python
# Find memories to potentially clean up
old_memories = await client.search_long_term_memory(
    text="",
    user_id="user-123", 
    created_before=datetime.now() - timedelta(days=90),
    limit=100
)

# Manually delete specific memories if needed
memory_ids = [mem.id for mem in old_memories.memories]
await client.delete_long_term_memories(memory_ids)
```

## Background Task System

### How Background Processing Works

The server uses **Docket** (a Redis-based task queue) to manage background operations including:

- **Periodic Forgetting**: `periodic_forget_long_term_memories` task runs based on `FORGETTING_EVERY_MINUTES`
- **Memory Compaction**: Optimization and deduplication processes  
- **Index Rebuilding**: Maintaining search index performance

### Task Worker Setup

Background tasks require a separate task worker process:

```bash
# Start the background task worker
uv run agent-memory task-worker

# Or with Docker
docker-compose up  # Includes task worker
```

Without a running task worker, automatic forgetting will not occur regardless of configuration settings.

### Monitoring Background Tasks

Administrators can monitor forgetting activity through:

```python
# Example: Check memory growth over time
async def monitor_memory_usage():
    # This would typically be implemented as part of server monitoring
    # Clients cannot directly access forgetting statistics
    pass
```

**Note**: Background forgetting processes operate independently to maintain consistent server resource management.

## Client-Side Memory Management  

Clients can perform manual memory management operations alongside automatic background processes:

### Bulk Memory Deletion

#### Delete by Search Criteria
```python
async def cleanup_old_sessions(client: MemoryAPIClient, days_old: int = 30):
    """Delete all memories from old sessions"""
    
    cutoff_date = datetime.now() - timedelta(days=days_old)

    # Find old memories
    old_memories = await client.search_long_term_memory(
        text="",
        created_before=cutoff_date,
        limit=5000  # Process in batches
    )

    # Delete in batches of 100 
    memory_ids = [mem.id for mem in old_memories.memories]
    batch_size = 100

    for i in range(0, len(memory_ids), batch_size):
        batch_ids = memory_ids[i:i + batch_size]
        await client.delete_long_term_memories(batch_ids)
        print(f"Deleted batch {i//batch_size + 1}")
```

#### Selective Cleanup by Topic
```python
async def cleanup_by_topic(client: MemoryAPIClient,
                          unwanted_topics: list[str], user_id: str):
    """Remove memories containing specific topics"""

    for topic in unwanted_topics:
        # Find memories with this topic
        topic_memories = await client.search_long_term_memory(
            text="",
            topics=[topic],
            user_id=user_id,
            limit=1000
        )

        # Delete them
        memory_ids = [mem.id for mem in topic_memories.memories]
        if memory_ids:
            await client.delete_long_term_memories(memory_ids)
            print(f"Deleted {len(memory_ids)} memories with topic '{topic}'")
```

### Working Memory Cleanup

Working memory has automatic TTL (1 hour by default) but can be manually managed:

```python
# Delete specific working memory session
await client.delete_working_memory("session-123")
```

**Note**: Working memory cleanup is primarily handled by Redis TTL with configurable session timeouts.

## Memory Compaction

### Background Compaction

The system automatically runs compaction tasks as background processes. These are server-controlled and include:

- Memory deduplication and merging
- Search index optimization  
- Storage cleanup

Compaction frequency is controlled by the server configuration:

```bash
# Environment variable (minutes)
COMPACTION_EVERY_MINUTES=10  # Default: every 10 minutes
```

Compaction runs automatically through background tasks, ensuring optimal storage and search performance.

## Server Administration

### Configuration Reference

Complete server configuration for memory lifecycle management:

```bash
# Forgetting Configuration
FORGETTING_ENABLED=false                   # Disabled by default
FORGETTING_EVERY_MINUTES=60               # Check every hour
FORGETTING_MAX_AGE_DAYS=90.0              # Age threshold (days)
FORGETTING_MAX_INACTIVE_DAYS=30.0         # Inactivity threshold (days)  
FORGETTING_BUDGET_KEEP_TOP_N=10000        # Budget-based limit

# Compaction Configuration
COMPACTION_EVERY_MINUTES=10               # Compaction frequency

# Working Memory TTL (handled by Redis)
# Configured in Redis or through server settings
```

### Deployment Considerations

#### Production Setup
```bash
# Recommended production settings
FORGETTING_ENABLED=true
FORGETTING_EVERY_MINUTES=240              # Every 4 hours
FORGETTING_MAX_AGE_DAYS=365.0             # 1 year retention
FORGETTING_MAX_INACTIVE_DAYS=90.0         # 3 months inactivity
FORGETTING_BUDGET_KEEP_TOP_N=50000        # Reasonable limit

# Ensure task worker is running
docker-compose up -d task-worker
```

#### Development Setup
```bash
# Development/testing settings
FORGETTING_ENABLED=false                   # Disable for testing
COMPACTION_EVERY_MINUTES=60               # Less frequent
```

## Best Practices

### 1. Server Configuration
- **Start with forgetting disabled** in new deployments to understand memory usage patterns
- **Enable gradually** with conservative thresholds
- **Monitor memory growth** before enabling aggressive policies
- **Always run task workers** in production

### 2. Client Design
- **Don't rely on specific retention periods** - memories may be deleted by server policies
- **Use explicit deletion** for memories that must be removed immediately
- **Design for eventual consistency** - recently deleted memories might still appear briefly in searches

### 3. Operational Considerations
- **Monitor task worker health** - forgetting stops if workers are down
- **Plan for storage growth** - configure budgets based on hardware capacity
- **Consider backup strategies** - automatic forgetting is permanent
- **Test policies in staging** before production deployment

### 4. Client Application Patterns

#### Robust Memory Usage
```python
# Good: Don't assume memories persist indefinitely
async def get_user_preference(client, user_id: str, preference_key: str):
    """Get user preference with fallback to default"""
    memories = await client.search_long_term_memory(
        text=f"user preference {preference_key}",
        user_id=user_id,
        limit=1
    )
    
    if memories.memories:
        return parse_preference(memories.memories[0].text)
    else:
        return get_default_preference(preference_key)
        
# Bad: Assuming specific memories will always exist
# Hypothetical: get_memory_by_id() does not exist in the real API
```

#### Explicit Cleanup
```python
# Good: Explicit cleanup when needed
async def handle_user_data_deletion(client: MemoryAPIClient, user_id: str):
    """Handle user's right to be forgotten request"""
    
    # Find all user memories
    all_memories = await client.search_long_term_memory(
        text="",
        user_id=user_id,
        limit=10000  # Large limit to get all
    )
    
    # Delete in batches
    memory_ids = [mem.id for mem in all_memories.memories]
    batch_size = 100
    
    for i in range(0, len(memory_ids), batch_size):
        batch = memory_ids[i:i + batch_size]
        await client.delete_long_term_memories(batch)
```

## Summary

The system provides **automated memory lifecycle management** through server-controlled background processes. Clients can:

1. **Delete specific memories** by ID using `delete_long_term_memories()`
2. **Delete working memory sessions** using `delete_working_memory()`
3. **Search and identify** memories for manual cleanup

Automatic lifecycle management (forgetting, compaction, optimization) operates server-side based on configuration and background task scheduling. This design ensures consistent resource management and optimal server performance.
