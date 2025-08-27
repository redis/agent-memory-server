# Memory Lifecycle Management

Redis Agent Memory Server provides sophisticated memory lifecycle management to prevent unlimited growth and maintain optimal performance. This includes automatic forgetting policies, manual cleanup operations, and memory compaction strategies.

## Overview

Memory lifecycle in the system follows these stages:

1. **Creation** - Memories are created in working memory or directly as long-term memories
2. **Promotion** - Working memories are automatically promoted to long-term storage
3. **Access** - Memories are tracked for access patterns and recency
4. **Aging** - Memories accumulate age and inactivity metrics
5. **Forgetting** - Memories are deleted based on configurable policies
6. **Compaction** - Background processes optimize storage and indexes

## Memory Forgetting

### Forgetting Policies

The system supports multiple forgetting strategies that can be combined:

#### 1. Age-Based Forgetting (TTL)
Removes memories older than a specified age:

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Delete memories older than 30 days
await client.forget_memories(policy={
    "max_age_days": 30.0
})
```

#### 2. Inactivity-Based Forgetting
Removes memories that haven't been accessed recently:

```python
# Delete memories not accessed in 14 days
await client.forget_memories(policy={
    "max_inactive_days": 14.0
})
```

#### 3. Combined Age + Inactivity Policy
Uses both age and inactivity with smart prioritization:

```python
# Combined policy: old AND inactive, or extremely old
await client.forget_memories(policy={
    "max_age_days": 30.0,           # Consider for deletion after 30 days
    "max_inactive_days": 7.0,       # If also inactive for 7 days
    "hard_age_multiplier": 12.0     # Force delete after 360 days (30 * 12)
})
```

**How Combined Policy Works:**
- Memories are deleted if they are both old (>30 days) AND inactive (>7 days)
- Memories are force-deleted if extremely old (>360 days) regardless of activity
- Recently accessed old memories are preserved unless extremely old

#### 4. Budget-Based Forgetting
Keep only the N most recently accessed memories:

```python
# Keep only top 1000 most recent memories
await client.forget_memories(policy={
    "budget": 1000
})
```

#### 5. Memory Type Filtering
Apply forgetting policies only to specific memory types:

```python
# Only forget episodic memories older than 7 days
await client.forget_memories(policy={
    "max_age_days": 7.0,
    "memory_type_allowlist": ["episodic"]
})
```

### Advanced Forgetting Examples

#### Tiered Forgetting Strategy
```python
class TieredMemoryManager:
    def __init__(self, client: MemoryAPIClient):
        self.client = client

    async def apply_tiered_forgetting(self, user_id: str):
        """Apply different policies for different memory types"""

        # Aggressive cleanup for episodic memories (events/conversations)
        await self.client.forget_memories(policy={
            "max_age_days": 30.0,
            "max_inactive_days": 7.0,
            "memory_type_allowlist": ["episodic"]
        }, user_id=user_id)

        # Conservative cleanup for semantic memories (facts/preferences)
        await self.client.forget_memories(policy={
            "max_age_days": 365.0,  # Keep facts for a full year
            "max_inactive_days": 90.0,
            "memory_type_allowlist": ["semantic"]
        }, user_id=user_id)

        # Budget-based cleanup to prevent unlimited growth
        await self.client.forget_memories(policy={
            "budget": 5000  # Keep top 5000 across all types
        }, user_id=user_id)
```

#### Contextual Forgetting
```python
async def forget_by_context(client: MemoryAPIClient, user_id: str):
    """Forget memories from specific contexts or sessions"""

    # Forget old conversation sessions
    old_sessions = await client.search_long_term_memory(
        text="",
        user_id=user_id,
        created_before=datetime.now() - timedelta(days=30),
        limit=1000
    )

    session_ids = {mem.session_id for mem in old_sessions.memories
                   if mem.session_id and mem.memory_type == "episodic"}

    for session_id in session_ids:
        await client.forget_memories(
            policy={"max_age_days": 1.0},  # Delete immediately
            user_id=user_id,
            session_id=session_id
        )
```

### Protecting Important Memories

#### Memory Pinning
Prevent specific memories from being deleted:

```python
# Pin important memories by ID
protected_ids = ["memory-id-1", "memory-id-2", "memory-id-3"]

await client.forget_memories(
    policy={"max_age_days": 30.0},
    pinned_ids=protected_ids  # These won't be deleted
)
```

#### Creating Protected Memory Types
```python
# Store critical user preferences with pinning
await client.create_long_term_memories([{
    "text": "User is allergic to peanuts - CRITICAL SAFETY INFORMATION",
    "memory_type": "semantic",
    "topics": ["health", "allergy", "safety"],
    "pinned": True,  # Mark as protected
    "user_id": "user-123"
}])
```

## Automatic Forgetting

### Configuration

Enable automatic periodic forgetting via environment variables:

```bash
# Enable automatic forgetting
FORGETTING_ENABLED=true

# Run forgetting every 4 hours (240 minutes)
FORGETTING_EVERY_MINUTES=240

# Automatic policy settings
FORGETTING_MAX_AGE_DAYS=90.0
FORGETTING_MAX_INACTIVE_DAYS=30.0
FORGETTING_BUDGET_KEEP_TOP_N=10000
```

### Monitoring Automatic Forgetting

```python
# Check forgetting status and history
async def monitor_forgetting(client: MemoryAPIClient):
    # Get current memory counts
    stats = await client.get_memory_statistics()
    print(f"Total memories: {stats.total_count}")
    print(f"Last compaction: {stats.last_compaction}")

    # Search for recent forgetting activity
    recent_deletions = await client.search_long_term_memory(
        text="forgetting deletion cleanup",
        created_after=datetime.now() - timedelta(hours=24),
        limit=10
    )
```

## Manual Memory Management

### Bulk Memory Operations

#### Delete by Criteria
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
        await client.delete_memories(batch_ids)
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
            await client.delete_memories(memory_ids)
            print(f"Deleted {len(memory_ids)} memories with topic '{topic}'")
```

### Working Memory Cleanup

Working memory has automatic TTL (1 hour by default) but can be manually managed:

```python
# Delete specific working memory session
await client.delete_working_memory("session-123")

# Clean up old working memory sessions (if TTL disabled)
async def cleanup_working_memory(client: MemoryAPIClient):
    # Get all active sessions
    active_sessions = await client.get_active_sessions()

    # Delete sessions older than 2 hours
    cutoff = datetime.now() - timedelta(hours=2)

    for session in active_sessions:
        if session.last_activity < cutoff:
            await client.delete_working_memory(session.session_id)
```

## Memory Compaction

### Background Compaction

The system automatically runs compaction tasks every 10 minutes to:

- Merge similar memories
- Update embeddings for improved accuracy
- Rebuild search indexes
- Clean up fragmented storage

```python
# Trigger manual compaction
await client.compact_memories(
    namespace="production",
    user_id="user-123"
)

# Schedule compaction for later
await client.schedule_compaction(
    run_at=datetime.now() + timedelta(hours=1),
    full_rebuild=False
)
```

### Compaction Strategies

#### Similarity-Based Merging
```python
# Configure automatic merging of similar memories
compaction_config = {
    "similarity_threshold": 0.95,  # Very similar memories
    "merge_strategy": "combine",   # or "keep_newest", "keep_oldest"
    "preserve_metadata": True
}

await client.compact_memories(
    user_id="user-123",
    config=compaction_config
)
```

## Performance Optimization

### Memory Usage Monitoring

```python
class MemoryMonitor:
    def __init__(self, client: MemoryAPIClient):
        self.client = client

    async def get_usage_report(self, user_id: str = None) -> dict:
        """Generate memory usage report"""

        # Get overall statistics
        stats = await self.client.get_memory_statistics(user_id=user_id)

        # Analyze by memory type
        type_breakdown = {}
        for memory_type in ["semantic", "episodic"]:
            type_memories = await self.client.search_long_term_memory(
                text="",
                memory_type=memory_type,
                user_id=user_id,
                limit=0  # Just get count
            )
            type_breakdown[memory_type] = type_memories.total_count

        # Analyze by age
        age_breakdown = {}
        for days in [1, 7, 30, 90, 365]:
            cutoff = datetime.now() - timedelta(days=days)
            recent_memories = await self.client.search_long_term_memory(
                text="",
                created_after=cutoff,
                user_id=user_id,
                limit=0
            )
            age_breakdown[f"last_{days}_days"] = recent_memories.total_count

        return {
            "total_memories": stats.total_count,
            "storage_size_mb": stats.storage_size_mb,
            "by_type": type_breakdown,
            "by_age": age_breakdown,
            "last_compaction": stats.last_compaction,
            "recommendations": self._get_recommendations(stats, type_breakdown)
        }

    def _get_recommendations(self, stats: dict, type_breakdown: dict) -> list[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if stats.total_count > 50000:
            recommendations.append("Consider enabling automatic forgetting")

        if type_breakdown.get("episodic", 0) > type_breakdown.get("semantic", 0) * 2:
            recommendations.append("High episodic memory ratio - consider shorter TTL")

        if stats.storage_size_mb > 1000:
            recommendations.append("Large storage size - run memory compaction")

        return recommendations
```

### Optimization Strategies

#### 1. Proactive Forgetting
```python
async def proactive_memory_management(client: MemoryAPIClient, user_id: str):
    """Implement proactive memory management strategy"""

    monitor = MemoryMonitor(client)
    report = await monitor.get_usage_report(user_id)

    # Apply recommendations
    if report["total_memories"] > 10000:
        # Aggressive cleanup for large memory stores
        await client.forget_memories(policy={
            "max_age_days": 60.0,
            "max_inactive_days": 14.0,
            "budget": 8000
        }, user_id=user_id)

    elif report["total_memories"] > 5000:
        # Moderate cleanup
        await client.forget_memories(policy={
            "max_age_days": 90.0,
            "max_inactive_days": 30.0
        }, user_id=user_id)

    # Run compaction if storage is large
    if report["storage_size_mb"] > 500:
        await client.compact_memories(user_id=user_id)
```

#### 2. Scheduled Maintenance
```python
import asyncio
from datetime import time

async def scheduled_maintenance(client: MemoryAPIClient):
    """Run daily maintenance at 2 AM"""

    while True:
        now = datetime.now()
        # Schedule for 2 AM next day
        tomorrow_2am = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if now.hour >= 2:
            tomorrow_2am += timedelta(days=1)

        # Wait until 2 AM
        wait_seconds = (tomorrow_2am - now).total_seconds()
        await asyncio.sleep(wait_seconds)

        # Run maintenance
        print("Starting daily memory maintenance...")

        # 1. Apply forgetting policies
        await client.forget_memories(policy={
            "max_age_days": 90.0,
            "max_inactive_days": 30.0
        })

        # 2. Compact memories
        await client.compact_memories()

        # 3. Rebuild indexes if needed
        await client.rebuild_indexes()

        print("Daily memory maintenance complete")
```

## Best Practices

### 1. Policy Design
- **Start Conservative**: Begin with longer retention periods and adjust based on usage
- **Layer Policies**: Combine multiple strategies (age + inactivity + budget)
- **Protect Critical Data**: Pin important memories or exclude them from policies
- **Monitor Impact**: Track deletion rates and user experience

### 2. Performance Considerations
- **Batch Operations**: Delete memories in batches to avoid overwhelming the system
- **Off-Peak Scheduling**: Run major cleanup during low-usage hours
- **Gradual Rollout**: Implement new policies gradually with dry-run testing
- **Index Maintenance**: Regular compaction maintains search performance

### 3. User Experience
- **Transparency**: Inform users about data retention policies
- **Control**: Allow users to protect important memories
- **Graceful Degradation**: Ensure forgetting doesn't break ongoing conversations
- **Recovery Options**: Consider soft-delete with recovery periods

### 4. Compliance and Privacy
- **Right to be Forgotten**: Implement complete user data deletion
- **Data Minimization**: Only retain necessary information
- **Audit Trails**: Log forgetting operations for compliance
- **Consent Management**: Respect user privacy preferences

## Configuration Reference

### Environment Variables

```bash
# Automatic Forgetting
FORGETTING_ENABLED=true                    # Enable automatic forgetting
FORGETTING_EVERY_MINUTES=240               # Run every 4 hours
FORGETTING_MAX_AGE_DAYS=90.0              # Delete after 90 days
FORGETTING_MAX_INACTIVE_DAYS=30.0         # Delete if inactive 30 days
FORGETTING_BUDGET_KEEP_TOP_N=10000        # Keep top 10k memories

# Working Memory TTL
WORKING_MEMORY_TTL_MINUTES=60             # Working memory expires in 1 hour

# Compaction Settings
AUTO_COMPACTION_ENABLED=true              # Enable automatic compaction
COMPACTION_SIMILARITY_THRESHOLD=0.95      # Merge very similar memories
```

### Policy Configuration Examples

```python
# Conservative policy for personal assistant
PERSONAL_ASSISTANT_POLICY = {
    "max_age_days": 365.0,        # Keep for 1 year
    "max_inactive_days": 90.0,    # Delete if unused for 3 months
    "budget": 20000,              # Maximum 20k memories
    "memory_type_allowlist": ["episodic"],  # Only clean conversations
    "hard_age_multiplier": 2.0    # Force delete after 2 years
}

# Aggressive policy for high-volume systems
HIGH_VOLUME_POLICY = {
    "max_age_days": 30.0,         # Keep for 1 month
    "max_inactive_days": 7.0,     # Delete if unused for 1 week
    "budget": 5000,               # Maximum 5k memories
    "hard_age_multiplier": 6.0    # Force delete after 6 months
}

# Selective policy for different content types
CONTENT_AWARE_POLICY = {
    "max_age_days": 60.0,
    "memory_type_allowlist": ["episodic"],
    "topic_exclusions": ["important", "pinned", "user_preference"]
}
```

Memory lifecycle management is crucial for maintaining system performance and managing storage costs while preserving valuable user context. The flexible policy system allows you to balance retention needs with resource constraints, ensuring your AI applications remain fast and relevant over time.
