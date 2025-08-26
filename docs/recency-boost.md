# Recency Boost

Recency boost is an intelligent memory ranking system that combines semantic similarity with time-based relevance to surface the most contextually appropriate memories. It ensures that recent and frequently accessed memories are weighted appropriately in search results while maintaining semantic accuracy.

## Overview

Traditional semantic search relies solely on vector similarity, which may return old or rarely-used memories that are semantically similar but not contextually relevant. Recency boost addresses this by incorporating temporal factors to provide more useful, context-aware search results.

**Key Benefits:**
- **Time-aware search**: Recent memories are weighted higher in results
- **Access pattern learning**: Frequently accessed memories get priority
- **Freshness boost**: Newly created memories are more likely to surface
- **Balanced ranking**: Combines semantic similarity with temporal relevance
- **Configurable weights**: Fine-tune the balance between similarity and recency

## How Recency Boost Works

### Scoring Algorithm

Recency boost uses a composite scoring system that combines multiple factors:

```python
final_score = (
    semantic_weight * semantic_similarity +
    recency_weight * recency_score
)

where:
recency_score = (
    freshness_weight * freshness_score +
    novelty_weight * novelty_score
)
```

### Scoring Components

1. **Semantic Similarity**: Vector cosine similarity between query and memory
2. **Freshness Score**: Based on when the memory was created (exponential decay)
3. **Novelty Score**: Based on when the memory was last accessed (exponential decay)

### Decay Functions

Both freshness and novelty use exponential decay with configurable half-lives:

```python
# Freshness: How recently was this memory created?
freshness_score = exp(-ln(2) * days_since_creation / freshness_half_life)

# Novelty: How recently was this memory accessed?
novelty_score = exp(-ln(2) * days_since_last_access / novelty_half_life)
```

## Configuration

Recency boost is controlled by several parameters that can be configured per search request or globally.

### Default Settings

```json
{
  "recency_boost": true,
  "recency_semantic_weight": 0.8,
  "recency_recency_weight": 0.2,
  "recency_freshness_weight": 0.6,
  "recency_novelty_weight": 0.4,
  "recency_half_life_last_access_days": 7.0,
  "recency_half_life_created_days": 30.0
}
```

### Parameter Descriptions

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `recency_boost` | Enable/disable recency boost | `true` | boolean |
| `recency_semantic_weight` | Weight for semantic similarity | `0.8` | 0.0-1.0 |
| `recency_recency_weight` | Weight for recency score | `0.2` | 0.0-1.0 |
| `recency_freshness_weight` | Weight for creation time within recency | `0.6` | 0.0-1.0 |
| `recency_novelty_weight` | Weight for access time within recency | `0.4` | 0.0-1.0 |
| `recency_half_life_last_access_days` | Days for novelty score to halve | `7.0` | > 0.0 |
| `recency_half_life_created_days` | Days for freshness score to halve | `30.0` | > 0.0 |

**Note**: `semantic_weight + recency_weight = 1.0` and `freshness_weight + novelty_weight = 1.0`

## Usage Examples

### REST API

Control recency boost parameters in search requests:

```bash
# Search with default recency boost
curl -X POST "http://localhost:8000/v1/long-term-memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "user food preferences",
    "limit": 5
  }'

# Search with custom recency weights
curl -X POST "http://localhost:8000/v1/long-term-memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "user food preferences",
    "limit": 5,
    "recency_boost": true,
    "recency_semantic_weight": 0.7,
    "recency_recency_weight": 0.3,
    "recency_freshness_weight": 0.8,
    "recency_novelty_weight": 0.2
  }'

# Disable recency boost (pure semantic search)
curl -X POST "http://localhost:8000/v1/long-term-memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "user food preferences",
    "limit": 5,
    "recency_boost": false
  }'
```

### MCP Server

```python
# Search with default recency boost
await client.call_tool("search_long_term_memory", {
    "text": "user preferences",
    "limit": 10
})

# Search with custom recency parameters
await client.call_tool("search_long_term_memory", {
    "text": "user preferences",
    "limit": 10,
    "recency_boost": True,
    "recency_semantic_weight": 0.6,
    "recency_recency_weight": 0.4
})

# Pure semantic search (no recency)
await client.call_tool("search_long_term_memory", {
    "text": "user preferences",
    "limit": 10,
    "recency_boost": False
})
```

### Python Client

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Search with default recency boost
results = await client.search_long_term_memory(
    text="project deadlines",
    limit=10
)

# Search with custom recency configuration
results = await client.search_long_term_memory(
    text="project deadlines",
    limit=10,
    recency_boost=True,
    recency_semantic_weight=0.7,
    recency_recency_weight=0.3,
    recency_half_life_last_access_days=3.0,  # Shorter novelty decay
    recency_half_life_created_days=14.0      # Shorter freshness decay
)
```

## Practical Examples

### Customer Support Scenario

**Without Recency Boost:**
```
Query: "customer complaint about product quality"

Results:
1. [2023-06-15] Customer complaint about old product line (high similarity)
2. [2024-01-20] Recent complaint about current product (medium similarity)
3. [2023-11-03] Historical quality issues (high similarity)
```

**With Recency Boost:**
```
Query: "customer complaint about product quality"

Results:
1. [2024-01-20] Recent complaint about current product (boosted)
2. [2024-01-18] Latest quality feedback (boosted)
3. [2023-06-15] Customer complaint about old product line (demoted)
```

### Personal Assistant Scenario

**Memory Timeline:**
- **30 days ago**: "User prefers Italian food" (created, never accessed)
- **7 days ago**: "User likes Thai food" (created, accessed 3 days ago)
- **2 days ago**: "User wants to try Mediterranean food" (created, accessed today)

**Search: "user food preferences"**

**Ranking with default recency boost:**
1. "User wants to try Mediterranean food" (freshest + recently accessed)
2. "User likes Thai food" (moderate age + recent access)
3. "User prefers Italian food" (old + never re-accessed)

## Tuning Recency Parameters

### Semantic vs Recency Balance

**High semantic weight (0.9 semantic, 0.1 recency):**
- Prioritizes content similarity over time
- Good for: Historical research, knowledge bases
- Risk: May return outdated information

**Balanced weights (0.7 semantic, 0.3 recency):**
- Good balance of relevance and timeliness
- Good for: General purpose applications
- Most common configuration

**High recency weight (0.5 semantic, 0.5 recency):**
- Strongly favors recent and accessed memories
- Good for: Real-time applications, current events
- Risk: May miss relevant but older information

### Freshness vs Novelty Balance

**High freshness weight (0.8 freshness, 0.2 novelty):**
- Prioritizes newly created memories
- Good for: Applications with constant new information
- Use case: News, social media, live events

**Balanced (0.6 freshness, 0.4 novelty):**
- Considers both creation and access patterns
- Good for: General purpose applications
- Default configuration

**High novelty weight (0.2 freshness, 0.8 novelty):**
- Prioritizes frequently accessed memories
- Good for: Reference systems, FAQs
- Use case: Documentation, support systems

### Half-Life Configuration

**Short half-lives (fast decay):**
```json
{
  "recency_half_life_last_access_days": 3.0,
  "recency_half_life_created_days": 14.0
}
```
- Aggressively favors very recent memories
- Good for: Fast-moving environments, news apps
- Risk: Older but relevant info quickly becomes invisible

**Long half-lives (slow decay):**
```json
{
  "recency_half_life_last_access_days": 30.0,
  "recency_half_life_created_days": 90.0
}
```
- More gradual preference for recent memories
- Good for: Reference systems, knowledge bases
- Maintains relevance of older information longer

## Access Pattern Learning

Recency boost learns from access patterns to improve relevance:

### Automatic Access Updates

The system automatically updates `last_accessed` timestamps when:
- Memories appear in search results
- Memories are retrieved by ID
- Memories are edited or updated
- Memories are used in memory prompts

### Rate Limiting

To prevent excessive database updates, access time updates are rate-limited:
- Maximum one update per memory per hour
- Batch updates are processed in background
- Failed updates don't affect search functionality

### Access Frequency Effects

Memories that are accessed frequently will:
- Maintain higher novelty scores longer
- Appear higher in recency-boosted searches
- Build "momentum" through repeated access
- Reflect actual usage patterns in rankings

## Real-World Tuning Examples

### E-commerce Customer Service

**Goal**: Surface recent customer issues while maintaining access to historical context

```json
{
  "recency_semantic_weight": 0.6,
  "recency_recency_weight": 0.4,
  "recency_freshness_weight": 0.7,
  "recency_novelty_weight": 0.3,
  "recency_half_life_last_access_days": 5.0,
  "recency_half_life_created_days": 21.0
}
```

**Rationale**: Higher recency weight to surface current issues, freshness-focused to catch new problems.

### Personal Knowledge Assistant

**Goal**: Balance between recent discoveries and established knowledge

```json
{
  "recency_semantic_weight": 0.8,
  "recency_recency_weight": 0.2,
  "recency_freshness_weight": 0.5,
  "recency_novelty_weight": 0.5,
  "recency_half_life_last_access_days": 14.0,
  "recency_half_life_created_days": 60.0
}
```

**Rationale**: Semantic-focused with balanced recency, longer half-lives to preserve knowledge.

### News and Content Analysis

**Goal**: Heavily prioritize recent content and trending topics

```json
{
  "recency_semantic_weight": 0.4,
  "recency_recency_weight": 0.6,
  "recency_freshness_weight": 0.8,
  "recency_novelty_weight": 0.2,
  "recency_half_life_last_access_days": 2.0,
  "recency_half_life_created_days": 7.0
}
```

**Rationale**: Heavy recency bias with short half-lives, freshness over novelty for breaking news.

## Monitoring and Analytics

### Search Result Analysis

Monitor how recency boost affects your search results:

```python
# Enable detailed logging to see scoring breakdown
LOG_LEVEL=DEBUG

# Check logs for scoring details:
# "Memory ID: abc123, semantic: 0.85, freshness: 0.62, novelty: 0.34, final: 0.79"
```

### Performance Metrics

Track these metrics to evaluate recency boost effectiveness:

1. **Click-through rates**: Are users selecting boosted results?
2. **Search satisfaction**: Do users find what they're looking for faster?
3. **Result diversity**: Is there good balance between old and new content?
4. **Query patterns**: Are repeat queries finding the same or different results?

### A/B Testing

Compare recency boost configurations:

```python
# Control group: Pure semantic search
control_config = {"recency_boost": False}

# Test group: Recency boost enabled
test_config = {
    "recency_boost": True,
    "recency_semantic_weight": 0.7,
    "recency_recency_weight": 0.3
}

# Measure: result relevance, user satisfaction, task completion
```

## Best Practices

### Configuration Strategy

1. **Start with defaults**: Begin with standard configuration and measure
2. **Understand your data**: Analyze temporal patterns in your memories
3. **Match use case**: Align configuration with application requirements
4. **Iterate gradually**: Make small adjustments and measure impact

### Content Strategy

1. **Regular updates**: Keep important memories current through editing
2. **Archive old content**: Remove or update outdated information
3. **Strategic access**: Access important memories to boost their novelty scores
4. **Content freshness**: Regularly add new relevant memories

### Performance Optimization

1. **Monitor query patterns**: Understand what users are searching for
2. **Optimize half-lives**: Tune decay rates based on your content lifecycle
3. **Balance weights**: Find the right semantic/recency balance for your use case
4. **Regular evaluation**: Periodically review and adjust configuration

## Troubleshooting

### Common Issues

**Too much recency bias:**
- Reduce `recency_recency_weight`
- Increase half-life values
- Check if important older memories are being missed

**Insufficient recency boost:**
- Increase `recency_recency_weight`
- Decrease half-life values
- Verify access patterns are being recorded correctly

**Inconsistent results:**
- Check for concurrent access updates affecting scores
- Ensure timestamps are accurate and consistent
- Verify decay calculations are working correctly

### Debug Techniques

**Enable scoring details:**
```bash
LOG_LEVEL=DEBUG
# Logs show individual scoring components
```

**Test with known data:**
```python
# Create test memories with known timestamps
# Search and verify scoring behavior matches expectations
```

**Compare with/without recency:**
```python
# Run same search with recency_boost true/false
# Compare result ordering and scores
```

This recency boost system ensures that your memory search results are not only semantically relevant but also temporally appropriate, adapting to usage patterns and the natural lifecycle of information in your application.
