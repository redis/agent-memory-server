# Memory Deduplication Analysis

> **Note:** This document was created during the investigation of GitHub Issue #110.
> The issue has been **resolved**. See `dev_docs/issue_110_duplicate_memories_fix.md` for the complete fix documentation.
>
> **Key changes made:**
> - **Tier 1 (Store-Time):** Threshold changed from 0.12 to 0.35 (catches paraphrased duplicates)
> - **Tier 1 (Store-Time):** Hybrid search removed from deduplication path (unnecessary complexity)
> - **Tier 1 (Store-Time):** `promote_working_memory_to_long_term()` now enables deduplication
> - **Tier 2 (Load-Time):** Implemented! Search results are now deduplicated before returning to the agent
>   - Clusters similar memories by vector similarity (threshold 0.4)
>   - Uses LLM to verify which memories represent the same fact
>   - Merges duplicate clusters into single coherent memories
>   - Configurable via `enable_load_time_deduplication`, `load_time_dedup_threshold`, `load_time_dedup_use_llm`

---

## Problem Statement

From [GitHub Issue #110](https://github.com/redis/agent-memory-server/issues/110):

> "3/7 of the extracted long term memories are about my preference for Flat White... 4/7 are about the fact that I broke my pour over set... So really this memory space is crowded and should be 2 entries in total with 2 discrete facts."

**Example duplicates stored in Redis:**
1. "User likes coffee, flat white usually"
2. "They are a coffee enthusiast, favorite coffee is flatwhite"
3. "User loves coffee, especially flat white"

These represent the **same fact** but are stored as 3 separate memories.

---

## Root Cause Analysis

### When Does Deduplication Happen?

Deduplication happens at **store time**, not load time:

```
Message arrives → Extract memories → Deduplicate → Store in Redis
                                         ↑
                                   (This is where it fails)
```

### Why Current Deduplication Fails

| Method | Threshold/Logic | Why It Misses Tyler's Duplicates |
|--------|-----------------|----------------------------------|
| `deduplicate_by_id` | Exact ID match | Each extraction creates new ULID |
| `deduplicate_by_hash` | SHA256 of text | Different text = different hash |
| `deduplicate_by_semantic_search` | Vector distance < 0.2 | Paraphrases have distance 0.3-0.5 |

### The Core Issue: Semantic Threshold Too Strict

```python
# Current setting in long_term_memory.py
vector_distance_threshold: float = 0.2  # Very strict!
```

For the example memories:
- "User likes coffee, flat white usually"
- "User loves coffee, especially flat white"

These might have a vector distance of ~0.35 (similar but not identical phrasing), so they **pass through** the 0.2 threshold and both get stored.

---

## Two Tiers: Store-Time vs Load-Time Deduplication

### Tier 1: Store-Time Deduplication (Current - Needs Improvement)

**When:** Before inserting a new memory into Redis

**Current Flow:**
```
New Memory → Check ID → Check Hash → Check Semantic → Store
                ↓           ↓              ↓
            (miss)      (miss)      (miss at 0.2)
```

**Problem:** All three checks miss paraphrased duplicates.

### Tier 2: Load-Time Deduplication (✅ Implemented)

**When:** When retrieving memories for a prompt/context

**Implemented Flow:**
```
Query → Vector Search → Get Top K → Cluster by Similarity → LLM Verify → Merge Clusters → Return
                                          ↑                      ↑              ↑
                                    (threshold 0.4)    (confirm same fact)  (coherent text)
```

**Why it helps:** Even if duplicates exist in storage, we merge them when serving to the agent.

**Configuration:**
```python
# In settings (config.py)
enable_load_time_deduplication: bool = True   # Enable/disable feature
load_time_dedup_threshold: float = 0.4        # Clustering threshold
load_time_dedup_use_llm: bool = True          # Use LLM for verification

# Per-request override (API/MCP)
results = await search_long_term_memory(
    text="user preferences",
    deduplicate=True,           # Override setting
    dedup_threshold=0.4,        # Override threshold
    dedup_use_llm=True          # Override LLM usage
)
```

---

## Detailed Solution: How Each Tier Solves the Problem

### Scenario: Three Memories Already in Redis

```
Memory 1: "User likes coffee, flat white usually"
Memory 2: "They are a coffee enthusiast, favorite coffee is flatwhite"
Memory 3: "User loves coffee, especially flat white"
```

### Tier 1 Fix: Aggressive Store-Time Deduplication

**Change 1:** Lower semantic threshold from 0.2 → 0.4

```python
# When Memory 2 arrives for storage:
existing = semantic_search("They are a coffee enthusiast...", threshold=0.4)
# Returns: Memory 1 (distance ~0.35)

# LLM merges them:
merged = merge_memories_with_llm([Memory1, Memory2])
# Result: "User is a coffee enthusiast who prefers flat white"

# Delete Memory 1, store merged memory
```

**Change 2:** LLM-based duplicate detection (more accurate than vectors)

```python
async def is_duplicate_fact(new_memory: str, existing_memories: list[str]) -> bool:
    prompt = f"""Do any of these existing memories express the same fact?
    
    New memory: {new_memory}
    
    Existing memories:
    {existing_memories}
    
    Answer YES or NO and which memory ID if yes."""
    ...
```

### Tier 2 Fix: Load-Time Deduplication

Even if all 3 memories exist, deduplicate when serving:

```python
async def search_with_deduplication(query: str, limit: int = 10):
    # Get more results than needed
    raw_results = await search_long_term_memories(query, limit=limit * 3)
    
    # Group by semantic similarity
    clusters = cluster_by_similarity(raw_results, threshold=0.4)
    
    # Take best from each cluster
    deduplicated = [pick_best(cluster) for cluster in clusters]
    
    return deduplicated[:limit]
```

**Result for query "What coffee does user like?":**
- Raw results: [Memory1, Memory2, Memory3] (all ~0.9 similarity)
- Clustered: [[Memory1, Memory2, Memory3]] (one cluster)
- Deduplicated: [Memory1] (best/most complete)

---

## Implementation Recommendations

### Priority 1: Fix Store-Time Deduplication

1. **Make threshold configurable** in `config.py`:
   ```python
   semantic_dedup_threshold: float = 0.35
   ```

2. **Add LLM verification** before storing:
   ```python
   if vector_distance < 0.5:  # Candidate duplicate
       is_dup = await llm_verify_duplicate(new, existing)
       if is_dup:
           merged = await merge_memories_with_llm([new, existing])
   ```

### Priority 2: Add Load-Time Deduplication

1. **Cluster search results** before returning
2. **Merge similar memories** in response
3. **Cache merged results** to avoid repeated computation

### Priority 3: Background Compaction Improvements

The existing `compact_long_term_memories()` runs periodically but uses the same strict threshold. Loosening it would clean up existing duplicates:

```python
async def compact_long_term_memories(
    vector_distance_threshold: float = 0.35,  # Was 0.2
    ...
)
```

---

## Summary

| Tier | When | Current State | Fix |
|------|------|---------------|-----|
| **Store-Time** | Before insert | Threshold too strict (0.2) | Lower to 0.35-0.4, add LLM verification |
| **Load-Time** | Before serving | Not implemented | Add result clustering/merging |
| **Background** | Periodic compaction | Threshold too strict | Lower threshold, run more frequently |

The ideal solution uses **both tiers**:
- Store-time catches most duplicates upfront
- Load-time handles edge cases and legacy data

---

## ACE Paper Insights (Agentic Context Engineering)

The [ACE paper](https://arxiv.org/abs/2510.04618) provides additional strategies for memory management:

### Key Concepts Applicable to Deduplication

1. **Incremental Delta Updates**: Instead of storing full memories, store "deltas" that modify existing knowledge
   - Example: First memory = "User likes flat white"
   - Delta = "Confirmed: still prefers flat white" (updates confidence, not new entry)

2. **Helpful/Harmful Counters**: Track which memories are useful
   - If duplicate memories keep getting retrieved, that's a signal to merge them
   - Low-access memories can be pruned

3. **Grow-and-Refine Mechanism**:
   - **Grow**: Add new facts
   - **Refine**: Merge/update existing facts when new info arrives
   - Current system only "grows", doesn't "refine" well

4. **Context Collapse Prevention**: Avoid over-summarization that loses details
   - When merging memories, preserve important nuances
   - "User likes flat white" + "User broke pour-over" → Keep both facts, don't over-merge

### Proposed ACE-Inspired Enhancements

```python
class MemoryRecord:
    # Existing fields...

    # ACE-inspired additions:
    access_count: int = 0           # How often retrieved
    helpful_count: int = 0          # Positive feedback
    harmful_count: int = 0          # Negative feedback
    confidence: float = 1.0         # Certainty level
    supersedes: list[str] = []      # IDs of memories this one replaced
```

This enables:
- **Confidence-weighted retrieval**: Prefer high-confidence memories
- **Feedback-driven pruning**: Remove memories marked harmful
- **Lineage tracking**: Know which memories were merged into which

---

## Hybrid Search for Deduplication

### Current State: Vector-Only Search

The codebase currently uses **pure vector search** for deduplication in `long_term_memory.py` lines 1187-1194:

```python
# deduplicate_by_semantic_search() - line 1187
search_result = await adapter.search_memories(
    query=memory.text,  # <-- Embedded to vector, then KNN search
    namespace=namespace_filter,
    user_id=user_id_filter,
    session_id=session_id_filter,
    distance_threshold=vector_distance_threshold,  # Default 0.2
    limit=10,
)
```

**This is the exact location where hybrid search would help most.**

The index schema shows `text` is stored but **not indexed for full-text search**:

```python
# vectorstore_factory.py - current schema
metadata_schema = [
    {"name": "session_id", "type": "tag"},
    {"name": "topics", "type": "tag"},      # TAG = exact match only
    {"name": "entities", "type": "tag"},
    # ... no TextField for full-text search on content
]
```

### Why Hybrid Search Would Help

| Search Type | Catches | Misses |
|-------------|---------|--------|
| **Vector-only** | Semantically similar | Exact keyword matches with different context |
| **Keyword-only** | Exact term matches | Paraphrases, synonyms |
| **Hybrid** | Both! | Much less |

**Example: "User likes flat white coffee"**

| Candidate Memory | Vector Distance | Keyword Match | Hybrid Score |
|------------------|-----------------|---------------|--------------|
| "User loves flat white" | 0.25 | "flat white" ✓ | HIGH |
| "User prefers espresso drinks" | 0.35 | None | MEDIUM |
| "flat white is their favorite" | 0.30 | "flat white" ✓ | HIGH |

With hybrid search, the keyword match on "flat white" **boosts** the score, making duplicates easier to detect.

### Implementation: RedisVL Native Hybrid Search

RedisVL **already supports hybrid search** via `Text` filters combined with `VectorQuery` and BM25 scoring!

**Step 1: Update schema in `vectorstore_factory.py`**

```python
metadata_schema = [
    # Existing fields...
    {"name": "text", "type": "text"},  # ADD THIS - enables FT search
]
```

**Step 2: Use RedisVL hybrid query with BM25 scoring**

```python
from redisvl.query import VectorQuery
from redisvl.query.filter import Text

async def deduplicate_with_hybrid_search(
    memory: MemoryRecord,
    vector_threshold: float = 0.4,
) -> tuple[MemoryRecord | None, bool]:
    """Use hybrid search for better duplicate detection."""

    # Extract key terms for keyword matching
    key_terms = extract_key_terms(memory.text)  # ["flat", "white", "coffee"]

    # Build optional text filter (~ means optional, boosts but doesn't require)
    # This combines vector similarity with BM25 text relevance
    text_query = "|".join(key_terms)  # "flat|white|coffee"

    v = VectorQuery(
        vector=embedding,
        vector_field_name="vector",
        return_fields=["id_", "text", "namespace", "user_id"],
        distance_threshold=vector_threshold,
        num_results=10,
    )

    # Add optional text filter - boosts matches but doesn't exclude non-matches
    v.set_filter(f"(~(@text:{text_query}))")
    v.scorer("BM25").with_scores()  # Enable BM25 scoring

    results = await index.query(v)
    # Results now have both vector_distance AND BM25 score
    # Higher combined score = more likely duplicate
```

**Step 3: Combine scores for final ranking**

```python
def compute_hybrid_score(result: dict) -> float:
    """Combine vector distance and BM25 score."""
    vector_dist = float(result.get("vector_distance", 1.0))
    bm25_score = float(result.get("score", 0.0))

    # Lower distance = more similar, higher BM25 = more keyword overlap
    # Normalize and combine (weights are tunable)
    vector_sim = 1.0 - vector_dist  # Convert distance to similarity

    return 0.7 * vector_sim + 0.3 * bm25_score

# Sort by hybrid score
ranked = sorted(results, key=compute_hybrid_score, reverse=True)
```

**Key RedisVL Features Used:**
- `~` prefix: Makes text filter **optional** (boosts but doesn't exclude)
- `|` operator: OR between terms ("flat|white|coffee")
- `.scorer("BM25")`: Enables BM25 relevance scoring
- `.with_scores()`: Returns scores in results

### Hybrid Search Benefits for Deduplication

| Scenario | Vector-Only | Hybrid |
|----------|-------------|--------|
| "User likes flat white" vs "User loves flat white" | Might miss (dist ~0.25) | Catches (keyword boost) |
| "Coffee preference: flat white" vs "Flat white is preferred" | Might miss (dist ~0.35) | Catches (keyword boost) |
| "User broke pour-over" vs "Pour-over set is broken" | Might miss (dist ~0.30) | Catches (keyword boost) |

### Trade-offs

| Aspect | Vector-Only | Hybrid |
|--------|-------------|--------|
| **Index size** | Smaller | Larger (text index) |
| **Query latency** | Faster | Slightly slower |
| **Accuracy** | Good for paraphrases | Better for exact + paraphrases |
| **Implementation** | Current | Requires schema change |

### Recommendation

1. **Short-term**: Lower vector threshold (0.2 → 0.35) - quick win, no schema change
2. **Medium-term**: Add `text` as TextField to schema + use hybrid search with BM25
3. **Long-term**: Tune hybrid weights based on deduplication accuracy metrics

### Migration Required

Adding `{"name": "text", "type": "text"}` to the schema requires:
1. Update `vectorstore_factory.py` metadata_schema
2. Run `uv run agent-memory rebuild-index` to recreate the index
3. Re-index existing memories (or let compaction gradually re-process them)

---

## Implementation Status (Completed)

### Changes Made

The deduplication system has been implemented with a **two-tier approach**:

#### Tier 1: Store-Time Deduplication

**Configuration Settings (`config.py`):**
```python
# Deduplication Settings (Store-Time - Tier 1)
deduplication_distance_threshold: float = 0.35  # Distance threshold for semantic similarity
```

**Deduplication Logic (`long_term_memory.py`):**
- `deduplicate_by_semantic_search()` uses vector search with threshold 0.35
- Catches paraphrased duplicates at write time
- Uses LLM to merge similar memories

#### Tier 2: Load-Time Deduplication

**Configuration Settings (`config.py`):**
```python
# Load-time deduplication settings (Tier 2)
enable_load_time_deduplication: bool = True
load_time_dedup_threshold: float = 0.4
load_time_dedup_use_llm: bool = True
```

**New Functions (`long_term_memory.py`):**
- `cluster_memories_by_similarity()` - Groups memories by vector similarity
- `verify_duplicates_with_llm()` - LLM verifies which memories represent the same fact
- `merge_memory_cluster_for_display()` - Merges clusters into coherent display text
- `deduplicate_search_results()` - Main orchestration function

### How It Works

**Store-Time (Tier 1):**
When a new memory is stored:
1. Vector search finds semantically similar memories (threshold 0.35)
2. If duplicates found, LLM merges them into a single memory
3. Only the merged memory is stored

**Load-Time (Tier 2):**
When memories are retrieved:
1. Fetch more results than needed (limit * 3)
2. Cluster by vector similarity (threshold 0.4)
3. LLM verifies which memories represent the same fact
4. Merge each cluster into a single coherent memory
5. Return deduplicated results

### Example: Coffee Preference Deduplication

**Before (threshold=0.12):**
```
Memory 1: "User likes coffee, flat white usually"     → Stored
Memory 2: "Coffee enthusiast, favorite is flatwhite"  → Vector dist 0.28 > 0.12 → Stored (DUPLICATE!)
Memory 3: "User loves coffee, especially flat white"  → Vector dist 0.25 > 0.12 → Stored (DUPLICATE!)
```

**After (threshold=0.35):**
```
Memory 1: "User likes coffee, flat white usually"     → Stored
Memory 2: "Coffee enthusiast, favorite is flatwhite"  → Vector dist 0.28 < 0.35 → MERGED with Memory 1
Memory 3: "User loves coffee, especially flat white"  → Vector dist 0.25 < 0.35 → MERGED with Memory 1
```

### Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `DEDUPLICATION_DISTANCE_THRESHOLD` | `0.35` | Store-time distance threshold |
| `ENABLE_LOAD_TIME_DEDUPLICATION` | `true` | Enable load-time deduplication |
| `LOAD_TIME_DEDUP_THRESHOLD` | `0.4` | Load-time clustering threshold |
| `LOAD_TIME_DEDUP_USE_LLM` | `true` | Use LLM for verification |

---

## Historical Note: Hybrid Search (Removed)

> **Note:** Hybrid search was investigated as a potential solution for deduplication but was ultimately removed.
> The vector-only approach with a properly tuned threshold (0.35) proved sufficient for catching paraphrased duplicates.
> Hybrid search added complexity without significant benefit for the deduplication use case.

The hybrid search code was implemented but later removed because:
1. Vector-only search with threshold 0.35 catches paraphrased duplicates effectively
2. Hybrid search added significant code complexity
3. The two-tier approach (store-time + load-time) provides better coverage than hybrid search alone
