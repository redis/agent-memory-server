# Summary Views

Summary Views provide **configurable, partitioned summaries** of long-term memories. They enable applications to generate and cache concise overviews of memory collections, organized by user, namespace, session, or other groupings.

## Overview

A Summary View is a configuration that defines:

- **Source**: Which memory pool to summarize (currently `long_term`)
- **Group By**: How to partition memories (e.g., by `user_id`, `namespace`)
- **Filters**: Static filters applied to every run
- **Time Window**: Optional rolling window of days to include
- **Prompt**: Custom summarization instructions

When run, the view fetches matching memories, partitions them by the specified fields, and generates an LLM summary for each partition.

| Feature | Details |
|---------|---------|
| **Source** | Long-term memory (working memory planned) |
| **Partitioning** | By user_id, namespace, session_id, memory_type |
| **Execution** | On-demand or continuous (background) |
| **Storage** | Cached partition results in Redis |
| **Summarization** | LLM-powered with custom prompts |

## Use Cases

### 1. User Profile Summaries
Generate a rolling summary of what the system knows about each user:

```json
{
  "name": "user_profile_30d",
  "source": "long_term",
  "group_by": ["user_id"],
  "filters": {"memory_type": "semantic"},
  "time_window_days": 30
}
```

### 2. Namespace Knowledge Digests
Summarize memories within each namespace:

```json
{
  "name": "namespace_digest",
  "source": "long_term",
  "group_by": ["namespace"],
  "filters": {},
  "time_window_days": 7
}
```

### 3. Session Recaps
Create summaries for each conversation session:

```json
{
  "name": "session_recap",
  "source": "long_term",
  "group_by": ["session_id"],
  "filters": {"memory_type": "episodic"}
}
```

## API Endpoints

### Create a Summary View

```http
POST /v1/summary-views
Content-Type: application/json

{
  "name": "ltm_by_user_30d",
  "source": "long_term",
  "group_by": ["user_id"],
  "filters": {"memory_type": "semantic"},
  "time_window_days": 30,
  "continuous": false,
  "prompt": "Summarize key facts and preferences for this user.",
  "model_name": "gpt-4o-mini"
}
```

Response:
```json
{
  "id": "01J5X...",
  "name": "ltm_by_user_30d",
  "source": "long_term",
  "group_by": ["user_id"],
  "filters": {"memory_type": "semantic"},
  "time_window_days": 30,
  "continuous": false,
  "prompt": "Summarize key facts and preferences for this user.",
  "model_name": "gpt-4o-mini"
}
```

### List Summary Views

```http
GET /v1/summary-views
```

### Get a Summary View

```http
GET /v1/summary-views/{view_id}
```

### Delete a Summary View

```http
DELETE /v1/summary-views/{view_id}
```

### Run a Single Partition

Synchronously compute a summary for one specific partition:

```http
POST /v1/summary-views/{view_id}/partitions/run
Content-Type: application/json

{
  "group": {"user_id": "alice"}
}
```

Response:
```json
{
  "view_id": "01J5X...",
  "group": {"user_id": "alice"},
  "summary": "Alice prefers dark mode and uses Python for ML projects...",
  "memory_count": 42,
  "computed_at": "2024-01-15T10:30:00Z"
}
```

### Run All Partitions (Async)

Trigger a full background recompute of all partitions:

```http
POST /v1/summary-views/{view_id}/run
Content-Type: application/json

{
  "task_id": "optional-client-provided-id"
}
```

Returns a Task that can be polled:
```json
{
  "id": "task_01J5X...",
  "type": "summary_view_full_run",
  "status": "pending",
  "view_id": "01J5X..."
}
```

### List Partition Results

Retrieve cached summaries with optional filtering:

```http
GET /v1/summary-views/{view_id}/partitions?user_id=alice
```

## Configuration Options

### SummaryView Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | No | Human-readable name |
| `source` | string | Yes | `"long_term"` (only supported source) |
| `group_by` | list | No | Fields to partition by: `user_id`, `namespace`, `session_id`, `memory_type` |
| `filters` | dict | No | Static filters: `user_id`, `namespace`, `session_id`, `memory_type` |
| `time_window_days` | int | No | Rolling window cutoff (memories newer than N days) |
| `continuous` | bool | No | If true, background workers refresh periodically |
| `prompt` | string | No | Custom summarization instructions |
| `model_name` | string | No | LLM model override (defaults to `fast_model` setting) |

### Supported Group By and Filter Keys

Both `group_by` and `filters` support:

- `user_id` - Partition/filter by user
- `namespace` - Partition/filter by namespace
- `session_id` - Partition/filter by session
- `memory_type` - Partition/filter by type (`semantic`, `episodic`, `message`)

## Continuous Mode

When `continuous: true`, background workers periodically refresh all partitions (default: every 60 minutes). This keeps summaries up-to-date without manual intervention.

```json
{
  "name": "always_fresh_user_summaries",
  "source": "long_term",
  "group_by": ["user_id"],
  "continuous": true,
  "time_window_days": 7
}
```

## Custom Prompts

Override the default summarization behavior:

```json
{
  "name": "technical_summary",
  "source": "long_term",
  "group_by": ["user_id"],
  "prompt": "Focus on technical skills, programming languages, and project experience. Output as bullet points."
}
```

Default prompt when not specified:
> "You are a summarization assistant. Given a set of long-term memories, produce a concise summary that highlights key facts, stable preferences, and important events relevant to the group."

## Task Polling

Full view runs execute asynchronously. Poll the task endpoint for status:

```http
GET /v1/tasks/{task_id}
```

Possible statuses: `pending`, `running`, `success`, `failed`

## Token-Aware Summarization

The summarization prompt is automatically truncated to fit within the model's context window:

- Uses `summarization_threshold` setting to budget prompt tokens
- Individual memories are capped to prevent one long memory dominating
- Reports how many memories were included vs. total available

## Related Documentation

- [Long-term Memory](long-term-memory.md) - The memory source for summary views
- [Working Memory](working-memory.md) - Session-scoped memory (future summary view source)
- [Configuration](configuration.md) - Server settings including `fast_model` and `summarization_threshold`
