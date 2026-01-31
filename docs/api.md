# REST API Reference

This documentation is auto-generated from the OpenAPI specification.
For interactive docs, run the server and visit `/docs`.

**API Version:** 0.12.7

## Table of Contents

- [Health](#health)
- [Working Memory](#working-memory)
- [Long-Term Memory](#long-term-memory)
- [Memory Prompt](#memory-prompt)
- [Summary Views](#summary-views)
- [Tasks](#tasks)
- [Data Models](#data-models)
- [Filter Options](#filter-options)

## Health

### GET `/v1/health`

**Get Health**

Health check endpoint

Returns:
    HealthCheckResponse with current timestamp

**Response:** [`HealthCheckResponse`](#data-models)

---

## Working Memory

### GET `/v1/working-memory/`

**List Sessions**

Get a list of session IDs, with optional pagination.

Args:
    options: Query parameters (limit, offset, namespace, user_id)

Returns:
    List of session IDs

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No |  |
| `offset` | integer | No |  |
| `namespace` | string (optional) | No |  |
| `user_id` | string (optional) | No |  |

**Response:** [`SessionListResponse`](#data-models)

---

### GET `/v1/working-memory/{session_id}`

**Get Working Memory**

Get working memory for a session.

This includes stored conversation messages, context, and structured memory records.
If the messages exceed the token limit, older messages will be truncated.

Args:
    session_id: The session ID
    user_id: The user ID to retrieve working memory for
    namespace: The namespace to use for the session
    model_name: The client's LLM model name (will determine context window size if provided)
    context_window_max: Direct specification of the context window max tokens (overrides model_name)
    recent_messages_limit: Maximum number of recent messages to return (most recent first)

Returns:
    Working memory containing messages, context, and structured memory records

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes |  |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string (optional) | No |  |
| `namespace` | string (optional) | No |  |
| `model_name` | string (optional) | No |  |
| `context_window_max` | integer (optional) | No |  |
| `recent_messages_limit` | integer (optional) | No |  |

**Response:** [`WorkingMemoryResponse`](#data-models)

---

### PUT `/v1/working-memory/{session_id}`

**Put Working Memory**

Set working memory for a session. Replaces existing working memory.

The session_id comes from the URL path, not the request body.
If the token count exceeds the context window threshold, messages will be summarized
immediately and the updated memory state returned to the client.

NOTE on context_percentage_* fields:
The response includes `context_percentage_total_used` and `context_percentage_until_summarization`
fields that show token usage. These fields will be `null` unless you provide either:
- `model_name` query parameter (e.g., `?model_name=gpt-4o-mini`)
- `context_window_max` query parameter (e.g., `?context_window_max=500`)

Args:
    session_id: The session ID (from URL path)
    memory: Working memory data to save (session_id not required in body)
    model_name: The client's LLM model name for context window determination
    context_window_max: Direct specification of context window max tokens (overrides model_name)
    background_tasks: DocketBackgroundTasks instance (injected automatically)
    response: FastAPI Response object for setting headers

Returns:
    Updated working memory (potentially with summary if tokens were condensed).
    Includes context_percentage_total_used and context_percentage_until_summarization
    if model information is provided.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes |  |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string (optional) | No |  |
| `context_window_max` | integer (optional) | No |  |

**Request Body:** [`UpdateWorkingMemory`](#data-models)

**Response:** [`WorkingMemoryResponse`](#data-models)

---

### DELETE `/v1/working-memory/{session_id}`

**Delete Working Memory**

Delete working memory for a session.

This deletes all stored memory (messages, context, structured memories) for a session.

Args:
    session_id: The session ID
    user_id: Optional user ID for the session
    namespace: Optional namespace for the session

Returns:
    Acknowledgement response

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes |  |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string (optional) | No |  |
| `namespace` | string (optional) | No |  |

**Response:** [`AckResponse`](#data-models)

---

## Long-Term Memory

### POST `/v1/long-term-memory/forget`

**Forget Endpoint**

Run a forgetting pass with the provided policy. Returns summary data.

This is an admin-style endpoint; auth is enforced by the standard dependency.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string (optional) | No |  |
| `user_id` | string (optional) | No |  |
| `session_id` | string (optional) | No |  |
| `limit` | integer | No |  |
| `dry_run` | boolean | No |  |

**Request Body:** [`Body_forget_endpoint_v1_long_term_memory_forget_post`](#data-models)

---

### POST `/v1/long-term-memory/`

**Create Long Term Memory**

Create a long-term memory

Args:
    payload: Long-term memory payload
    background_tasks: DocketBackgroundTasks instance (injected automatically)

Returns:
    Acknowledgement response

**Request Body:** [`CreateMemoryRecordRequest`](#data-models)

**Response:** [`AckResponse`](#data-models)

---

### POST `/v1/long-term-memory/search`

**Search Long Term Memory**

Run a semantic search on long-term memory with filtering options.

Args:
    payload: Search payload with filter objects for precise queries
    optimize_query: Whether to optimize the query for vector search using a fast model (default: False)

Returns:
    List of search results

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `optimize_query` | boolean | No |  |

**Request Body:** [`SearchRequest`](#data-models)

**Response:** [`MemoryRecordResultsResponse`](#data-models)

---

### DELETE `/v1/long-term-memory`

**Delete Long Term Memory**

Delete long-term memories by ID

Args:
    memory_ids: List of memory IDs to delete (passed as query parameters)

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_ids` | array[string] | No |  |

**Response:** [`AckResponse`](#data-models)

---

### GET `/v1/long-term-memory/{memory_id}`

**Get Long Term Memory**

Get a long-term memory by its ID

Args:
    memory_id: The ID of the memory to retrieve

Returns:
    The memory record if found

Raises:
    HTTPException: 404 if memory not found, 400 if long-term memory disabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | string | Yes |  |

**Response:** [`MemoryRecord`](#data-models)

---

### PATCH `/v1/long-term-memory/{memory_id}`

**Update Long Term Memory**

Update a long-term memory by its ID

Args:
    memory_id: The ID of the memory to update
    updates: The fields to update

Returns:
    The updated memory record

Raises:
    HTTPException: 404 if memory not found, 400 if invalid fields or long-term memory disabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | string | Yes |  |

**Request Body:** [`EditMemoryRecordRequest`](#data-models)

**Response:** [`MemoryRecord`](#data-models)

---

## Memory Prompt

### POST `/v1/memory/prompt`

**Memory Prompt**

Hydrate a user query with memory context and return a prompt
ready to send to an LLM.

`query` is the query for vector search that the caller of this API wants to use to find
relevant context. If `session_id` is provided and matches an existing
session, the resulting prompt will include those messages as the immediate
history of messages leading to a message containing `query`.

If `long_term_search_payload` is provided, the resulting prompt will include
relevant long-term memories found via semantic search with the options
provided in the payload.

Args:
    params: MemoryPromptRequest
    optimize_query: Whether to optimize the query for vector search using a fast model (default: False)

Returns:
    List of messages to send to an LLM, hydrated with relevant memory context

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `optimize_query` | boolean | No |  |

**Request Body:** [`MemoryPromptRequest`](#data-models)

**Response:** [`MemoryPromptResponse`](#data-models)

---

## Summary Views

### GET `/v1/summary-views`

**List Summary Views Endpoint**

List all registered SummaryViews.

Filtering by source/continuous can be added later if needed.

---

### POST `/v1/summary-views`

**Create Summary View**

Create a new SummaryView configuration.

The server assigns an ID; the configuration can then be run on-demand or
by background workers.

**Request Body:** [`CreateSummaryViewRequest`](#data-models)

**Response:** [`SummaryView`](#data-models)

---

### GET `/v1/summary-views/{view_id}`

**Get Summary View**

Get a SummaryView configuration by ID.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `view_id` | string | Yes |  |

**Response:** [`SummaryView`](#data-models)

---

### DELETE `/v1/summary-views/{view_id}`

**Delete Summary View Endpoint**

Delete a SummaryView configuration.

Stored partition summaries are left as-is for now.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `view_id` | string | Yes |  |

**Response:** [`AckResponse`](#data-models)

---

### POST `/v1/summary-views/{view_id}/partitions/run`

**Run Summary View Partition**

Synchronously compute a summary for a single partition of a view.

For long-term memory views this will query the underlying memories
and run a real summarization. For other sources it currently returns
a placeholder summary.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `view_id` | string | Yes |  |

**Request Body:** [`RunSummaryViewPartitionRequest`](#data-models)

**Response:** [`SummaryViewPartitionResult`](#data-models)

---

### GET `/v1/summary-views/{view_id}/partitions`

**List Summary View Partitions**

List materialized partition summaries for a SummaryView.

This does not trigger recomputation; it simply reads stored
SummaryViewPartitionResult entries from Redis. Optional query
parameters filter by group fields when present.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `view_id` | string | Yes |  |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string (optional) | No |  |
| `namespace` | string (optional) | No |  |
| `session_id` | string (optional) | No |  |
| `memory_type` | string (optional) | No |  |

---

### POST `/v1/summary-views/{view_id}/run`

**Run Summary View Full**

Trigger an asynchronous full recompute of all partitions for a view.

Returns a Task that can be polled for status. The actual work is
performed by a Docket worker running refresh_summary_view.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `view_id` | string | Yes |  |

**Request Body:** [`RunSummaryViewRequest`](#data-models)

**Response:** [`Task`](#data-models)

---

## Tasks

### GET `/v1/tasks/{task_id}`

**Get Task Status**

Get the status of a background Task by ID.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_id` | string | Yes |  |

**Response:** [`Task`](#data-models)

---

## Data Models

Key request and response models used by the API.

### WorkingMemoryResponse

Response containing working memory

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array[MemoryMessage] | No | Conversation messages (role/content pairs) |
| `memories` | array[MemoryRecord | ClientMemoryRecord] | No | Structured memory records for promotion to long-term storage |
| `data` | object | null | No | Arbitrary JSON data storage (key-value pairs) |
| `context` | string | null | No | Summary of past session messages if server has auto-summariz |
| `user_id` | string | null | No | Optional user ID for the working memory |
| `tokens` | integer | No | Optional number of tokens in the working memory |
| `session_id` | string | Yes |  |
| `namespace` | string | null | No | Optional namespace for the working memory |
| `long_term_memory_strategy` | MemoryStrategyConfig | No | Configuration for memory extraction strategy when promoting  |
| `ttl_seconds` | integer | null | No | TTL for the working memory in seconds |
| `last_accessed` | string | No | Datetime when the working memory was last accessed |
| `created_at` | string | No | Datetime when the working memory was created |
| `updated_at` | string | No | Datetime when the working memory was last updated |
| `context_percentage_total_used` | number | null | No | Percentage of total context window currently used (0-100) |
| `context_percentage_until_summarization` | number | null | No | Percentage until auto-summarization triggers (0-100, reaches |
| `new_session` | boolean | null | No | True if session was created, False if existing session was f |
| `unsaved` | boolean | null | No | True if this session data has not been persisted to Redis ye |

### UpdateWorkingMemory

Working memory update payload for PUT requests - session_id comes from URL path

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array[MemoryMessage] | No | Conversation messages (role/content pairs) |
| `memories` | array[MemoryRecord | ClientMemoryRecord] | No | Structured memory records for promotion to long-term storage |
| `data` | object | null | No | Arbitrary JSON data storage (key-value pairs) |
| `context` | string | null | No | Summary of past session messages if server has auto-summariz |
| `user_id` | string | null | No | Optional user ID for the working memory |
| `tokens` | integer | No | Optional number of tokens in the working memory |
| `namespace` | string | null | No | Optional namespace for the working memory |
| `long_term_memory_strategy` | MemoryStrategyConfig | No | Configuration for memory extraction strategy when promoting  |
| `ttl_seconds` | integer | null | No | TTL for the working memory in seconds |
| `last_accessed` | string | No | Datetime when the working memory was last accessed |
| `created_at` | string | No | Datetime when the working memory was created |
| `updated_at` | string | No | Datetime when the working memory was last updated |

### MemoryMessage

A message in the memory system

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes |  |
| `content` | string | Yes |  |
| `id` | string | No | Unique identifier for the message (auto-generated if not pro |
| `created_at` | string | No | Timestamp when the message was created (should be provided b |
| `persisted_at` | string | null | No | Server-assigned timestamp when message was persisted to long |
| `discrete_memory_extracted` | enum: `t`, `f` | No | Whether memory extraction has run for this message |

### MemoryRecord

A memory record

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Client-provided ID for deduplication and overwrites |
| `text` | string | Yes |  |
| `session_id` | string | null | No | Optional session ID for the memory record |
| `user_id` | string | null | No | Optional user ID for the memory record |
| `namespace` | string | null | No | Optional namespace for the memory record |
| `last_accessed` | string | No | Datetime when the memory was last accessed |
| `created_at` | string | No | Datetime when the memory was created |
| `updated_at` | string | No | Datetime when the memory was last updated |
| `pinned` | boolean | No | Whether this memory is pinned and should not be auto-deleted |
| `access_count` | integer | No | Number of times this memory has been accessed (best-effort,  |
| `topics` | array[string] | null | No | Optional topics for the memory record |
| `entities` | array[string] | null | No | Optional entities for the memory record |
| `memory_hash` | string | null | No | Hash representation of the memory for deduplication |
| `discrete_memory_extracted` | enum: `t`, `f` | No | Whether memory extraction has run for this memory |
| `memory_type` | agent_memory_server__models__MemoryTypeEnum | No | Type of memory |
| `persisted_at` | string | null | No | Server-assigned timestamp when memory was persisted to long- |
| `extracted_from` | array[string] | null | No | List of message IDs that this memory was extracted from |
| `event_date` | string | null | No | Date/time when the event described in this memory occurred ( |
| `extraction_strategy` | string | No | Memory extraction strategy used when this was promoted from  |
| `extraction_strategy_config` | object | No | Configuration for the extraction strategy used |

### CreateMemoryRecordRequest

Payload for creating memory records

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `memories` | array[ExtractedMemoryRecord] | Yes |  |
| `deduplicate` | boolean | No | Whether to deduplicate memories before indexing |

### EditMemoryRecordRequest

Payload for editing a memory record

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | null | No | Updated text content for the memory |
| `topics` | array[string] | null | No | Updated topics for the memory |
| `entities` | array[string] | null | No | Updated entities for the memory |
| `memory_type` | agent_memory_server__models__MemoryTypeEnum | null | No | Updated memory type (semantic, episodic, message) |
| `namespace` | string | null | No | Updated namespace for the memory |
| `user_id` | string | null | No | Updated user ID for the memory |
| `session_id` | string | null | No | Updated session ID for the memory |
| `event_date` | string | null | No | Updated event date for episodic memories |

### MemoryRecordResultsResponse

Response containing memory search results

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `memories` | array[MemoryRecordResult] | Yes |  |
| `total` | integer | Yes |  |
| `next_offset` | integer | null | No |  |

### SearchRequest

Payload for long-term memory search

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | null | No | Optional text to use for a semantic search |
| `session_id` | SessionId | null | No | Optional session ID to filter by |
| `namespace` | Namespace | null | No | Optional namespace to filter by |
| `topics` | Topics | null | No | Optional topics to filter by |
| `entities` | Entities | null | No | Optional entities to filter by |
| `created_at` | CreatedAt | null | No | Optional created at timestamp to filter by |
| `last_accessed` | LastAccessed | null | No | Optional last accessed timestamp to filter by |
| `user_id` | UserId | null | No | Optional user ID to filter by |
| `distance_threshold` | number | null | No | Optional distance threshold to filter by |
| `memory_type` | MemoryType | null | No | Optional memory type to filter by |
| `event_date` | EventDate | null | No | Optional event date to filter by (for episodic memories) |
| `limit` | integer | No | Optional limit on the number of results |
| `offset` | integer | No | Optional offset |
| `recency_boost` | boolean | null | No | Enable recency-aware re-ranking (defaults to enabled if None |
| `recency_semantic_weight` | number | null | No | Weight for semantic similarity |
| `recency_recency_weight` | number | null | No | Weight for recency score |
| `recency_freshness_weight` | number | null | No | Weight for freshness component |
| `recency_novelty_weight` | number | null | No | Weight for novelty (age) component |
| `recency_half_life_last_access_days` | number | null | No | Half-life (days) for last_accessed decay |
| `recency_half_life_created_days` | number | null | No | Half-life (days) for created_at decay |
| `server_side_recency` | boolean | null | No | If true, attempt server-side recency-aware re-ranking when s |

### MemoryPromptRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes |  |
| `session` | WorkingMemoryRequest | null | No |  |
| `long_term_search` | SearchRequest | boolean | No |  |

### MemoryPromptResponse

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array[Message | SystemMessage] | Yes |  |

### AckResponse

Generic acknowledgement response

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | string | Yes |  |

### SessionListResponse

Response containing a list of sessions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sessions` | array[string] | Yes |  |
| `total` | integer | Yes |  |

### HealthCheckResponse

Response for health check endpoint

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `now` | integer | Yes |  |

### SummaryView

Configuration for a summary view over memories.

A SummaryView fully specifies what pool of memories to summarize and how
to partition and filter them, so it can be run on-demand or by a
background worker without additional runtime parameters.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the summary view |
| `name` | string | null | No | Optional human-readable name for the view |
| `source` | enum: `long_term`, `working_memory` | Yes | Memory source to summarize. Currently only 'long_term' is su |
| `group_by` | array[string] | No | Fields used to partition summaries (e.g. ['user_id'], ['user |
| `filters` | object | No | Static filters applied to every run (e.g. memory_type, names |
| `time_window_days` | integer | null | No | If set, each run uses now() - time_window_days as a cutoff f |
| `continuous` | boolean | No | If true, background workers periodically refresh all partiti |
| `prompt` | string | null | No | Optional custom summarization instructions. If omitted, a se |
| `model_name` | string | null | No | Optional model override for summarization. Defaults to a fas |

### CreateSummaryViewRequest

Payload for creating a new SummaryView.

Same fields as SummaryView except for the server-assigned id.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | null | No | Optional human-readable name for the view |
| `source` | enum: `long_term`, `working_memory` | Yes | Memory source to summarize: long-term or working memory |
| `group_by` | array[string] | No | Fields used to partition summaries (e.g. ['user_id'], ['user |
| `filters` | object | No | Static filters applied to every run (e.g. memory_type, names |
| `time_window_days` | integer | null | No | If set, each run uses now() - time_window_days as a cutoff f |
| `continuous` | boolean | No | If true, background workers periodically refresh all partiti |
| `prompt` | string | null | No | Optional custom summarization instructions. If omitted, a se |
| `model_name` | string | null | No | Optional model override for summarization. Defaults to a fas |

### SummaryViewPartitionResult

Result of summarizing one partition of a SummaryView.

A partition is defined by a concrete combination of the view's
group_by fields, e.g. {"user_id": "alice"} or
{"user_id": "alice", "namespace": "chat"}.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `view_id` | string | Yes | ID of the SummaryView that produced this result |
| `group` | object | Yes | Concrete values for the view's group_by fields |
| `summary` | string | Yes | Summarized text for this partition |
| `memory_count` | integer | Yes | Number of memories that contributed to this summary |
| `computed_at` | string | No | When this summary was computed |

### Task

Client-visible background task tracked in Redis as JSON.

These tasks represent long-running operations such as a full recompute
of all partitions for a SummaryView.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique task identifier (client or server generated) |
| `type` | TaskTypeEnum | Yes | Type of task, e.g. summary_view_full_run |
| `status` | TaskStatusEnum | No | Current task status |
| `view_id` | string | null | No | Associated SummaryView ID, if applicable |
| `created_at` | string | No | When the task record was created |
| `started_at` | string | null | No | When execution of the task actually started |
| `completed_at` | string | null | No | When execution of the task finished (success or failure) |
| `error_message` | string | null | No | Error message if the task failed |

## Filter Options

Filters are used in search requests to narrow down results.

### Tag Filters

Apply to: `session_id`, `namespace`, `topics`, `entities`, `user_id`, `memory_type`

```json
{ "eq": "value" }           // Equals
{ "ne": "value" }           // Not equals
{ "any": ["a", "b"] }       // Contains any of these
{ "all": ["a", "b"] }       // Contains all of these
```

### Numeric Filters

Apply to: `created_at`, `last_accessed`, `event_date` (Unix timestamps)

```json
{ "gt": 1704067200 }        // Greater than
{ "lt": 1704153600 }        // Less than
{ "gte": 1704067200 }       // Greater than or equal
{ "lte": 1704153600 }       // Less than or equal
{ "eq": 1704067200 }        // Equals
{ "between": [1704067200, 1704153600] }  // Between (inclusive)
```

### Example Search Request

```json
{
  "text": "user preferences for notifications",
  "limit": 10,
  "offset": 0,
  "namespace": { "eq": "production" },
  "session_id": { "eq": "session-abc123" },
  "topics": { "any": ["preferences", "settings"] },
  "memory_type": { "eq": "semantic" },
  "created_at": { "gte": 1704067200 }
}
```

## Authentication

All endpoints except `/v1/health` require authentication via Bearer token.
Set the `Authorization` header:

```
Authorization: Bearer <your-jwt-token>
```

For development, set `DISABLE_AUTH=true` to skip authentication.
