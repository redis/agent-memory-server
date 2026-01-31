/**
 * Data models for the Agent Memory Client.
 *
 * This module contains essential data models needed by the client.
 */

import { ulid } from "ulid";

/**
 * Supported LLM model names for context window determination
 */
export type ModelNameLiteral =
  | "gpt-3.5-turbo"
  | "gpt-3.5-turbo-16k"
  | "gpt-4"
  | "gpt-4-32k"
  | "gpt-4o"
  | "gpt-4o-mini"
  | "o1"
  | "o1-mini"
  | "o3-mini"
  | "gpt-5-mini"
  | "gpt-5-nano"
  | "gpt-5.1-chat-latest"
  | "gpt-5.2-chat-latest"
  | "text-embedding-ada-002"
  | "text-embedding-3-small"
  | "text-embedding-3-large"
  | "claude-3-opus-20240229"
  | "claude-3-sonnet-20240229"
  | "claude-3-haiku-20240307"
  | "claude-3-5-sonnet-20240620"
  | "claude-3-7-sonnet-20250219"
  | "claude-3-5-sonnet-20241022"
  | "claude-3-5-haiku-20241022"
  | "claude-3-7-sonnet-latest"
  | "claude-3-5-sonnet-latest"
  | "claude-3-5-haiku-latest"
  | "claude-3-opus-latest";

/**
 * Enum for memory types
 */
export enum MemoryTypeEnum {
  EPISODIC = "episodic",
  SEMANTIC = "semantic",
  MESSAGE = "message",
}

/**
 * Memory extraction strategy types
 */
export type MemoryStrategyType = "discrete" | "summary" | "preferences" | "custom";

/**
 * Configuration for memory extraction strategy
 */
export interface MemoryStrategyConfig {
  /** Type of memory extraction strategy to use */
  strategy?: MemoryStrategyType;
  /** Strategy-specific configuration options */
  config?: Record<string, unknown>;
}

/**
 * A message in the memory system
 */
export interface MemoryMessage {
  /** Message role (user, assistant, system, tool) */
  role: string;
  /** Message content */
  content: string;
  /** Unique identifier for the message */
  id?: string;
  /** Timestamp when the message was created */
  created_at?: string;
  /** Server-assigned timestamp when message was persisted */
  persisted_at?: string | null;
  /** Whether memory extraction has run for this message */
  discrete_memory_extracted?: "t" | "f";
}

/**
 * A memory record
 */
export interface MemoryRecord {
  /** Client-provided ID for deduplication and overwrites */
  id: string;
  /** Memory content text */
  text: string;
  /** Optional session ID for the memory record */
  session_id?: string | null;
  /** Optional user ID for the memory record */
  user_id?: string | null;
  /** Optional namespace for the memory record */
  namespace?: string | null;
  /** Datetime when the memory was last accessed */
  last_accessed?: string;
  /** Datetime when the memory was created */
  created_at?: string;
  /** Datetime when the memory was last updated */
  updated_at?: string;
  /** Optional topics for the memory record */
  topics?: string[] | null;
  /** Optional entities for the memory record */
  entities?: string[] | null;
  /** Hash representation of the memory for deduplication */
  memory_hash?: string | null;
  /** Whether memory extraction has run for this memory */
  discrete_memory_extracted?: "t" | "f";
  /** Type of memory */
  memory_type?: MemoryTypeEnum;
  /** Server-assigned timestamp when memory was persisted */
  persisted_at?: string | null;
  /** List of message IDs that this memory was extracted from */
  extracted_from?: string[] | null;
  /** Date/time when the event described in this memory occurred */
  event_date?: string | null;
}

/** JSON value types for working memory data */
export type JSONValue =
  | string
  | number
  | boolean
  | null
  | JSONValue[]
  | { [key: string]: JSONValue };

/**
 * Working memory for a session
 */
export interface WorkingMemory {
  /** Session ID (required) */
  session_id: string;
  /** Conversation messages */
  messages?: MemoryMessage[];
  /** Structured memory records */
  memories?: MemoryRecord[];
  /** Arbitrary JSON data storage */
  data?: Record<string, JSONValue> | null;
  /** Optional summary of past session messages */
  context?: string | null;
  /** Optional user ID */
  user_id?: string | null;
  /** Token count */
  tokens?: number;
  /** Optional namespace */
  namespace?: string | null;
  /** Configuration for memory extraction strategy */
  long_term_memory_strategy?: MemoryStrategyConfig;
  /** TTL for the working memory in seconds */
  ttl_seconds?: number | null;
  /** Datetime when the working memory was last accessed */
  last_accessed?: string;
}

/**
 * Response from working memory operations
 */
export interface WorkingMemoryResponse extends WorkingMemory {
  /** Percentage of total context window currently used (0-100) */
  context_percentage_total_used?: number | null;
  /** Percentage until auto-summarization triggers (0-100) */
  context_percentage_until_summarization?: number | null;
  /** True if session was created, False if existing */
  new_session?: boolean | null;
  /** True if this session data has not been persisted yet */
  unsaved?: boolean | null;
}

/** Generate a new ULID */
export function generateId(): string {
  return ulid();
}

/**
 * Generic acknowledgement response
 */
export interface AckResponse {
  status: string;
}

/**
 * Health check response
 */
export interface HealthCheckResponse {
  now: number;
}

/**
 * Response containing a list of sessions
 */
export interface SessionListResponse {
  sessions: string[];
  total: number;
}

/**
 * Result from a memory search
 */
export interface MemoryRecordResult extends MemoryRecord {
  /** Distance/similarity score */
  dist: number;
}

/**
 * Results from memory search operations
 */
export interface MemoryRecordResults {
  memories: MemoryRecordResult[];
  total: number;
  next_offset?: number | null;
}

/**
 * Client-side configuration for recency-aware ranking
 */
export interface RecencyConfig {
  /** Enable recency-aware re-ranking */
  recency_boost?: boolean | null;
  /** Weight for semantic similarity */
  semantic_weight?: number | null;
  /** Weight for recency score */
  recency_weight?: number | null;
  /** Weight for freshness component */
  freshness_weight?: number | null;
  /** Weight for novelty/age component */
  novelty_weight?: number | null;
  /** Half-life (days) for last_accessed decay */
  half_life_last_access_days?: number | null;
  /** Half-life (days) for created_at decay */
  half_life_created_days?: number | null;
  /** If true, attempt server-side recency ranking */
  server_side_recency?: boolean | null;
}

/**
 * Response from memory prompt endpoint
 */
export interface MemoryPromptResponse {
  messages: Record<string, unknown>[];
}

/**
 * Session parameters for memory prompt
 */
export interface SessionParams {
  session_id: string;
  user_id?: string | null;
  namespace?: string | null;
  model_name?: ModelNameLiteral | null;
  context_window_max?: number | null;
}

/**
 * Request for memory prompt endpoint
 */
export interface MemoryPromptRequest {
  query: string;
  session?: SessionParams | null;
  long_term_search?: SearchRequestParams | boolean | null;
}

/**
 * Parameters for long-term memory search
 */
export interface SearchRequestParams {
  text?: string;
  session_id?: SessionIdFilter | null;
  namespace?: NamespaceFilter | null;
  topics?: TopicsFilter | null;
  entities?: EntitiesFilter | null;
  created_at?: CreatedAtFilter | null;
  last_accessed?: LastAccessedFilter | null;
  user_id?: UserIdFilter | null;
  memory_type?: MemoryTypeFilter | null;
  event_date?: EventDateFilter | null;
  distance_threshold?: number | null;
  limit?: number;
  offset?: number;
  recency_boost?: boolean | null;
  recency_semantic_weight?: number | null;
  recency_recency_weight?: number | null;
  recency_freshness_weight?: number | null;
  recency_novelty_weight?: number | null;
  recency_half_life_last_access_days?: number | null;
  recency_half_life_created_days?: number | null;
  server_side_recency?: boolean | null;
}

// Filter type imports (forward declarations - actual types in filters.ts)
export interface SessionIdFilter {
  eq?: string | null;
  in_?: string[] | null;
  not_eq?: string | null;
  not_in?: string[] | null;
  ne?: string | null;
}

export interface NamespaceFilter {
  eq?: string | null;
  in_?: string[] | null;
  not_eq?: string | null;
  not_in?: string[] | null;
}

export interface UserIdFilter {
  eq?: string | null;
  in_?: string[] | null;
  not_eq?: string | null;
  not_in?: string[] | null;
}

export interface TopicsFilter {
  any?: string[] | null;
  all?: string[] | null;
  none?: string[] | null;
}

export interface EntitiesFilter {
  any?: string[] | null;
  all?: string[] | null;
  none?: string[] | null;
}

export interface CreatedAtFilter {
  gte?: string | null;
  lte?: string | null;
  eq?: string | null;
}

export interface LastAccessedFilter {
  gte?: string | null;
  lte?: string | null;
  eq?: string | null;
}

export interface EventDateFilter {
  gte?: string | null;
  lte?: string | null;
  eq?: string | null;
}

export interface MemoryTypeFilter {
  eq?: string | null;
  in_?: string[] | null;
  not_eq?: string | null;
  not_in?: string[] | null;
}

// ==================== Forget ====================

/**
 * Policy for forgetting memories
 */
export interface ForgetPolicy {
  /** Maximum age in days for memories to keep */
  max_age_days?: number | null;
  /** Maximum inactive days before forgetting */
  max_inactive_days?: number | null;
  /** Budget limit for forgetting operation */
  budget?: number | null;
  /** Allowlist of memory types to consider for forgetting */
  memory_type_allowlist?: string[] | null;
}

/**
 * Response from forget operation
 */
export interface ForgetResponse {
  /** Number of memories scanned */
  scanned: number;
  /** Number of memories deleted */
  deleted: number;
  /** IDs of deleted memories */
  deleted_ids: string[];
  /** Whether this was a dry run */
  dry_run: boolean;
}

// ==================== Summary Views ====================

/**
 * Source type for summary views
 */
export type SummaryViewSource = "long_term" | "working_memory";

/**
 * Summary view configuration
 */
export interface SummaryView {
  /** Unique identifier for the view */
  id: string;
  /** Optional human-readable name */
  name?: string | null;
  /** Memory source to summarize */
  source: SummaryViewSource;
  /** Fields to group by for partitioning */
  group_by: string[];
  /** Optional filters to apply */
  filters?: Record<string, unknown> | null;
  /** Time window in days for filtering */
  time_window_days?: number | null;
  /** Whether background workers refresh this view */
  continuous?: boolean;
  /** Custom summarization prompt */
  prompt?: string | null;
  /** Model override for summarization */
  model_name?: string | null;
}

/**
 * Request to create a summary view
 */
export interface CreateSummaryViewRequest {
  /** Optional human-readable name */
  name?: string | null;
  /** Memory source to summarize */
  source: SummaryViewSource;
  /** Fields to group by for partitioning */
  group_by: string[];
  /** Optional filters to apply */
  filters?: Record<string, unknown> | null;
  /** Time window in days for filtering */
  time_window_days?: number | null;
  /** Whether background workers refresh this view */
  continuous?: boolean;
  /** Custom summarization prompt */
  prompt?: string | null;
  /** Model override for summarization */
  model_name?: string | null;
}

/**
 * Result of summarizing one partition
 */
export interface SummaryViewPartitionResult {
  /** ID of the SummaryView that produced this result */
  view_id: string;
  /** Concrete values for the view's group_by fields */
  group: Record<string, string>;
  /** Summarized text for this partition */
  summary: string;
  /** Number of memories that contributed to this summary */
  memory_count: number;
  /** When this summary was computed */
  computed_at?: string;
}

/**
 * Request to run a summary view partition
 */
export interface RunSummaryViewPartitionRequest {
  /** Concrete values for the view's group_by fields */
  group: Record<string, string>;
}

/**
 * Request to run a full summary view
 */
export interface RunSummaryViewRequest {
  /** Force recomputation even if cached */
  force?: boolean;
}

// ==================== Tasks ====================

/**
 * Task type enum
 */
export type TaskType = "summary_view_full_run" | string;

/**
 * Task status enum
 */
export type TaskStatus = "pending" | "running" | "completed" | "failed";

/**
 * Background task representation
 */
export interface Task {
  /** Unique task identifier */
  id: string;
  /** Type of task */
  type: TaskType;
  /** Current task status */
  status: TaskStatus;
  /** Associated SummaryView ID, if applicable */
  view_id?: string | null;
  /** When the task record was created */
  created_at?: string;
  /** When execution started */
  started_at?: string | null;
  /** When execution finished */
  completed_at?: string | null;
  /** Error message if failed */
  error_message?: string | null;
}
