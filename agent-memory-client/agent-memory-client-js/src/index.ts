/**
 * Agent Memory Client for JavaScript/TypeScript
 *
 * A client SDK for interacting with the Agent Memory Server REST API.
 */

// Export client
export { MemoryAPIClient } from "./client";
export type { MemoryClientConfig, SearchOptions } from "./client";

// Export errors
export {
  MemoryClientError,
  MemoryNotFoundError,
  MemoryServerError,
  MemoryValidationError,
} from "./errors";

// Export filters
export {
  SessionId,
  Namespace,
  UserId,
  Topics,
  Entities,
  CreatedAt,
  LastAccessed,
  EventDate,
  MemoryType,
} from "./filters";

// Export models
export type {
  WorkingMemory,
  WorkingMemoryResponse,
  MemoryMessage,
  MemoryRecord,
  MemoryRecordResult,
  MemoryRecordResults,
  AckResponse,
  HealthCheckResponse,
  SessionListResponse,
  MemoryPromptRequest,
  MemoryPromptResponse,
  RecencyConfig,
  // Forget
  ForgetPolicy,
  ForgetResponse,
  // Summary Views
  SummaryView,
  SummaryViewSource,
  CreateSummaryViewRequest,
  SummaryViewPartitionResult,
  RunSummaryViewPartitionRequest,
  RunSummaryViewRequest,
  // Tasks
  Task,
  TaskType,
  TaskStatus,
} from "./models";
