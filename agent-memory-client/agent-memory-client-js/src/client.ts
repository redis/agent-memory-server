/**
 * Agent Memory API Client
 *
 * A TypeScript/JavaScript client for the Agent Memory Server REST API.
 */

import {
  MemoryClientError,
  MemoryNotFoundError,
  MemoryServerError,
  MemoryValidationError,
} from "./errors";
import {
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
import {
  AckResponse,
  CreateSummaryViewRequest,
  ForgetPolicy,
  ForgetResponse,
  HealthCheckResponse,
  MemoryPromptRequest,
  MemoryPromptResponse,
  MemoryRecord,
  MemoryRecordResults,
  MemoryStrategyConfig,
  ModelNameLiteral,
  RecencyConfig,
  SessionListResponse,
  SummaryView,
  SummaryViewPartitionResult,
  Task,
  WorkingMemory,
  WorkingMemoryResponse,
} from "./models";

const VERSION = "0.3.1";

/**
 * Configuration for the Memory API Client
 */
export interface MemoryClientConfig {
  /** Base URL of the memory server (e.g., 'http://localhost:8000') */
  baseUrl: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Optional default namespace to use for operations */
  defaultNamespace?: string;
  /** Optional default model name for auto-summarization */
  defaultModelName?: ModelNameLiteral;
  /** Optional default context window limit for auto-summarization */
  defaultContextWindowMax?: number;
  /** Optional API key for authentication */
  apiKey?: string;
  /** Optional bearer token for authentication */
  bearerToken?: string;
  /** Custom fetch function (for testing or custom implementations) */
  fetch?: typeof fetch;
}

/**
 * Options for search operations
 */
export interface SearchOptions {
  text: string;
  sessionId?: SessionId | { eq?: string; in_?: string[]; not_eq?: string; not_in?: string[] };
  namespace?: Namespace | { eq?: string; in_?: string[]; not_eq?: string; not_in?: string[] };
  topics?: Topics | { any?: string[]; all?: string[]; none?: string[] };
  entities?: Entities | { any?: string[]; all?: string[]; none?: string[] };
  createdAt?: CreatedAt | { gte?: Date | string; lte?: Date | string; eq?: Date | string };
  lastAccessed?: LastAccessed | { gte?: Date | string; lte?: Date | string; eq?: Date | string };
  userId?: UserId | { eq?: string; in_?: string[]; not_eq?: string; not_in?: string[] };
  memoryType?: MemoryType | { eq?: string; in_?: string[]; not_eq?: string; not_in?: string[] };
  eventDate?: EventDate | { gte?: Date | string; lte?: Date | string; eq?: Date | string };
  distanceThreshold?: number;
  limit?: number;
  offset?: number;
  recency?: RecencyConfig;
  optimizeQuery?: boolean;
}

/**
 * Client for the Agent Memory Server REST API.
 *
 * Provides methods to interact with all server endpoints:
 * - Health check
 * - Session management (list, get, put, delete)
 * - Long-term memory (create, search, edit, delete)
 */
export class MemoryAPIClient {
  private readonly config: Required<
    Pick<MemoryClientConfig, "baseUrl" | "timeout">
  > &
    MemoryClientConfig;
  private readonly fetchFn: typeof fetch;

  constructor(config: MemoryClientConfig) {
    this.config = {
      timeout: 30000,
      ...config,
      baseUrl: config.baseUrl.replace(/\/$/, ""), // Remove trailing slash
    };
    this.fetchFn = config.fetch ?? fetch;
  }

  /**
   * Get default headers for requests
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": `agent-memory-client-js/${VERSION}`,
      "X-Client-Version": VERSION,
    };

    if (this.config.apiKey) {
      headers["X-API-Key"] = this.config.apiKey;
    }
    if (this.config.bearerToken) {
      headers["Authorization"] = `Bearer ${this.config.bearerToken}`;
    }

    return headers;
  }

  /**
   * Make an HTTP request with error handling
   */
  private async request<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      params?: Record<string, string | number | boolean | string[] | undefined>;
    } = {}
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    // Add query parameters
    if (options.params) {
      for (const [key, value] of Object.entries(options.params)) {
        if (value !== undefined) {
          if (Array.isArray(value)) {
            // Handle array params by appending multiple values with the same key
            for (const item of value) {
              url.searchParams.append(key, String(item));
            }
          } else {
            url.searchParams.set(key, String(value));
          }
        }
      }
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await this.fetchFn(url.toString(), {
        method,
        headers: this.getHeaders(),
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return this.handleHttpError(response);
      }

      return (await response.json()) as T;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof MemoryClientError) {
        throw error;
      }
      if (error instanceof Error && error.name === "AbortError") {
        throw new MemoryClientError(`Request timeout after ${this.config.timeout}ms`);
      }
      throw new MemoryClientError(`Request failed: ${String(error)}`);
    }
  }

  /**
   * Handle HTTP error responses
   */
  private async handleHttpError(response: Response): Promise<never> {
    let message: string;
    // Read body as text first to avoid "Body has already been read" errors
    const text = await response.text();
    try {
      const body = JSON.parse(text) as Record<string, unknown>;
      message =
        (body.detail as string) || (body.message as string) || JSON.stringify(body);
    } catch {
      message = text || `HTTP ${response.status}`;
    }

    if (response.status === 404) {
      throw new MemoryNotFoundError(message);
    }
    throw new MemoryServerError(message, response.status);
  }

  // ==================== Health ====================

  /**
   * Check server health
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    return this.request<HealthCheckResponse>("GET", "/v1/health");
  }

  // ==================== Working Memory ====================

  /**
   * List all session IDs
   */
  async listSessions(options: {
    namespace?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<SessionListResponse> {
    return this.request<SessionListResponse>("GET", "/v1/working-memory/", {
      params: {
        namespace: options.namespace ?? this.config.defaultNamespace,
        limit: options.limit,
        offset: options.offset,
      },
    });
  }

  /**
   * Get working memory for a session
   */
  async getWorkingMemory(
    sessionId: string,
    options: {
      namespace?: string;
      modelName?: ModelNameLiteral;
      contextWindowMax?: number;
    } = {}
  ): Promise<WorkingMemoryResponse | null> {
    try {
      return await this.request<WorkingMemoryResponse>(
        "GET",
        `/v1/working-memory/${encodeURIComponent(sessionId)}`,
        {
          params: {
            namespace: options.namespace ?? this.config.defaultNamespace,
            model_name: options.modelName ?? this.config.defaultModelName,
            context_window_max: options.contextWindowMax ?? this.config.defaultContextWindowMax,
          },
        }
      );
    } catch (error) {
      if (error instanceof MemoryNotFoundError) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get or create working memory for a session
   */
  async getOrCreateWorkingMemory(
    sessionId: string,
    options: {
      namespace?: string;
      userId?: string;
      modelName?: ModelNameLiteral;
      contextWindowMax?: number;
      ttlSeconds?: number;
      longTermMemoryStrategy?: MemoryStrategyConfig;
    } = {}
  ): Promise<WorkingMemoryResponse> {
    const existing = await this.getWorkingMemory(sessionId, options);
    if (existing) {
      return existing;
    }

    // Create new working memory
    const workingMemory: WorkingMemory = {
      session_id: sessionId,
      namespace: options.namespace ?? this.config.defaultNamespace,
      user_id: options.userId,
      messages: [],
      memories: [],
      ttl_seconds: options.ttlSeconds,
      long_term_memory_strategy: options.longTermMemoryStrategy,
    };

    return this.putWorkingMemory(sessionId, workingMemory, options);
  }

  /**
   * Create or update working memory for a session
   */
  async putWorkingMemory(
    sessionId: string,
    workingMemory: Partial<WorkingMemory>,
    options: {
      namespace?: string;
      modelName?: ModelNameLiteral;
      contextWindowMax?: number;
      background?: boolean;
    } = {}
  ): Promise<WorkingMemoryResponse> {
    const body: WorkingMemory = {
      session_id: sessionId,
      ...workingMemory,
      namespace: workingMemory.namespace ?? options.namespace ?? this.config.defaultNamespace,
    };

    return this.request<WorkingMemoryResponse>(
      "PUT",
      `/v1/working-memory/${encodeURIComponent(sessionId)}`,
      {
        body,
        params: {
          model_name: options.modelName ?? this.config.defaultModelName,
          context_window_max: options.contextWindowMax ?? this.config.defaultContextWindowMax,
          background: options.background,
        },
      }
    );
  }

  /**
   * Delete working memory for a session
   */
  async deleteWorkingMemory(
    sessionId: string,
    options: { namespace?: string } = {}
  ): Promise<AckResponse> {
    return this.request<AckResponse>(
      "DELETE",
      `/v1/working-memory/${encodeURIComponent(sessionId)}`,
      {
        params: {
          namespace: options.namespace ?? this.config.defaultNamespace,
        },
      }
    );
  }

  // ==================== Long-term Memory ====================

  /**
   * Create long-term memory records
   */
  async createLongTermMemory(
    memories: MemoryRecord[],
    options: { namespace?: string } = {}
  ): Promise<AckResponse> {
    return this.request<AckResponse>("POST", "/v1/long-term-memory/", {
      body: { memories },
      params: {
        namespace: options.namespace ?? this.config.defaultNamespace,
      },
    });
  }

  /**
   * Search long-term memory
   */
  async searchLongTermMemory(options: SearchOptions): Promise<MemoryRecordResults> {
    const body: Record<string, unknown> = {
      text: options.text,
      limit: options.limit,
      offset: options.offset,
      distance_threshold: options.distanceThreshold,
    };

    // Add filters
    if (options.sessionId) {
      body.session_id =
        options.sessionId instanceof SessionId
          ? options.sessionId.toJSON()
          : options.sessionId;
    }
    if (options.namespace) {
      body.namespace =
        options.namespace instanceof Namespace
          ? options.namespace.toJSON()
          : options.namespace;
    }
    if (options.topics) {
      body.topics =
        options.topics instanceof Topics ? options.topics.toJSON() : options.topics;
    }
    if (options.entities) {
      body.entities =
        options.entities instanceof Entities
          ? options.entities.toJSON()
          : options.entities;
    }
    if (options.createdAt) {
      body.created_at =
        options.createdAt instanceof CreatedAt
          ? options.createdAt.toJSON()
          : options.createdAt;
    }
    if (options.lastAccessed) {
      body.last_accessed =
        options.lastAccessed instanceof LastAccessed
          ? options.lastAccessed.toJSON()
          : options.lastAccessed;
    }
    if (options.userId) {
      body.user_id =
        options.userId instanceof UserId ? options.userId.toJSON() : options.userId;
    }
    if (options.memoryType) {
      body.memory_type =
        options.memoryType instanceof MemoryType
          ? options.memoryType.toJSON()
          : options.memoryType;
    }
    if (options.eventDate) {
      body.event_date =
        options.eventDate instanceof EventDate
          ? options.eventDate.toJSON()
          : options.eventDate;
    }

    // Add recency config
    if (options.recency) {
      body.recency_boost = options.recency.recency_boost;
      body.recency_semantic_weight = options.recency.semantic_weight;
      body.recency_recency_weight = options.recency.recency_weight;
      body.recency_freshness_weight = options.recency.freshness_weight;
      body.recency_novelty_weight = options.recency.novelty_weight;
      body.recency_half_life_last_access_days = options.recency.half_life_last_access_days;
      body.recency_half_life_created_days = options.recency.half_life_created_days;
      body.server_side_recency = options.recency.server_side_recency;
    }

    return this.request<MemoryRecordResults>("POST", "/v1/long-term-memory/search", {
      body,
    });
  }

  /**
   * Get a long-term memory by ID
   */
  async getLongTermMemory(
    memoryId: string,
    options: { namespace?: string } = {}
  ): Promise<MemoryRecord | null> {
    try {
      return await this.request<MemoryRecord>(
        "GET",
        `/v1/long-term-memory/${encodeURIComponent(memoryId)}`,
        {
          params: {
            namespace: options.namespace ?? this.config.defaultNamespace,
          },
        }
      );
    } catch (error) {
      if (error instanceof MemoryNotFoundError) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Delete long-term memories by IDs
   */
  async deleteLongTermMemories(
    memoryIds: string[],
    options: { namespace?: string } = {}
  ): Promise<AckResponse> {
    return this.request<AckResponse>("DELETE", "/v1/long-term-memory", {
      params: {
        memory_ids: memoryIds,
        namespace: options.namespace ?? this.config.defaultNamespace,
      },
    });
  }

  // ==================== Memory Prompt ====================

  /**
   * Get memory-enhanced prompt
   */
  async memoryPrompt(request: MemoryPromptRequest): Promise<MemoryPromptResponse> {
    return this.request<MemoryPromptResponse>("POST", "/v1/memory/prompt", {
      body: request,
    });
  }

  // ==================== Edit Long-term Memory ====================

  /**
   * Edit a long-term memory by ID
   */
  async editLongTermMemory(
    memoryId: string,
    updates: Partial<MemoryRecord>
  ): Promise<MemoryRecord> {
    return this.request<MemoryRecord>(
      "PATCH",
      `/v1/long-term-memory/${encodeURIComponent(memoryId)}`,
      { body: updates }
    );
  }

  // ==================== Forget ====================

  /**
   * Run a forgetting pass with the provided policy
   */
  async forgetLongTermMemories(options: {
    policy: ForgetPolicy;
    namespace?: string;
    userId?: string;
    sessionId?: string;
    limit?: number;
    dryRun?: boolean;
    pinnedIds?: string[];
  }): Promise<ForgetResponse> {
    return this.request<ForgetResponse>("POST", "/v1/long-term-memory/forget", {
      params: {
        namespace: options.namespace,
        user_id: options.userId,
        session_id: options.sessionId,
        limit: options.limit,
        dry_run: options.dryRun,
      },
      body: {
        policy: options.policy,
        pinned_ids: options.pinnedIds,
      },
    });
  }

  // ==================== Summary Views ====================

  /**
   * List all summary views
   */
  async listSummaryViews(): Promise<SummaryView[]> {
    return this.request<SummaryView[]>("GET", "/v1/summary-views");
  }

  /**
   * Create a new summary view
   */
  async createSummaryView(request: CreateSummaryViewRequest): Promise<SummaryView> {
    return this.request<SummaryView>("POST", "/v1/summary-views", {
      body: request,
    });
  }

  /**
   * Get a summary view by ID
   */
  async getSummaryView(viewId: string): Promise<SummaryView | null> {
    try {
      return await this.request<SummaryView>(
        "GET",
        `/v1/summary-views/${encodeURIComponent(viewId)}`
      );
    } catch (error) {
      if (error instanceof MemoryNotFoundError) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Delete a summary view
   */
  async deleteSummaryView(viewId: string): Promise<AckResponse> {
    return this.request<AckResponse>(
      "DELETE",
      `/v1/summary-views/${encodeURIComponent(viewId)}`
    );
  }

  /**
   * Run a summary view partition
   */
  async runSummaryViewPartition(
    viewId: string,
    group: Record<string, string>
  ): Promise<SummaryViewPartitionResult> {
    return this.request<SummaryViewPartitionResult>(
      "POST",
      `/v1/summary-views/${encodeURIComponent(viewId)}/partitions/run`,
      { body: { group } }
    );
  }

  /**
   * List summary view partitions
   */
  async listSummaryViewPartitions(
    viewId: string,
    options: {
      namespace?: string;
      userId?: string;
      sessionId?: string;
      memoryType?: string;
    } = {}
  ): Promise<SummaryViewPartitionResult[]> {
    return this.request<SummaryViewPartitionResult[]>(
      "GET",
      `/v1/summary-views/${encodeURIComponent(viewId)}/partitions`,
      {
        params: {
          namespace: options.namespace,
          user_id: options.userId,
          session_id: options.sessionId,
          memory_type: options.memoryType,
        },
      }
    );
  }

  /**
   * Run a full summary view (async task)
   */
  async runSummaryView(
    viewId: string,
    options: { force?: boolean } = {}
  ): Promise<Task> {
    return this.request<Task>(
      "POST",
      `/v1/summary-views/${encodeURIComponent(viewId)}/run`,
      { body: options }
    );
  }

  // ==================== Tasks ====================

  /**
   * Get a task by ID
   */
  async getTask(taskId: string): Promise<Task | null> {
    try {
      return await this.request<Task>(
        "GET",
        `/v1/tasks/${encodeURIComponent(taskId)}`
      );
    } catch (error) {
      if (error instanceof MemoryNotFoundError) {
        return null;
      }
      throw error;
    }
  }

  // ==================== Utility Methods ====================

  /**
   * Close the client and release any resources.
   * This is a no-op for the fetch-based client but provided for API consistency.
   */
  close(): void {
    // No-op for fetch-based client
    // Provided for API compatibility with other clients
  }

  /**
   * Validate a memory record before sending to the server.
   * @throws MemoryValidationError if validation fails
   */
  validateMemoryRecord(memory: MemoryRecord): void {
    if (!memory.text || !memory.text.trim()) {
      throw new MemoryValidationError("Memory text cannot be empty");
    }

    const validMemoryTypes = ["episodic", "semantic", "message"];
    if (memory.memory_type && !validMemoryTypes.includes(memory.memory_type)) {
      throw new MemoryValidationError(`Invalid memory type: ${memory.memory_type}`);
    }

    if (memory.id && !this.isValidUlid(memory.id)) {
      throw new MemoryValidationError(`Invalid ID format: ${memory.id}`);
    }
  }

  /**
   * Validate search filter parameters before API call.
   * @throws MemoryValidationError if validation fails
   */
  validateSearchFilters(filters: {
    limit?: number;
    offset?: number;
    distanceThreshold?: number;
  }): void {
    if (filters.limit !== undefined && (typeof filters.limit !== "number" || filters.limit <= 0)) {
      throw new MemoryValidationError("Limit must be a positive integer");
    }

    if (filters.offset !== undefined && (typeof filters.offset !== "number" || filters.offset < 0)) {
      throw new MemoryValidationError("Offset must be a non-negative integer");
    }

    if (
      filters.distanceThreshold !== undefined &&
      (typeof filters.distanceThreshold !== "number" || filters.distanceThreshold < 0)
    ) {
      throw new MemoryValidationError("Distance threshold must be a non-negative number");
    }
  }

  /**
   * Check if a string is a valid ULID.
   */
  private isValidUlid(id: string): boolean {
    // ULID: 26 characters, Crockford Base32
    const ulidRegex = /^[0-7][0-9A-HJKMNP-TV-Z]{25}$/i;
    return ulidRegex.test(id);
  }

  /**
   * Bulk create long-term memories with rate limiting.
   * Useful for importing large datasets.
   */
  async bulkCreateLongTermMemories(
    memoryBatches: MemoryRecord[][],
    options: {
      batchSize?: number;
      delayBetweenBatches?: number;
      namespace?: string;
    } = {}
  ): Promise<AckResponse[]> {
    const { batchSize = 50, delayBetweenBatches = 100, namespace } = options;
    const results: AckResponse[] = [];

    for (const batch of memoryBatches) {
      // Split large batches into smaller chunks
      for (let i = 0; i < batch.length; i += batchSize) {
        const chunk = batch.slice(i, i + batchSize);
        const response = await this.createLongTermMemory(chunk, { namespace });
        results.push(response);

        // Rate limiting delay
        if (delayBetweenBatches > 0) {
          await this.sleep(delayBetweenBatches);
        }
      }
    }

    return results;
  }

  /**
   * Auto-paginating search that yields all matching long-term memory results.
   * Automatically handles pagination to retrieve all results.
   */
  async *searchAllLongTermMemories(
    options: Omit<SearchOptions, "limit" | "offset"> & { batchSize?: number }
  ): AsyncGenerator<MemoryRecord, void, undefined> {
    const { batchSize = 50, ...searchOptions } = options;
    let offset = 0;

    while (true) {
      const results = await this.searchLongTermMemory({
        ...searchOptions,
        limit: batchSize,
        offset,
      });

      if (!results.memories || results.memories.length === 0) {
        break;
      }

      for (const memory of results.memories) {
        yield memory;
      }

      // If we got fewer results than batchSize, we've reached the end
      if (results.memories.length < batchSize) {
        break;
      }

      offset += batchSize;
    }
  }

  /**
   * Sleep for a specified number of milliseconds.
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
