/**
 * Agent Memory API Client
 *
 * A TypeScript/JavaScript client for the Agent Memory Server REST API.
 */

import {
  MemoryClientError,
  MemoryNotFoundError,
  MemoryServerError,
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
  HealthCheckResponse,
  MemoryPromptRequest,
  MemoryPromptResponse,
  MemoryRecord,
  MemoryRecordResults,
  MemoryStrategyConfig,
  ModelNameLiteral,
  RecencyConfig,
  SessionListResponse,
  WorkingMemory,
  WorkingMemoryResponse,
} from "./models";

const VERSION = "0.1.0";

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
      params?: Record<string, string | number | boolean | undefined>;
    } = {}
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    // Add query parameters
    if (options.params) {
      for (const [key, value] of Object.entries(options.params)) {
        if (value !== undefined) {
          url.searchParams.set(key, String(value));
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
    try {
      const body = (await response.json()) as Record<string, unknown>;
      message =
        (body.detail as string) || (body.message as string) || JSON.stringify(body);
    } catch {
      message = await response.text();
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
      body: { memory_ids: memoryIds },
      params: {
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
}
