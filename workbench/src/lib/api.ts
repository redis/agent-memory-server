// Use relative path to leverage Vite's proxy (avoids CORS issues)
// Vite proxy rewrites /api/* to /v1/* on the target server
const API_BASE = '/api'

// Types
export interface AckResponse {
  status: string
}

type SearchMemoryLike = {
  id?: unknown
  id_?: unknown
  text?: unknown
  content?: unknown
} & Partial<MemoryRecordResult>

function hasMeaningfulString(value: unknown): value is string {
  if (typeof value !== 'string') return false
  const normalized = value.replace(/[\p{C}\p{Z}\p{M}]/gu, '')
  return normalized.length > 0
}

function normalizeSearchMemory(memory: unknown): MemoryRecordResult | null {
  if (!memory || typeof memory !== 'object') return null
  const record = memory as SearchMemoryLike
  const normalizedId = hasMeaningfulString(record.id)
    ? record.id
    : hasMeaningfulString(record.id_)
      ? record.id_
      : null
  const normalizedText = hasMeaningfulString(record.text)
    ? record.text
    : hasMeaningfulString(record.content)
      ? record.content
      : null

  if (!normalizedId || !normalizedText) {
    return null
  }

  return {
    ...(record as MemoryRecordResult),
    id: normalizedId,
    text: normalizedText,
  }
}

export interface MemoryRecord {
  id: string
  text: string
  memory_type: 'semantic' | 'episodic' | 'message'
  topics?: string[]
  entities?: string[]
  event_date?: string
  created_at: string
  last_accessed: string
  updated_at?: string
  access_count: number
  namespace?: string
  user_id?: string
  session_id?: string
  id_hash?: string
  memory_hash?: string
  persisted_at?: string
  pinned?: boolean
  discrete_memory_extracted?: string
  extraction_strategy?: string
  extraction_strategy_config?: Record<string, unknown>
  extracted_from?: string[]
}

export interface MemoryRecordResult extends MemoryRecord {
  dist?: number
}

export interface Filter {
  eq?: string
  ne?: string
  any?: string[]
  all?: string[]
  none?: string[]
}

export interface DateTimeFilter {
  gte?: string
  lte?: string
  gt?: string
  lt?: string
}

export interface SearchRequest {
  text?: string
  limit?: number
  offset?: number
  session_id?: Filter
  namespace?: Filter
  user_id?: Filter
  topics?: Filter
  entities?: Filter
  memory_type?: Filter
  created_at?: DateTimeFilter
  last_accessed?: DateTimeFilter
  event_date?: DateTimeFilter
  distance_threshold?: number
  recency_boost?: boolean
}

export interface SearchResponse {
  total: number
  memories: MemoryRecordResult[]
}

export interface WorkingMemory {
  session_id: string
  user_id?: string
  namespace?: string
  messages: Array<{ role: string; content: string }>
  memories: MemoryRecord[]
  context?: string
  created_at: string
  updated_at: string
  ttl_seconds: number
  data?: Record<string, unknown>
}

// API returns WorkingMemory fields directly at top level, plus extra fields
export interface WorkingMemoryResponse extends WorkingMemory {
  context_percentage_total_used: number | null
  context_percentage_until_summarization: number | null
  new_session: boolean | null
  unsaved: boolean | null
}

export interface HealthResponse {
  now: number // Server returns timestamp in milliseconds when healthy
}

export interface CreateMemoryRequest {
  id: string // Required unique identifier
  text: string
  memory_type?: 'semantic' | 'episodic'
  topics?: string[]
  entities?: string[]
  event_date?: string
  namespace?: string
  user_id?: string
  session_id?: string
}

export interface UpdateMemoryRequest {
  text?: string
  topics?: string[]
  entities?: string[]
  event_date?: string
}

// API Client
async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  const hasBody = options?.body !== undefined && options?.body !== null
  const hasExplicitContentType =
    !!options?.headers &&
    new Headers(options.headers).has('Content-Type')
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(hasBody && !hasExplicitContentType
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response
      .json()
      .catch(async () => ({ detail: (await response.text()) || 'Unknown error' }))
    throw new Error(error.detail || `API error: ${response.status}`)
  }

  // Handle empty responses
  const text = await response.text()
  if (!text) return {} as T
  return JSON.parse(text)
}

function buildQuery(params?: Record<string, string | number | undefined>): string {
  if (!params) return ''
  const qs = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) qs.set(key, String(value))
  }
  const query = qs.toString()
  return query ? `?${query}` : ''
}

export const memoryApi = {
  // Health
  health: () => fetchApi<HealthResponse>('/health'),

  // Long-term Memory - Compact (deduplicate)
  compactMemories: (params?: { namespace?: string; user_id?: string }) => {
    const qs = new URLSearchParams()
    if (params?.namespace) qs.set('namespace', params.namespace)
    if (params?.user_id) qs.set('user_id', params.user_id)
    const query = qs.toString()
    return fetchApi<{ status: string }>(
      `/long-term-memory/compact${query ? `?${query}` : ''}`,
      { method: 'POST' }
    )
  },

  // Long-term Memory - Search
  search: async (request: SearchRequest) => {
    // Some server versions take a different (and buggy) path when `text`
    // is omitted entirely. Always send text explicitly, including empty string.
    const normalizedRequest: SearchRequest = {
      ...request,
      text: request.text ?? '',
    }

    const result = await fetchApi<SearchResponse>('/long-term-memory/search', {
      method: 'POST',
      body: JSON.stringify(normalizedRequest),
    })

    const original = Array.isArray(result.memories) ? result.memories : []
    const memories = original
      .map((memory) => normalizeSearchMemory(memory))
      .filter((memory): memory is MemoryRecordResult => memory !== null)

    const dropped = original.length - memories.length
    if (dropped > 0) {
      // Defensive fallback for older/dirty backends that return placeholder
      // rows with empty id/text. The server should already filter these.
      console.warn(
        `[workbench] dropped ${dropped} malformed memories from search response (empty id/text)`
      )
    }

    return {
      ...result,
      memories,
      total:
        typeof result.total === 'number'
          ? Math.max(0, result.total - dropped)
          : memories.length,
    }
  },

  // Long-term Memory - CRUD
  createMemory: (memory: CreateMemoryRequest) =>
    fetchApi<AckResponse>('/long-term-memory/', {
      method: 'POST',
      body: JSON.stringify({ memories: [memory] }),
    }),

  createMemories: (memories: CreateMemoryRequest[]) =>
    fetchApi<AckResponse>('/long-term-memory/', {
      method: 'POST',
      body: JSON.stringify({ memories }),
    }),

  getMemory: (id: string) =>
    fetchApi<MemoryRecord>(`/long-term-memory/${id}`),

  updateMemory: (id: string, updates: UpdateMemoryRequest) =>
    fetchApi<MemoryRecord>(`/long-term-memory/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    }),

  deleteMemory: (id: string) =>
    fetchApi<AckResponse>(
      `/long-term-memory?memory_ids=${encodeURIComponent(id)}`,
      { method: 'DELETE' }
    ),

  deleteMemories: (ids: string[]) =>
    fetchApi<{ status: string }>(
      `/long-term-memory?${ids.map((id) => `memory_ids=${encodeURIComponent(id)}`).join('&')}`,
      { method: 'DELETE' }
    ),

  // Working Memory
  // Note: list returns session IDs as strings, not full WorkingMemory objects
  listSessions: (params?: { limit?: number; offset?: number; namespace?: string }) =>
    fetchApi<{ sessions: string[]; total: number }>(
      `/working-memory/${buildQuery(params)}`
    ),

  getSession: (sessionId: string, params?: { namespace?: string; user_id?: string }) =>
    fetchApi<WorkingMemoryResponse>(
      `/working-memory/${sessionId}${buildQuery(params)}`
    ),

  updateSession: (
    sessionId: string,
    data: Partial<{
      messages: Array<{ role: string; content: string }>
      memories: MemoryRecord[]
      context: string
      user_id: string
      namespace: string
      data: Record<string, unknown>
    }>
  ) =>
    fetchApi<WorkingMemoryResponse>(`/working-memory/${sessionId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  deleteSession: (sessionId: string, params?: { namespace?: string; user_id?: string }) =>
    fetchApi<AckResponse>(
      `/working-memory/${sessionId}${buildQuery(params)}`,
      { method: 'DELETE' }
    ),

  // Memory Prompt (context retrieval)
  // Combines working memory + long-term memories for AI context
  getMemoryPrompt: (params: {
    query: string
    session?: {
      session_id: string
      namespace?: string
      user_id?: string
    }
    long_term_search?: {
      text?: string
      namespace?: Filter
      limit?: number
    } | boolean
  }) =>
    fetchApi<{
      messages?: Array<{ role: string; content: unknown }>
      long_term_memories?: MemoryRecord[]
    }>('/memory/prompt', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
}
