// Use relative path to leverage Vite's proxy (avoids CORS issues)
// Vite proxy rewrites /api/* to /v1/* on the target server
const API_BASE = '/api'

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
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `API error: ${response.status}`)
  }

  // Handle empty responses
  const text = await response.text()
  if (!text) return {} as T
  return JSON.parse(text)
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
  search: (request: SearchRequest) =>
    fetchApi<SearchResponse>('/long-term-memory/search', {
      method: 'POST',
      body: JSON.stringify({ ...request, text: request.text ?? '' }),
    }),

  // Long-term Memory - CRUD
  createMemory: (memory: CreateMemoryRequest) =>
    fetchApi<{ memories: MemoryRecord[] }>('/long-term-memory/', {
      method: 'POST',
      body: JSON.stringify({ memories: [memory] }),
    }),

  createMemories: (memories: CreateMemoryRequest[]) =>
    fetchApi<{ memories: MemoryRecord[] }>('/long-term-memory/', {
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
    fetchApi<{ acknowledged: boolean }>(
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
      `/working-memory/?${new URLSearchParams(
        Object.entries(params || {}).reduce(
          (acc, [k, v]) => (v !== undefined ? { ...acc, [k]: String(v) } : acc),
          {}
        )
      )}`
    ),

  getSession: (sessionId: string, params?: { namespace?: string; user_id?: string }) =>
    fetchApi<WorkingMemoryResponse>(
      `/working-memory/${sessionId}?${new URLSearchParams(
        Object.entries(params || {}).reduce(
          (acc, [k, v]) => (v !== undefined ? { ...acc, [k]: String(v) } : acc),
          {}
        )
      )}`
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
    fetchApi<{ acknowledged: boolean }>(
      `/working-memory/${sessionId}?${new URLSearchParams(
        Object.entries(params || {}).reduce(
          (acc, [k, v]) => (v !== undefined ? { ...acc, [k]: String(v) } : acc),
          {}
        )
      )}`,
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
