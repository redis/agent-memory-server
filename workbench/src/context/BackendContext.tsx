import {
  createContext,
  useContext,
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react'
import {
  memoryApi,
  type SearchRequest,
  type SearchResponse,
  type HealthResponse,
  type CreateMemoryRequest,
  type WorkingMemoryResponse,
  type MemoryRecord,
  type Filter,
} from '@/lib/api'
import { McpSseClient, type McpStatus } from '@/lib/mcp-client'

// ---------------------------------------------------------------------------
// MemoryBackend interface — every operation the workbench needs
// ---------------------------------------------------------------------------

export type TransportMode = 'rest' | 'mcp'

export interface MemoryPromptParams {
  query: string
  session?: {
    session_id: string
    namespace?: string
    user_id?: string
  }
  long_term_search?:
    | {
        text?: string
        namespace?: Filter
        limit?: number
      }
    | boolean
}

export interface MemoryPromptResponse {
  messages?: Array<{ role: string; content: unknown }>
  long_term_memories?: MemoryRecord[]
}

export interface MemoryBackend {
  health(): Promise<HealthResponse>
  search(request: SearchRequest): Promise<SearchResponse>
  deleteMemory(id: string): Promise<{ acknowledged: boolean }>
  deleteMemories(ids: string[]): Promise<{ status: string }>
  compactMemories(params?: { namespace?: string; user_id?: string }): Promise<{ status: string }>
  createMemory(memory: CreateMemoryRequest): Promise<unknown>
  getMemoryPrompt(params: MemoryPromptParams): Promise<MemoryPromptResponse>
  updateSession(
    sessionId: string,
    data: {
      messages?: Array<{ role: string; content: string }>
      user_id?: string
      namespace?: string
    }
  ): Promise<WorkingMemoryResponse>
}

// ---------------------------------------------------------------------------
// REST backend — thin wrapper around the existing memoryApi
// ---------------------------------------------------------------------------

function createRestBackend(): MemoryBackend {
  return {
    health: () => memoryApi.health(),
    search: (req) => memoryApi.search(req),
    deleteMemory: (id) => memoryApi.deleteMemory(id),
    deleteMemories: (ids) => memoryApi.deleteMemories(ids),
    compactMemories: (p) => memoryApi.compactMemories(p),
    createMemory: (m) => memoryApi.createMemory(m),
    getMemoryPrompt: (p) => memoryApi.getMemoryPrompt(p),
    updateSession: (sid, data) => memoryApi.updateSession(sid, data),
  }
}

// ---------------------------------------------------------------------------
// MCP backend — calls MCP tools via the SSE client
// ---------------------------------------------------------------------------

function createMcpBackend(client: McpSseClient): MemoryBackend {
  return {
    // Health is not an MCP tool — always use REST
    health: () => memoryApi.health(),

    search: async (req) => {
      const args: Record<string, unknown> = { limit: req.limit ?? 10 }
      args.text = req.text ?? ''
      if (req.offset) args.offset = req.offset
      if (req.session_id) args.session_id = req.session_id
      if (req.namespace) args.namespace = req.namespace
      if (req.user_id) args.user_id = req.user_id
      if (req.topics) args.topics = req.topics
      if (req.entities) args.entities = req.entities
      if (req.memory_type) args.memory_type = req.memory_type
      if (req.distance_threshold) args.distance_threshold = req.distance_threshold
      if (req.created_at) args.created_at = req.created_at
      if (req.last_accessed) args.last_accessed = req.last_accessed
      const result = (await client.callTool(
        'search_long_term_memory',
        args
      )) as { memories: MemoryRecord[]; total: number }
      return { memories: result.memories ?? [], total: result.total ?? 0 }
    },

    deleteMemory: async (id) => {
      await client.callTool('delete_long_term_memories', {
        memory_ids: [id],
      })
      return { acknowledged: true }
    },

    deleteMemories: async (ids) => {
      await client.callTool('delete_long_term_memories', {
        memory_ids: ids,
      })
      return { status: `ok, deleted ${ids.length} memories` }
    },

    compactMemories: async (p) => {
      const args: Record<string, unknown> = {}
      if (p?.namespace) args.namespace = p.namespace
      if (p?.user_id) args.user_id = p.user_id
      const result = (await client.callTool(
        'compact_long_term_memories',
        args
      )) as { status: string }
      return { status: result.status ?? 'ok' }
    },

    createMemory: async (memory) => {
      return client.callTool('create_long_term_memories', {
        memories: [memory],
      })
    },

    getMemoryPrompt: async (params) => {
      const args: Record<string, unknown> = { query: params.query }

      // Map session to flat MCP params
      if (params.session) {
        if (params.session.session_id) {
          args.session_id = { eq: params.session.session_id }
        }
        if (params.session.user_id) {
          args.user_id = { eq: params.session.user_id }
        }
        if (params.session.namespace) {
          args.namespace = { eq: params.session.namespace }
        }
      }

      // Map long_term_search
      if (
        params.long_term_search &&
        typeof params.long_term_search === 'object'
      ) {
        if (params.long_term_search.limit) {
          args.limit = params.long_term_search.limit
        }
      }

      const result = (await client.callTool('memory_prompt', args)) as {
        messages?: Array<{ role: string; content: unknown }>
        long_term_memories?: MemoryRecord[]
      }
      return result
    },

    updateSession: async (sessionId, data) => {
      const args: Record<string, unknown> = { session_id: sessionId }
      if (data.messages) args.messages = data.messages
      if (data.user_id) args.user_id = data.user_id
      if (data.namespace) args.namespace = data.namespace
      return (await client.callTool(
        'set_working_memory',
        args
      )) as WorkingMemoryResponse
    },
  }
}

// ---------------------------------------------------------------------------
// React context
// ---------------------------------------------------------------------------

interface BackendContextValue {
  transport: TransportMode
  setTransport: (t: TransportMode) => Promise<void>
  backend: MemoryBackend
  mcpStatus: McpStatus
}

const BackendContext = createContext<BackendContextValue | null>(null)

export function BackendProvider({ children }: { children: ReactNode }) {
  const [transport, setTransportState] = useState<TransportMode>('rest')
  const [mcpStatus, setMcpStatus] = useState<McpStatus>('disconnected')
  const mcpClientRef = useRef<McpSseClient | null>(null)
  const restBackend = useMemo(() => createRestBackend(), [])

  const backend = useMemo<MemoryBackend>(() => {
    if (
      transport === 'mcp' &&
      mcpClientRef.current?.status === 'connected'
    ) {
      return createMcpBackend(mcpClientRef.current)
    }
    return restBackend
  }, [transport, mcpStatus, restBackend])

  const setTransport = useCallback(async (mode: TransportMode) => {
    if (mode === 'mcp') {
      // Connect
      const client = new McpSseClient('/mcp', setMcpStatus)
      mcpClientRef.current = client
      setTransportState('mcp')
      try {
        await client.connect()
      } catch (err) {
        console.error('MCP connection failed:', err)
        // Fall back to REST
        client.disconnect()
        mcpClientRef.current = null
        setTransportState('rest')
        setMcpStatus('disconnected')
      }
    } else {
      // Disconnect MCP if active
      mcpClientRef.current?.disconnect()
      mcpClientRef.current = null
      setTransportState('rest')
      setMcpStatus('disconnected')
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mcpClientRef.current?.disconnect()
    }
  }, [])

  return (
    <BackendContext.Provider
      value={{ transport, setTransport, backend, mcpStatus }}
    >
      {children}
    </BackendContext.Provider>
  )
}

export function useBackend(): BackendContextValue {
  const ctx = useContext(BackendContext)
  if (!ctx) throw new Error('useBackend must be used within BackendProvider')
  return ctx
}
