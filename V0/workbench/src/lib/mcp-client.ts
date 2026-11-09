/**
 * Lightweight MCP SSE client for the browser.
 *
 * Implements the MCP JSON-RPC protocol over SSE transport:
 *   1. GET  /sse       → opens an EventSource, receives an `endpoint` event
 *   2. POST <endpoint> → sends JSON-RPC requests
 *   3. SSE `message` events carry JSON-RPC responses back
 */

export type McpStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

interface JsonRpcRequest {
  jsonrpc: '2.0'
  id: number
  method: string
  params?: Record<string, unknown>
}

interface JsonRpcResponse {
  jsonrpc: '2.0'
  id?: number
  result?: unknown
  error?: { code: number; message: string; data?: unknown }
}

export class McpSseClient {
  private eventSource: EventSource | null = null
  private postEndpoint: string | null = null
  private nextId = 1
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >()
  private _status: McpStatus = 'disconnected'
  private statusCb?: (s: McpStatus) => void
  private baseUrl: string

  constructor(baseUrl: string, onStatusChange?: (s: McpStatus) => void) {
    this.baseUrl = baseUrl
    this.statusCb = onStatusChange
  }

  get status(): McpStatus {
    return this._status
  }

  private setStatus(s: McpStatus) {
    this._status = s
    this.statusCb?.(s)
  }

  /** Connect to the MCP SSE endpoint and complete the initialization handshake. */
  async connect(): Promise<void> {
    if (this._status === 'connected' || this._status === 'connecting') return
    this.setStatus('connecting')

    return new Promise<void>((resolve, reject) => {
      const sseUrl = `${this.baseUrl}/sse`
      this.eventSource = new EventSource(sseUrl)

      // The server sends an `endpoint` event with the POST URL
      this.eventSource.addEventListener('endpoint', (event: MessageEvent) => {
        const endpointPath: string = event.data
        // Prepend our proxy base so the browser routes through Vite proxy
        this.postEndpoint = endpointPath.startsWith('http')
          ? endpointPath
          : `${this.baseUrl}${endpointPath.startsWith('/') ? '' : '/'}${endpointPath}`

        // Run the MCP initialization handshake
        this.initialize()
          .then(() => {
            this.setStatus('connected')
            resolve()
          })
          .catch((err) => {
            this.setStatus('error')
            reject(err)
          })
      })

      // JSON-RPC responses arrive as `message` events
      this.eventSource.addEventListener('message', (event: MessageEvent) => {
        try {
          const response: JsonRpcResponse = JSON.parse(event.data)
          if (response.id !== undefined && response.id !== null) {
            const p = this.pending.get(response.id)
            if (p) {
              this.pending.delete(response.id)
              if (response.error) {
                p.reject(new Error(response.error.message))
              } else {
                p.resolve(response.result)
              }
            }
          }
        } catch {
          // ignore unparsable events
        }
      })

      this.eventSource.onerror = () => {
        if (this._status === 'connecting') {
          this.setStatus('error')
          reject(new Error('MCP SSE connection failed'))
        } else {
          this.setStatus('error')
        }
      }
    })
  }

  /** Perform the MCP initialize → initialized handshake. */
  private async initialize(): Promise<void> {
    await this.sendRequest('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'agent-memory-workbench', version: '1.0.0' },
    })
    // Send the "initialized" notification (no id, no response expected)
    await this.sendNotification('notifications/initialized')
  }

  /** Send a JSON-RPC request and wait for the response. */
  private sendRequest(
    method: string,
    params?: Record<string, unknown>
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (!this.postEndpoint) {
        reject(new Error('Not connected'))
        return
      }
      const id = this.nextId++
      this.pending.set(id, { resolve, reject })

      const body: JsonRpcRequest = {
        jsonrpc: '2.0',
        id,
        method,
        ...(params !== undefined ? { params } : {}),
      }

      fetch(this.postEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }).catch((err) => {
        this.pending.delete(id)
        reject(err)
      })
    })
  }

  /** Send a JSON-RPC notification (no response expected). */
  private async sendNotification(
    method: string,
    params?: Record<string, unknown>
  ): Promise<void> {
    if (!this.postEndpoint) throw new Error('Not connected')
    await fetch(this.postEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method,
        ...(params !== undefined ? { params } : {}),
      }),
    })
  }

  /**
   * Call an MCP tool and return the parsed result.
   *
   * MCP tool results have the shape:
   *   { content: [{ type: "text", text: "<json>" }] }
   *
   * This method extracts and JSON-parses the text content.
   */
  async callTool(
    name: string,
    args: Record<string, unknown>
  ): Promise<unknown> {
    const result = await this.sendRequest('tools/call', {
      name,
      arguments: args,
    })
    const toolResult = result as {
      content?: Array<{ type: string; text: string }>
    }
    if (toolResult?.content?.[0]?.type === 'text') {
      try {
        return JSON.parse(toolResult.content[0].text)
      } catch {
        return toolResult.content[0].text
      }
    }
    return result
  }

  /** Close the SSE connection and reject pending requests. */
  disconnect() {
    this.eventSource?.close()
    this.eventSource = null
    this.postEndpoint = null
    this.setStatus('disconnected')
    for (const [, p] of this.pending) {
      p.reject(new Error('Disconnected'))
    }
    this.pending.clear()
  }
}
