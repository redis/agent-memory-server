import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react-swc'
import path from 'path'
import fs from 'fs'
import { brotliDecompressSync, gunzipSync, inflateSync } from 'zlib'

type SearchSanitizeResult = {
  body: string
  dropped: number
  parse_error: boolean
}

type SearchMemoryLike = {
  id?: unknown
  id_?: unknown
  text?: unknown
  content?: unknown
  [key: string]: unknown
}

function hasMeaningfulString(value: unknown): value is string {
  if (typeof value !== 'string') return false
  // Treat Unicode control/separator/mark-only strings as empty placeholders.
  const normalized = value.replace(/[\p{C}\p{Z}\p{M}]/gu, '')
  return normalized.length > 0
}

function normalizeSearchMemory(memory: unknown): SearchMemoryLike | null {
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
    ...record,
    id: normalizedId,
    text: normalizedText,
  }
}

function sanitizeSearchResponseBody(rawText: string): SearchSanitizeResult {
  try {
    const parsed = JSON.parse(rawText) as {
      memories?: unknown[]
      total?: number
      [key: string]: unknown
    }
    const original = Array.isArray(parsed.memories) ? parsed.memories : []
    const memories = original
      .map((memory) => normalizeSearchMemory(memory))
      .filter((memory): memory is SearchMemoryLike => !!memory)
    const dropped = original.length - memories.length

    if (dropped <= 0) {
      if (memories.length === original.length) {
        return { body: rawText, dropped: 0, parse_error: false }
      }
      return {
        body: JSON.stringify({
          ...parsed,
          memories,
        }),
        dropped: 0,
        parse_error: false,
      }
    }

    console.warn(
      `[workbench] dropped ${dropped} malformed memories from proxied /api/long-term-memory/search response`
    )

    return {
      body: JSON.stringify({
        ...parsed,
        memories,
        total:
          typeof parsed.total === 'number'
            ? Math.max(0, parsed.total - dropped)
            : memories.length,
      }),
      dropped,
      parse_error: false,
    }
  } catch {
    // Search route must always be JSON. Mark parse failure so caller can fail loudly.
    return { body: rawText, dropped: 0, parse_error: true }
  }
}

function decodeProxyBody(buffer: Buffer, contentEncoding: string | undefined): string {
  const encoding = (contentEncoding || '').toLowerCase().trim()

  if (!encoding || encoding === 'identity') {
    return buffer.toString('utf8')
  }
  if (encoding.includes('gzip')) {
    return gunzipSync(buffer).toString('utf8')
  }
  if (encoding.includes('br')) {
    return brotliDecompressSync(buffer).toString('utf8')
  }
  if (encoding.includes('deflate')) {
    return inflateSync(buffer).toString('utf8')
  }

  // Unknown encoding; best effort as UTF-8.
  return buffer.toString('utf8')
}

function normalizePathCandidate(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  if (!trimmed) return null

  try {
    // Handles absolute-form request targets if present.
    return new URL(trimmed).pathname
  } catch {
    return trimmed
  }
}

function isSearchRoute(pathname: string | undefined): boolean {
  if (!pathname) return false
  return /^(?:\/api(?:\/v1)?|\/v1)?\/long-term-memory\/search\/?(?:\?.*)?$/.test(
    pathname
  )
}

function shouldSanitizeSearchResponse(req: any, proxyRes: any): boolean {
  const candidates = [
    normalizePathCandidate(req?.originalUrl),
    normalizePathCandidate(req?.url),
    normalizePathCandidate(req?.path),
    normalizePathCandidate(proxyRes?.req?.path),
    normalizePathCandidate(proxyRes?.req?.url),
  ].filter((path): path is string => !!path)

  return candidates.some(isSearchRoute)
}

function rewriteApiPath(path: string, memoryServerHasBasePath: boolean): string {
  const stripped = path.replace(/^\/api/, '')
  const normalized = stripped || '/'
  if (memoryServerHasBasePath) {
    // Prevent accidental /v1/v1 duplication when client requests /api/v1/*
    // and target already has a /v1 base path.
    return normalized.startsWith('/v1/') || normalized === '/v1'
      ? normalized.replace(/^\/v1/, '') || '/'
      : normalized
  }

  // Prevent accidental /v1/v1 duplication when client requests /api/v1/*.
  return normalized.startsWith('/v1') ? normalized : `/v1${normalized}`
}

function createSearchSanitizeProxy(
  memoryServerUrl: string,
  memoryServerHasBasePath: boolean
) {
  return {
    target: memoryServerUrl,
    changeOrigin: true,
    selfHandleResponse: true,
    rewrite: (path: string) => rewriteApiPath(path, memoryServerHasBasePath),
    configure: (proxy: any) => {
      proxy.on('proxyReq', (proxyReq: any, req: any) => {
        const reqPath = normalizePathCandidate(req?.originalUrl) ?? normalizePathCandidate(req?.url)
        if (isSearchRoute(reqPath)) {
          // Keep upstream response uncompressed so sanitization always operates on
          // a deterministic JSON payload.
          proxyReq.setHeader('accept-encoding', 'identity')
        }
      })

      proxy.on('proxyRes', (proxyRes: any, req: any, res: any) => {
        const chunks: Buffer[] = []
        proxyRes.on('data', (chunk: Buffer) => {
          chunks.push(Buffer.from(chunk))
        })

        proxyRes.on('end', () => {
          const rawBuffer = Buffer.concat(chunks)
          const contentEncoding = Array.isArray(proxyRes.headers['content-encoding'])
            ? proxyRes.headers['content-encoding'][0]
            : proxyRes.headers['content-encoding']

          let rawText = ''
          let decodeFailed = false
          try {
            rawText = decodeProxyBody(rawBuffer, contentEncoding)
          } catch (err) {
            decodeFailed = true
            rawText = rawBuffer.toString('utf8')
            console.warn(
              `[workbench] failed to decode proxied /api long-term-memory/search body (encoding=${contentEncoding || 'identity'}):`,
              err
            )
          }

          const isSearchRoute = shouldSanitizeSearchResponse(req, proxyRes)
          const upstreamStatusCode =
            typeof proxyRes.statusCode === 'number' ? proxyRes.statusCode : undefined
          const isSuccessful =
            typeof upstreamStatusCode === 'number' &&
            upstreamStatusCode >= 200 &&
            upstreamStatusCode < 300
          // Always attempt to parse/sanitize search-shaped JSON payloads.
          // If route/status detection is off for any reason, we still drop
          // malformed placeholder rows instead of leaking them to the UI.
          const shouldSanitizeBody = true
          const sanitizedCandidate = sanitizeSearchResponseBody(rawText)
          const useSanitizedBody = isSearchRoute || sanitizedCandidate.dropped > 0
          const sanitized = useSanitizedBody
            ? sanitizedCandidate
            : { body: rawText, dropped: 0, parse_error: false }
          const responseBody = sanitized.body

          res.statusCode =
            isSearchRoute && sanitized.parse_error
              ? 502
              : (upstreamStatusCode || 502)
          Object.entries(proxyRes.headers).forEach(([key, value]) => {
            if (!key || value === undefined) {
              return
            }
            const normalizedKey = key.toLowerCase()
            if (
              normalizedKey === 'content-length' ||
              normalizedKey === 'content-encoding' ||
              normalizedKey === 'transfer-encoding'
            ) {
              return
            }
            res.setHeader(key, value as string | string[])
          })
          if (isSearchRoute) {
            res.setHeader(
              'x-workbench-memory-target',
              memoryServerHasBasePath
                ? `${memoryServerUrl}/long-term-memory/search`
                : `${memoryServerUrl}/v1/long-term-memory/search`
            )
            res.setHeader('x-workbench-search-proxy', 'sanitize-proxy')
            res.setHeader(
              'x-workbench-search-route-detected',
              isSearchRoute ? 'true' : 'false'
            )
            res.setHeader(
              'x-workbench-search-decode',
              decodeFailed ? 'failed' : (contentEncoding || 'identity')
            )
            res.setHeader(
              'x-workbench-search-sanitized',
              shouldSanitizeBody ? 'true' : 'false'
            )
            res.setHeader('x-workbench-search-dropped', String(sanitized.dropped))
            res.setHeader(
              'x-workbench-search-parse-error',
              sanitized.parse_error ? 'true' : 'false'
            )
            res.setHeader(
              'x-workbench-search-upstream-status',
              upstreamStatusCode !== undefined ? String(upstreamStatusCode) : 'missing'
            )
            res.setHeader('content-type', 'application/json')
            if (sanitized.parse_error) {
              res.end(
                JSON.stringify({
                  detail:
                    'Workbench proxy received a non-JSON payload from memory search upstream',
                })
              )
              return
            }
          } else {
            res.setHeader('x-workbench-memory-target', memoryServerUrl)
          }
          res.end(responseBody)
        })
      })
    },
  }
}

function parseDotEnvFile(filePath: string): Record<string, string> {
  if (!fs.existsSync(filePath)) {
    return {}
  }

  const result: Record<string, string> = {}
  const text = fs.readFileSync(filePath, 'utf8')
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue

    const eqIndex = line.indexOf('=')
    if (eqIndex <= 0) continue

    const key = line.slice(0, eqIndex).trim()
    let value = line.slice(eqIndex + 1).trim()

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    }

    result[key] = value
  }
  return result
}

function loadEnvFileValues(mode: string): Record<string, string> {
  const files = [
    '.env',
    '.env.local',
    `.env.${mode}`,
    `.env.${mode}.local`,
  ]
  const merged: Record<string, string> = {}

  for (const fileName of files) {
    const fullPath = path.resolve(__dirname, fileName)
    const parsed = parseDotEnvFile(fullPath)
    Object.assign(merged, parsed)
  }
  return merged
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Always resolve .env files relative to workbench/, regardless of the shell cwd.
  // This prevents accidental proxying to the default server when Vite is launched
  // from the repo root.
  const env = loadEnv(mode, __dirname, '')
  const envFileValues = loadEnvFileValues(mode)
  const fileMemoryServerUrl = envFileValues.MEMORY_SERVER_URL
  const fileViteMemoryServerUrl = envFileValues.VITE_MEMORY_SERVER_URL

  // Guard against silent shell env overrides that can accidentally proxy /api
  // to a different backend than what workbench/.env specifies.
  if (
    process.env.MEMORY_SERVER_URL &&
    fileMemoryServerUrl &&
    process.env.MEMORY_SERVER_URL !== fileMemoryServerUrl
  ) {
    throw new Error(
      `[workbench] Shell MEMORY_SERVER_URL (${process.env.MEMORY_SERVER_URL}) overrides workbench/.env MEMORY_SERVER_URL (${fileMemoryServerUrl}). Update or unset the shell variable so proxy target is deterministic.`
    )
  }
  if (
    process.env.VITE_MEMORY_SERVER_URL &&
    fileViteMemoryServerUrl &&
    process.env.VITE_MEMORY_SERVER_URL !== fileViteMemoryServerUrl
  ) {
    throw new Error(
      `[workbench] Shell VITE_MEMORY_SERVER_URL (${process.env.VITE_MEMORY_SERVER_URL}) overrides workbench/.env VITE_MEMORY_SERVER_URL (${fileViteMemoryServerUrl}). Update or unset the shell variable so UI and proxy stay aligned.`
    )
  }

  // For the dev-server proxy, prefer MEMORY_SERVER_URL (server-side intent)
  // over VITE_MEMORY_SERVER_URL (browser runtime config) to avoid accidental
  // shell-level VITE_* overrides routing API calls to a different backend.
  const memoryServerUrl =
    fileMemoryServerUrl ||
    fileViteMemoryServerUrl ||
    env.MEMORY_SERVER_URL ||
    env.VITE_MEMORY_SERVER_URL
  if (!memoryServerUrl) {
    throw new Error(
      '[workbench] MEMORY_SERVER_URL or VITE_MEMORY_SERVER_URL must be set in workbench/.env. Refusing to start with an implicit default target.'
    )
  }
  const proxyConfiguredMemoryServerUrl =
    fileMemoryServerUrl || env.MEMORY_SERVER_URL
  const browserConfiguredMemoryServerUrl =
    fileViteMemoryServerUrl || env.VITE_MEMORY_SERVER_URL
  if (
    proxyConfiguredMemoryServerUrl &&
    browserConfiguredMemoryServerUrl &&
    proxyConfiguredMemoryServerUrl !== browserConfiguredMemoryServerUrl
  ) {
    throw new Error(
      `[workbench] MEMORY_SERVER_URL (${proxyConfiguredMemoryServerUrl}) and VITE_MEMORY_SERVER_URL (${browserConfiguredMemoryServerUrl}) differ. Set both to the same value so /api proxy and browser config stay aligned.`
    )
  }
  console.info(`[workbench] /api proxy target: ${memoryServerUrl}`)
  let memoryServerHasBasePath = false
  try {
    const parsed = new URL(memoryServerUrl)
    memoryServerHasBasePath =
      parsed.pathname !== '' && parsed.pathname !== '/'
  } catch {
    // If the URL cannot be parsed, fall back to default rewrite behaviour.
    memoryServerHasBasePath = false
  }

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 5173,
      host: true,
      allowedHosts: ["localhost", "andrews-macbook-pro.taila74d4.ts.net"],
      proxy: {
        '/api/long-term-memory/search': createSearchSanitizeProxy(
          memoryServerUrl,
          memoryServerHasBasePath
        ),
        '/api/v1/long-term-memory/search': createSearchSanitizeProxy(
          memoryServerUrl,
          memoryServerHasBasePath
        ),
        '^/api(?:/v1)?/long-term-memory/search/?(?:\\?.*)?$': {
          // Dedicated proxy so response sanitization is guaranteed for all
          // search route variants (with/without /v1, optional trailing slash,
          // and optional query string).
          ...createSearchSanitizeProxy(memoryServerUrl, memoryServerHasBasePath),
        },
        '/api': {
          // REST API proxy for the memory server.
          // Supports MEMORY_SERVER_URL with or without a base path (/v1).
          // Uses self-handled responses so search sanitization still runs even
          // if this catch-all route handles /api/long-term-memory/search.
          ...createSearchSanitizeProxy(memoryServerUrl, memoryServerHasBasePath),
        },
        '/mcp': {
          // MCP SSE: MCP Server on port 9000
          target: env.MCP_SERVER_URL || 'http://localhost:9000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/mcp/, ''),
          // Required for SSE streaming
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              // Ensure SSE responses are not buffered
              if (
                proxyRes.headers['content-type']?.includes('text/event-stream')
              ) {
                proxyRes.headers['cache-control'] = 'no-cache'
                proxyRes.headers['x-accel-buffering'] = 'no'
              }
            })
          },
        },
      },
    },
  }
})
