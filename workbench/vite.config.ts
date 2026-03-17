import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react-swc'
import path from 'path'
import fs from 'fs'

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
  const files = ['.env', '.env.local', `.env.${mode}`, `.env.${mode}.local`]
  const merged: Record<string, string> = {}

  for (const fileName of files) {
    const fullPath = path.resolve(__dirname, fileName)
    const parsed = parseDotEnvFile(fullPath)
    Object.assign(merged, parsed)
  }
  return merged
}

function rewriteApiPath(pathname: string, memoryServerHasBasePath: boolean): string {
  const stripped = pathname.replace(/^\/api/, '')
  const normalized = stripped || '/'

  if (memoryServerHasBasePath) {
    return normalized.startsWith('/v1/') || normalized === '/v1'
      ? normalized.replace(/^\/v1/, '') || '/'
      : normalized
  }

  return normalized.startsWith('/v1') ? normalized : `/v1${normalized}`
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Always resolve .env files relative to workbench/, regardless of the shell cwd.
  const env = loadEnv(mode, __dirname, '')
  const envFileValues = loadEnvFileValues(mode)
  const fileMemoryServerUrl = envFileValues.MEMORY_SERVER_URL
  const fileViteMemoryServerUrl = envFileValues.VITE_MEMORY_SERVER_URL

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
    memoryServerHasBasePath = parsed.pathname !== '' && parsed.pathname !== '/'
  } catch {
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
      proxy: {
        '/api': {
          target: memoryServerUrl,
          changeOrigin: true,
          rewrite: (pathname) => rewriteApiPath(pathname, memoryServerHasBasePath),
        },
        '/mcp': {
          // MCP SSE: MCP Server on port 9000
          target: env.MCP_SERVER_URL || 'http://localhost:9000',
          changeOrigin: true,
          rewrite: (pathname) => pathname.replace(/^\/mcp/, ''),
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
