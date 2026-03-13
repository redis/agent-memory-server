# Agent Memory Workbench

A browser-based demo UI for the [Redis Agent Memory Server](../README.md). It
provides two interactive views — a **Memory Explorer** for browsing and managing
long-term memories, and a **Chat** interface that demonstrates memory-augmented
conversations using OpenAI.

The workbench can talk to the memory server over **REST** or **MCP** (Model
Context Protocol), switchable at runtime from the header toggle.

## Features

### Memory Explorer (`/`)

- Full-text and semantic search across long-term memories
- Filter by memory type (semantic, episodic, message) and result limit
- Results with a detail panel showing all memory fields (topics, entities,
  namespace, user, session, timestamps, and more)
- **Delete** individual memories or **Clear All** in one click
- **Deduplicate** — triggers the server's compaction logic to merge duplicate and
  semantically similar memories

### Chat (`/chat`)

- Persona selector — switch between preconfigured user identities, each with a
  unique colour
- Memory-augmented conversations: each message round-trips through the memory
  server's `/memory/prompt` endpoint to retrieve relevant long-term memories and
  session history before calling OpenAI
- Conversations are persisted to working memory, which triggers automatic
  extraction of discrete memories (semantic/episodic) on the server side
- **Memory Operations** sidebar shows a live log of every retrieval, inference,
  and persistence step with expandable details
- **New Session** button to start fresh conversations

### Transport Toggle (REST / MCP)

Both transport modes are fully functional. The header pill lets you switch
between:

| Mode | How it works |
|------|--------------|
| **REST** | Standard HTTP calls to the memory server's `/v1/*` endpoints |
| **MCP** | JSON-RPC over SSE — connects to the MCP server and calls tools (`memory_prompt`, `search_long_term_memory`, `set_working_memory`, `compact_long_term_memories`, etc.) |

A connection status dot (green / yellow / red) appears on the active transport
button.

## Prerequisites

- **Node.js** >= 18
- A running **Redis Agent Memory Server** (REST API on port 8000, MCP on port
  9000)
- A running **Redis 8** instance
- An **OpenAI API key** (for Chat and for the server's embedding/extraction)

## Quick Start

### 1. Set up API keys

```bash
cd workbench
cp .env.example .env
```

Open `.env` and add your OpenAI API key in **both** places:

```env
# Browser-side — used by the Chat page to call OpenAI directly
VITE_OPENAI_API_KEY=sk-proj-your-key-here

# Server-side — used by the memory server for embeddings and memory extraction
OPENAI_API_KEY=sk-proj-your-key-here
```

> **Why two variables?** Vite only exposes `VITE_`-prefixed vars to the browser.
> The memory server reads `OPENAI_API_KEY` (without the prefix) for its own
> embedding generation and LLM-based memory extraction. The devcontainer's
> `start-services.sh` sources this `.env` file to set both.

### 2. Install and run

```bash
npm install
npm run dev
```

The Vite dev server starts on `http://localhost:5173` and proxies API calls to
the memory server automatically:

| Path | Proxied to |
|------|-----------|
| `/api/*` | `${MEMORY_SERVER_URL:-$VITE_MEMORY_SERVER_URL}/v1/*` (REST API) |
| `/mcp/*` | `http://localhost:9000/*` (MCP SSE) |

Set both `MEMORY_SERVER_URL` and `VITE_MEMORY_SERVER_URL` in
`workbench/.env` to the same value. The workbench now fails fast if neither is
set, and also fails if they differ. Example:

```env
MEMORY_SERVER_URL=http://localhost:8081
VITE_MEMORY_SERVER_URL=http://localhost:8081
```

### 3. Start the memory server

In a separate terminal (from the repo root):

```bash
# Start Redis
docker-compose up redis -d

# Start REST API server
uv run agent-memory api --port 8000 --task-backend=asyncio

# Start MCP server (in another terminal)
uv run agent-memory mcp --mode sse --port 9000 --task-backend=asyncio
```

Or use the devcontainer which starts everything automatically.

## Devcontainer

The repo ships a devcontainer configuration that runs Redis, the memory server,
the MCP server, and the workbench UI together. Port mappings:

| Service | Container Port | Host Port |
|---------|---------------|-----------|
| Workbench UI | 5173 | 25173 |
| REST API | 8000 | 28000 |
| MCP Server | 9000 | — |
| Redis | 6379 | 26379 |
| RedisInsight | 5540 | 28001 |

Environment variables are loaded from `workbench/.env` by
`.devcontainer/start-services.sh` at container startup.

## Project Structure

```
workbench/
├── src/
│   ├── pages/
│   │   ├── ExplorerPage.tsx    # Memory search, detail view, delete, deduplicate
│   │   └── ChatPage.tsx        # Memory-augmented chat with OpenAI
│   ├── components/
│   │   ├── layout/Header.tsx   # Nav, REST/MCP toggle, health indicator
│   │   └── RedisLogo.tsx       # Official Redis logo SVG
│   ├── config/personas.ts      # Chat persona definitions
│   ├── context/
│   │   └── BackendContext.tsx   # MemoryBackend interface + REST/MCP providers
│   ├── lib/
│   │   ├── api.ts              # REST API client (memoryApi)
│   │   ├── mcp-client.ts       # Browser MCP SSE client (McpSseClient)
│   │   ├── chat.ts             # Chat orchestration (memory → OpenAI → persist)
│   │   ├── openai.ts           # OpenAI client setup
│   │   └── utils.ts            # Helpers (cn, generateSessionId)
│   ├── App.tsx                 # Router: / → Explorer, /chat → Chat
│   └── index.css               # Tailwind v4 + Radar design tokens
├── .env.example                # Template — copy to .env and add keys
├── vite.config.ts              # Dev server + REST/MCP proxies
└── package.json
```

## Design

The UI follows the **Radar** design system (Redis product UI) with a dark theme:

- Midnight/dusk colour palette
- Geist + Space Grotesk typography
- Redis blue buttons, redis-red accents, memory-type colour coding
  (blue = semantic, green = episodic, purple = message)

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server with HMR |
| `npm run build` | Type-check and build for production |
| `npm run preview` | Preview production build locally |
| `npm run lint` | ESLint |
| `npm run format` | Prettier |
