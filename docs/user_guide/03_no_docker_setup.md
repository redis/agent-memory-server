# Running Agent Memory Server on macOS (No Docker)

Natively on macOS, including Apple Silicon M-series. No Docker required.

## Prerequisites

- macOS (Intel or Apple Silicon)
- Homebrew (https://brew.sh)
- An LLM provider (one of):
 - **OpenAI API key** (easiest), or
 - **Ollama** (fully local, no API key needed)

---

## Part 1: Redis

Agent Memory Server uses Redis as its database. Install and start it:

```bash
brew install redis
brew services start redis
```

Verify:
```bash
redis-cli ping
# PONG
```

---

## Part 2: Agent Memory Server

### Install

```bash
brew install python@3.12
git clone https://github.com/redis/agent-memory-server.git
cd agent-memory-server
pip install uv
make setup
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with **one** of the following configurations:

**Option A: OpenAI (cloud)**
```
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=<your-openai-api-key>
LONG_TERM_MEMORY=true
GENERATION_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
DISABLE_AUTH=true
```

**Option B: Ollama (fully local, no API key)**

First install and start Ollama:
```bash
brew install ollama
ollama serve                    # Start Ollama server (keep running)
ollama pull llama3.2            # Pull a generation model
ollama pull nomic-embed-text    # Pull an embedding model
```

Then set `.env`:
```
REDIS_URL=redis://localhost:6379
OLLAMA_API_BASE=http://localhost:11434
GENERATION_MODEL=ollama/llama3.2
EMBEDDING_MODEL=ollama/nomic-embed-text
REDISVL_VECTOR_DIMENSIONS=768
LONG_TERM_MEMORY=true
DISABLE_AUTH=true
```

> **Note:** When using Ollama, you must set `REDISVL_VECTOR_DIMENSIONS` to match your embedding model (nomic-embed-text = 768, mxbai-embed-large = 1024).

### Build the search index

```bash
source .venv/bin/activate
uv run agent-memory rebuild-index
```

### Run

Pick what you need. Each runs in its own terminal.

**MCP Server** (for AI agent integration, e.g. Orchestrate, Cursor, Claude Desktop):
```bash
source .venv/bin/activate
uv run agent-memory mcp --mode sse --port 9000
```

**REST API** (optional, for HTTP clients):
```bash
source .venv/bin/activate
uv run agent-memory api
```

**Background task worker** (optional, for async extraction/compaction):
```bash
source .venv/bin/activate
uv run agent-memory task-worker
```

### Verify

```bash
# MCP server
curl -s http://localhost:9000/sse -H "Accept: text/event-stream" --max-time 3
# Expected: event: endpoint ...

# REST API
curl http://localhost:8000/v1/health
# Expected: {"status":"ok"}
```

---

## Stopping everything

```bash
brew services stop redis       # Stop Redis
# Ctrl+C in each terminal      # Stop Agent Memory Server
```

## Restarting after reboot

```bash
brew services start redis
cd agent-memory-server
source .venv/bin/activate
uv run agent-memory mcp --mode sse --port 9000
```

---

## Connecting to IBM watsonx Orchestrate (Remote MCP)

If you need to connect your local MCP server to a cloud service like IBM watsonx Orchestrate,
you need a public URL. Use **ngrok** or **cloudflared** to create a tunnel.

### Option A: ngrok

> Requires a free account. Sign up at https://ngrok.com and get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken

```bash
# Install
brew install ngrok

# Authenticate (one-time)
ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>

# Start tunnel (in a new terminal, while MCP server is running)
ngrok http 9000
```

Copy the `https://...ngrok-free.dev` URL from the output.

### Option B: cloudflared

> No account required for quick tunnels.

```bash
# Install
brew install cloudflared

# Start tunnel (in a new terminal, while MCP server is running)
cloudflared tunnel --url http://localhost:9000
```

Copy the `https://...trycloudflare.com` URL from the output.

### Verify the tunnel

```bash
curl -s https://<YOUR_TUNNEL_URL>/sse \
  -H "Accept: text/event-stream" \
  --max-time 3
# Expected: event: endpoint ...
```

### Add to Orchestrate

Go to **Agent settings > MCP Servers > Add remote MCP server** and fill in:

**Server name:** `Redis-Agent-Memory-MCP-Server`
**Description:** `Tools to manage an agent's memory with Redis. Two-tier memory: working/session memory management and long term memory management.`
**MCP server URL:** `https://<YOUR_TUNNEL_URL>/sse`
**Transport type:** Server-Sent Events (SSE)

> ⚠️ The URL **must** end with `/sse`. Without it you'll get a `502 / 404 Not Found` error.

> ⚠️ Free-tier tunnel URLs change on every restart. Update the URL in Orchestrate each time you restart the tunnel.

---

## Troubleshooting

**Port 9000/8000 in use:**
```bash
lsof -ti :9000 | xargs kill -9
```

**Redis not running:**
```bash
brew services restart redis
```

**Search index missing:**
```bash
uv run agent-memory rebuild-index
```

**Python version error:**
Requires Python >=3.12, <3.13. Check with `python3.12 --version`

**Orchestrate 502/404:**
Make sure the MCP server URL ends with `/sse`

**ngrok tunnel already exists:**
```bash
pkill -f "ngrok http"
ngrok http 9000
```
