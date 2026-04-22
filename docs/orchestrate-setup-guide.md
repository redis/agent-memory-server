# Agent Memory Server → IBM Orchestrate (via ngrok)

Step-by-step commands to run a local MCP memory server and connect it to IBM watsonx Orchestrate.

---

## 1. Install ngrok

> **Yes, an ngrok account is required** (free tier works). Sign up at https://ngrok.com and get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken

```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download

# Authenticate (one-time setup)
ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
```

## 2. Start Redis

```bash
docker-compose up redis -d
```

## 3. Set up the project (first time only)

```bash
pip install uv
make setup
```

## 4. Set environment variables

```bash
export DISABLE_AUTH=true
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export LONG_TERM_MEMORY=true
```

## 5. Start the MCP server

```bash
source .venv/bin/activate
uv run agent-memory mcp --mode sse --port 9000
```

Wait for:
```
Uvicorn running on http://0.0.0.0:9000
```

## 6. Start ngrok (new terminal)

```bash
ngrok http 9000
```

Look for the `Forwarding` line:
```
https://xxxx-xxxx-xxxx.ngrok-free.dev -> http://localhost:9000
```

Copy that `https://...ngrok-free.dev` URL.

## 7. Verify the tunnel works (new terminal)

```bash
curl -s https://<YOUR_NGROK_URL>/sse \
  -H "ngrok-skip-browser-warning: true" \
  -H "Accept: text/event-stream" \
  --max-time 3
```

Expected output:
```
event: endpoint
data: /messages/?session_id=...
```

## 8. Connect in IBM Orchestrate

Go to **Agent settings → MCP Servers → Add remote MCP server** and fill in:

| Field | Value |
|---|---|
| Server name | `Redis-Agent-Memory-MCP-Server` |
| Description | `Tools to manage an agent's memory with Redis. Two-tier memory: working/session memory management and long term memory management.` |
| MCP server URL | `https://<YOUR_NGROK_URL>/sse` |
| Transport type | **Server-Sent Events (SSE)** |

> ⚠️ The URL **must** end with `/sse`. Without it you'll get a `502 / 404 Not Found` error.

Click **Save**.

## 9. Verify tools appear

You should see 10 tools:

`create_long_term_memories`, `search_long_term_memory`, `get_long_term_memory`,
`edit_long_term_memory`, `delete_long_term_memories`, `compact_long_term_memories`,
`memory_prompt`, `set_working_memory`, `get_working_memory`, `get_current_datetime`

---

## Restarting (after reboot or closing terminals)

```bash
# Terminal 1: Redis
docker-compose up redis -d

# Terminal 2: MCP server
cd agent-memory-server
source .venv/bin/activate
export DISABLE_AUTH=true
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
uv run agent-memory mcp --mode sse --port 9000

# Terminal 3: ngrok
ngrok http 9000
```

> ⚠️ The ngrok URL changes on every restart (free tier). Update the MCP server URL in Orchestrate each time.

---

## Troubleshooting

**Port 9000 already in use:**
```bash
lsof -ti :9000 | xargs kill -9
```

**ngrok tunnel already exists:**
```bash
pkill -f "ngrok http"
ngrok http 9000
```

**Search index missing (after Redis restart):**
```bash
source .venv/bin/activate
uv run agent-memory rebuild-index
```
