# Agent Memory Server

> **For new agent memory projects on Redis, use [Redis Agent Memory](https://redis.io/agent-memory/) in [Redis Iris](https://redis.io/iris/).** It is the supported, maintained product. It runs as a managed service on Redis Cloud or self-managed on your own Kubernetes cluster.  
>   
> This repository holds the original open source Agent Memory Server (in [`V0/`](./V0/)), the research that shaped Redis Agent Memory. Agent Memory Server is deprecated and is not the supported production path.

## Start with Redis Agent Memory

Redis Agent Memory is a memory service for AI agents. It keeps the active conversation in session memory and extracts durable facts, preferences and events into long-term memory in the background, so a later session can recall them without replaying the whole history into the prompt. It works with any agent framework or LLM provider.

1. Create an Agent Memory store:  
     
   - Redis Cloud: [Create an Agent Memory service](https://redis.io/docs/latest/operate/iris/agent-memory/create-service/).  
   - Self-managed: [Deploy Agent Memory on Kubernetes](https://redis.io/docs/latest/operate/iris/agent-memory/self-managed/).

   

2. Install the Redis Iris Python package, which includes the `iris` CLI:

```sh
pip install 'redis-iris>=1.0.0'
```

3. Set `IRIS_AGENT_MEMORY_URL`, `IRIS_AGENT_MEMORY_STORE_ID` and `IRIS_AGENT_MEMORY_API_KEY` from your store, then record a conversation and recall what Agent Memory extracted:

```py
import time
from redis_iris.agent_memory import AgentMemory

memory = AgentMemory.from_env()

conversation = memory.conversation(owner="quickstart-user", session_id="travel-planning")
conversation.user("I am visiting Tokyo and Kyoto next month. I am vegetarian and prefer spicy food.")

question = "What dietary requirements and food preferences does the user have?"
for _ in range(30):  # extraction runs in the background
    known = memory.recall(question, owner="quickstart-user")
    if known:
        break
    time.sleep(10)
print(known.as_prompt())
```

Next steps:

- [Quickstart](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/quickstart/) in Python, the `iris` CLI or cURL  
- [Developer guide](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/developer-guide/)  
- [MCP](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/mcp/) for agents that search and curate memory through the store's MCP endpoint   
- [API reference](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/api-reference/)

## What is in this repository

| Path | What it is | Status |
| :---- | :---- | :---- |
| [`V0/`](./V0/) | Agent Memory Server, the original open source memory server: Python, FastAPI, RedisVL and LiteLLM, with a REST API, Python client, CLI and MCP server | Deprecated. Preserved for study. |
| [`agent-memory-benchmark/`](./agent-memory-benchmark/) | A LongMemEval harness that runs against the Redis Agent Memory REST API and other memory providers |  |

## How Agent Memory Server relates to Redis Agent Memory

Redis built Agent Memory Server first to test an idea: give agents a fast session memory and a durable long-term memory in Redis, and let a background process decide what moves from one to the other. That work validated the model that Redis Agent Memory is built on.

What carried over:

- Two tiers: session (working) memory for the active conversation and long-term memory for durable facts, preferences and summaries.  
- Background extraction from session data into long-term memory, so the agent loop stays fast.  
- Retrieval with Redis vector search plus metadata filters.  
- APIs to write, search, update and delete memory.

What is different:

- Redis Agent Memory is a new implementation, not a hosted copy of this server. Redis rebuilt it as a product service with its own API, data model, SDKs and runtime.  
- It adds what this server left to the operator: store provisioning, store-scoped API keys and access control, managed background workers, retention settings, and supported Cloud and Kubernetes deployment.  
- It adds features this server does not have, including [custom memory types](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/long-term-memory/#custom-memory-types), [sensitive-data exclusions](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/developer-guide/#exclude-sensitive-data-from-automatic-extraction), [namespaces](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/namespaces/), [Summary Views](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/summary-views/) and [MemoryETL](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/memory-etl/) for building memory from documents.

The two are not drop-in compatible. Moving an application from Agent Memory Server to Redis Agent Memory means changing endpoints, credentials, store IDs, and request and response models. [Migrate to Redis Agent Memory](https://redis.io/docs/latest/integrate/google-adk/agent-memory-server/#migrate-to-redis-agent-memory) in the Google ADK docs walks through the change for ADK apps. Switching does not copy existing memories, so plan for long-term memory to repopulate.

## Which one to use

Use **Redis Agent Memory** when you are building something new, when you want Redis to run the service, or when you need supported security, isolation and lifecycle controls. That covers almost every case.

Use **Agent Memory Server (V0)** when you want to read a working reference implementation of agent memory, study the extraction and retrieval approaches Redis tried first, or keep an existing deployment running while you migrate. You own everything if you run it: Redis, LLM access, authentication, workers, scaling, upgrades and backups.

## Benchmarks

[`agent-memory-benchmark/`](./agent-memory-benchmark/) runs LongMemEval (Wu et al., ICLR 2025\) against the Redis Agent Memory REST API and other memory providers on a shared answer model. Setup and provider recipes are in [`agent-memory-benchmark/README.md`](./agent-memory-benchmark/README.md). The full write-up is the [Agent Memory benchmark report](https://redis.io/resources/agent-memory-benchmark-report/).

## Agent Memory Server (V0) quick start

These instructions run the original server. They are kept accurate for existing users and for study. Full docs are in [`V0/docs/`](./V0/docs/index.md) and [`V0/README.md`](./V0/README.md).

Requirements: Docker, or Python with [uv](https://docs.astral.sh/uv/), and an OpenAI API key (or another provider configured through LiteLLM).

### Run with Docker

Images: [`redislabs/agent-memory-server`](https://hub.docker.com/r/redislabs/agent-memory-server) on Docker Hub and [`ghcr.io/redis/agent-memory-server`](https://github.com/redis/agent-memory-server/pkgs/container/agent-memory-server). The last release is 0.15.2.

```sh
git clone https://github.com/redis/agent-memory-server.git
cd agent-memory-server/V0

# Development: API and Redis, background tasks run in process
docker compose up api redis

# Production-like: API, background worker, Redis and MCP server
docker compose up api task-worker redis mcp
```

The image defaults to the Docket task backend, which needs a separate `agent-memory task-worker` process. For a single container, pass `--task-backend=asyncio`:

```sh
docker run -p 8000:8000 \
  -e REDIS_URL=redis://your-redis:6379 \
  -e OPENAI_API_KEY=your-key \
  redislabs/agent-memory-server:0.15.2 \
  agent-memory api --host 0.0.0.0 --port 8000 --task-backend=asyncio
```

### Run from source

```sh
cd V0
pip install uv
uv sync --all-extras
docker compose up redis
uv run agent-memory api --task-backend=asyncio
```

| Command | Use |
| :---- | :---- |
| `uv run agent-memory api --task-backend=asyncio` | Local development, one process |
| `uv run agent-memory api` | API server with Docket; also run `task-worker` |
| `uv run agent-memory task-worker --concurrency 10` | Background worker for Docket tasks |
| `uv run agent-memory mcp` | MCP server over stdio, for desktop MCP clients |
| `uv run agent-memory mcp --mode sse --port 9000` | MCP server over SSE |

### Python client

```sh
pip install agent-memory-client
```

```py
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

client = MemoryAPIClient(MemoryClientConfig(base_url="http://localhost:8000"))

await client.create_long_term_memory([
    {"text": "User prefers morning meetings", "user_id": "user123", "memory_type": "preference"}
])

results = await client.search_long_term_memory(
    text="What time does the user like meetings?",
    user_id="user123",
)
```

For agents, adding messages to working memory and letting the server extract memories in the background usually works better than writing memories by hand. See [Memory integration patterns](./V0/docs/memory-integration-patterns.md). The client also converts its tools for LangChain: [LangChain integration](./V0/docs/langchain-integration.md).

### MCP configuration

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["--from", "agent-memory-server", "agent-memory", "mcp"],
      "env": {
        "DISABLE_AUTH": "true",
        "REDIS_URL": "redis://localhost:6379",
        "OPENAI_API_KEY": "<your-openai-key>"
      }
    }
  }
}
```

Remove `DISABLE_AUTH` and set up [authentication](./V0/docs/authentication.md) before you expose the server to anyone else.

## Maintenance and contributing

Agent Memory Server is deprecated. 

For questions about Redis Agent Memory, use the [Redis Agent Memory docs](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/).

## Citing this work

Redis Applied AI Research, "How Redis Agent Memory Stacks Up: Benchmarking Memory Systems for AI Agents on LongMemEval," 2026\.

## License

Apache License 2.0, Copyright Redis, Inc. See [`LICENSE`](./LICENSE).
