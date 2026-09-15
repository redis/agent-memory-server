# Redis Agent Memory

This adapter runs LongMemEval against the Redis Agent Memory REST API. It writes
session events, waits for automatic extraction, searches long-term memory, and
uses the harness answer model.

Docs:

- [Redis Agent Memory product overview](https://redis.io/agent-memory/)
- [Redis Agent Memory](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/)
- [Create an Agent Memory service](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/create-service/)
- [REST API quickstart](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/rest-api-quickstart)

CLI id: `redis-agent-memory`. No optional extra is required.

## Configure

[Create an Agent Memory service](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/create-service/).
Copy its endpoint, Store ID, and API key:

```bash
REDIS_AGENT_MEMORY_URL=https://replace-me
REDIS_AGENT_MEMORY_STORE_ID=replace-me
REDIS_AGENT_MEMORY_API_KEY=replace-me
OPENAI_API_KEY=replace-me
```

The API key must have access to the selected store. Set a short extraction
cadence, such as one minute, before a benchmark run.

## Run

Start with one oracle example:

```bash
uv run memory-bench run \
  --provider redis-agent-memory \
  --split oracle \
  --limit 1 \
  --run-name smoke-redis-agent-memory \
  --provider-param model=gpt-4o \
  --provider-param search_limit=10

uv run memory-bench judge \
  --experiment smoke-redis-agent-memory \
  --judge-model gpt-4o
```

The adapter waits up to 30 minutes for extraction and requires the memory count
to remain stable for two minutes. Override these values when needed:

```bash
--provider-param extraction_timeout=900 \
--provider-param extraction_poll_interval=15 \
--provider-param extraction_stable_seconds=60
```

## Cleanup

Each example uses a separate owner ID. The adapter deletes its sessions and
long-term memories after the answer. A terminated run can leave data behind;
remove benchmark owners from the service before another run.
