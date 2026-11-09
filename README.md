# Redis Agent Memory

[Redis Agent Memory](https://redis.io/agent-memory/) is Redis's managed memory layer for AI agents: a persistent store that lets an agent recall facts, events, and preferences across sessions, without stuffing every token of history back into the prompt. It is part of [Redis Iris](https://redis.io/iris/), Redis's real-time context engine for agents.

[Redis Agent Memory docs](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/) · [Benchmark harness](./agent-memory-benchmark/) · [Benchmark write-up](https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/)

## Getting started

- Product overview: [Redis Iris](https://redis.io/iris/)
- Agent Memory overview: [Redis Agent Memory docs](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/)
- Redis Cloud service guide: [Redis Agent Memory on Redis Cloud](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/)

On Redis Cloud:

1. [Create a database](https://redis.io/docs/latest/operate/rc/databases/create-database/).
2. [Create an Agent Memory service](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/create-service/).
3. Use the [Python SDK](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/python-sdk-quickstart/), [TypeScript SDK](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/typescript-sdk-quickstart/), or [REST API](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/rest-api-quickstart/) from your app.
4. Manage the service with the [Redis Cloud guide](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/).

Implementation details: [API and SDK examples](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/) and the [API reference](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/api-reference/).

## How it works

Redis Agent Memory uses a two-tier model. Session memory keeps active conversation state close at hand, with configurable TTL. Long-term memory stores extracted facts from past sessions as text plus vector embeddings for later retrieval.

As new events land in session memory, Redis Agent Memory extracts what matters and promotes it to long-term memory in the background. Memory can accumulate without slowing the live agent loop.

Keeping full conversation history in the prompt does not scale. Context windows are bounded, long prompts are slow and expensive, and accuracy degrades as prompts grow. Agent memory lets an agent stay fast and cheap on every turn while still recalling what happened weeks ago.

## Validated independently

We ran a best-effort LongMemEval comparison across a sweep of memory systems, on a shared answer model and judge, to put those systems on more equal footing (Wu et al., ICLR 2025). Some vendors publish higher scores than we report. We were not able to reproduce those numbers under this protocol.

A separate research configuration that combined two retrieval strategies (Instruct extraction and Remis) reached **86.5%** task-averaged accuracy in our evaluation, at about $0.07 per session. Remis is **not** in Redis Agent Memory in Iris yet, and that configuration is **not** what this public harness runs. Do not treat 86.5% as a Redis Cloud product score.

The public harness in [`agent-memory-benchmark/`](./agent-memory-benchmark/) evaluates the shipping Redis Agent Memory REST API and other adapters. It is an audit surface, not a timeless ranking. A result is evidence about a specific provider version, split, models, retrieval settings, artifacts, and run date.

```bash
git clone https://github.com/redis/agent-memory-server.git
cd agent-memory-server/agent-memory-benchmark
```

Setup, protocol, and provider recipes: [`agent-memory-benchmark/README.md`](./agent-memory-benchmark/README.md).

## V0 — the open-source research foundation

[`V0/`](./V0/) is the original Redis Agent Memory Server. It informed the Iris product. We keep it here as an open research artifact, not as the supported production distribution. Start at [`V0/README.md`](./V0/README.md).

## Citing this work

Redis Applied AI Research, "How Redis Agent Memory Stacks Up: Benchmarking Memory Systems for AI Agents on LongMemEval," 2026.

Also see [Building and evaluating long-term conversational memory](https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/).

## License

Apache License 2.0 (Redis, Inc.). See [`LICENSE`](./LICENSE).
