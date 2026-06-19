<div align=center>

# Redis Agent Memory

A memory layer that gives agents intelligent short-term memory and persistent context across conversations.

</div>

This repository contains **V0**, the original open source research foundation and reference architecture for agent memory. While it remains a useful reference and resource, Redis’s official supported path for production use is now **Redis Agent Memory** in **Redis Iris**, our production-ready agent memory service.

## Redis Agent Memory in Redis Iris

[Redis Agent Memory in Redis Iris](https://redis.io/agent-memory/) is Redis’s official managed path for teams that want agent memory as a service, not another subsystem to build and operate themselves. [Redis Iris](https://redis.io/iris/) is the real-time context engine for agents, designed to deliver fresh, relevant context at runtime, and Redis Agent Memory is the part of Iris that makes context compound across turns, sessions, channels, and agents.

Where V0 gives you the original open-source research foundation, Redis Agent Memory in Iris gives you the Redis-managed experience: a persistent, structured memory layer for AI agents exposed through a REST API and client libraries, with dedicated endpoints, secure API key management, configurable memory schemas, and automatic TTL-based lifecycle management. The point is not just storage. It is to remove the custom memory infrastructure teams otherwise end up building around session handling, extraction, retrieval, and lifecycle management.

Redis Agent Memory uses a two-tier model. Session memory keeps the active conversation state, session history, and session-specific metadata close at hand, with configurable TTL control for retention. Long-term memory stores extracted facts and learned patterns from past interactions as text plus vector embeddings for semantic retrieval. As new events are written to working memory, Redis Agent Memory automatically extracts important information and promotes it to long-term memory in the background, so memory accumulates without slowing down the live agent loop.

That matters because Redis Iris is not just a memory feature in isolation. It is a broader context engine built to address the production problems agents actually hit: fragmented data, stale operational state, slow retrieval, and interactions that do not improve over time. Within that story, Redis Agent Memory is the compounding memory layer; [Redis Context Retriever](https://redis.io/context-retriever/) makes business data navigable; [Redis Data Integration](https://redis.io/data-integration/) keeps operational state fresh; and [Redis LangCache](https://redis.io/langcache/) helps repeated work stay inside the latency budget.

If you are evaluating the supported Redis path, these are the best places to start:

- Product overview: [Redis Iris](https://redis.io/iris/)
- Agent Memory overview: [Redis Agent Memory docs](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/)
- Redis Cloud service guide: [Redis Agent Memory on Redis Cloud](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/)

A practical getting-started flow on Redis Cloud looks like this:

- [Create a database](https://redis.io/docs/latest/operate/rc/databases/create-database/)
- [Create an Agent Memory service](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/create-service/)
- [Use the Agent Memory API](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/use-agent-memory/) from your application
- [View and manage your service](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/view-service/)

For implementation details and usage examples, see:

- [API and SDK examples](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/api-examples/)
- [API reference](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/api-reference/)


## V0 — the open-source research foundation

[**`V0/`**](./V0/) contains the original Redis Agent Memory Server: an open-source reference implementation for agent memory with REST and MCP interfaces, working and long-term memory, configurable extraction strategies, and Redis-backed semantic search.
It serves as the research foundation and architectural starting point for Redis Agent Memory, but it is not the current supported production path.

- **Start here:** [`V0/README.md`](./V0/README.md)
- **Documentation:** https://redis.github.io/agent-memory-server/
- Build, test, and run everything from inside `V0/` (e.g. `cd V0 && make test`).

## License

This project is licensed under the **Apache License 2.0** (Redis, Inc.). A copy
is provided at both the repository root ([`LICENSE`](./LICENSE)) and inside
[`V0/LICENSE`](./V0/LICENSE).
