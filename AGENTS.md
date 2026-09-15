# AGENTS.md

The repository root is a thin landing layer — see [`README.md`](./README.md). Two
projects live below it. Each has its own tooling, tests, and docs. Run build,
lint, and test commands from inside the project directory, not from the root.

| Directory | Project | Instructions |
|-----------|---------|--------------|
| [`V0/`](./V0/) | Agent Memory Server, the pivotal research foundation preserved as a V0 artifact | [`V0/AGENTS.md`](./V0/AGENTS.md) (and [`V0/CLAUDE.md`](./V0/CLAUDE.md)) |
| [`agent-memory-benchmark/`](./agent-memory-benchmark/) | Open LongMemEval audit harness (`memory-bench`) | [`agent-memory-benchmark/README.md`](./agent-memory-benchmark/README.md) |

Examples: `cd V0 && make test`, or
`cd agent-memory-benchmark && uv run pytest`.

## Looking for the supported production path?

The code in this repository is open-source research and evaluation work. Redis's official
managed path for production use is **[Redis Agent Memory in Redis Iris](https://redis.io/agent-memory/)**
— agent memory as a service, with a REST API, client libraries, secure API key
management, and automatic TTL-based lifecycle management. See the
[README](./README.md#redis-agent-memory-in-redis-iris) for the full overview and links.
