# CLAUDE.md - Redis Agent Memory Project Context

Read [`AGENTS.md`](./AGENTS.md) first. It maps the two projects in this
repository: [`V0/`](./V0/) (the Agent Memory Server) and
[`agent-memory-benchmark/`](./agent-memory-benchmark/) (the LongMemEval
harness).

For the server, see [`V0/AGENTS.md`](./V0/AGENTS.md) and
[`V0/CLAUDE.md`](./V0/CLAUDE.md). For the harness, see
[`agent-memory-benchmark/README.md`](./agent-memory-benchmark/README.md).

Run commands from inside the project directory, not from the repository root.

Keep the repository narrative intact when changing landing or project context:

- `V0/` is intentionally preserved as the pivotal open research foundation that
  informed Redis Agent Memory productization, not as the supported production
  distribution.
- Redis Agent Memory in Redis Iris is the official product path; keep its
  create-service and SDK/API quickstarts easy to find.
- The benchmark is an open audit and reproducibility surface, not a timeless
  vendor ranking. Benchmark claims must name the provider configuration, split,
  models, retrieval settings, artifacts, and run date.
