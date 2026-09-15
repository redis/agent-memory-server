# Provider recipes

Each wrapper implements the same `MemoryStore` contract: ingest sessions, retrieve for a question, reset between examples. Install the matching `uv` extra, set keys from the recipe, then run a small smoke (oracle, one question) before a full Small run:

```bash
uv run memory-bench providers
uv run memory-bench run --provider <id> --split oracle --limit 1 --run-name smoke
uv run memory-bench judge --experiment smoke
```

Pass wrapper options with `--provider-param KEY=VALUE`.

These pages cite public vendor docs. Replace the placeholder project ids, regions, and keys with your own. The cloud recipes were written from those docs and not run against a live account.

| Provider | Extra | CLI id |
|----------|-------|--------|
| [Redis Agent Memory](redis-agent-memory.md) | — | `redis-agent-memory` |
| [Mem0](mem0.md) | `mem0` | `mem0` |
| [LangMem](langmem.md) | `langmem` | `langmem` |
| [Zep](zep.md) | `zep` | `zep` |
| [Graphiti](graphiti.md) | `graphiti` | `graphiti` |
| [Supermemory](supermemory.md) | `supermemory` | `supermemory` |
| [Google Vertex Memory Bank](google-vertex-memory-bank.md) | `google` | `vertex-memory-bank` |
| [AWS Bedrock AgentCore](aws-bedrock-agentcore.md) | `aws` | `bedrock-agentcore` |
| [Oracle Agent Memory](oracle-agent-memory.md) | `oracle` | `oracle-agent-memory` |
