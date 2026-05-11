# Repository Map

A module-by-module guide to the Agent Memory Server source tree, written for an
agent that needs to change something and wants to know where to look.

## Source layout

```
agent_memory_server/
  main.py                       FastAPI application entry point. Builds the app,
                                wires routers, mounts MCP, registers startup hooks.
  api.py                        REST endpoints (/v1/working-memory, /v1/long-term-memory,
                                /v1/memory/prompt). Thin layer over working_memory.py
                                and long_term_memory.py.
  mcp.py                        Model Context Protocol server. Exposes
                                create_long_term_memories, search_long_term_memory,
                                memory_prompt, set_working_memory tools.
  cli.py                        `agent-memory` Click CLI: api, mcp, task-worker,
                                rebuild-index, migrate-memories, schedule-task.
  config.py                     Pydantic Settings. Env vars, OAuth2 config,
                                model selection, feature flags.
  models.py                     Pydantic data models: WorkingMemory, MemoryRecord,
                                MemoryRecordResult, MemoryMessage, etc.
  auth.py                       OAuth2/JWT validation with JWKS. Multi-provider
                                (Auth0, Cognito, Okta, Azure AD).
  dependencies.py               FastAPI dependency-injection helpers.
  working_memory.py             Session-scoped memory: get/set/delete, automatic
                                summarization when context window exceeded.
  long_term_memory.py           Persistent storage with semantic search via RedisVL.
                                Promotion from working memory, deduplication.
  summarization.py              Conversation summarization using configured LLM.
  summary_views.py              Aggregated views over stored memories.
  extraction.py                 Topic and entity extraction (BERTopic + NER).
  filters.py                    Filter compilation for RedisVL FilterQuery.
  memory_strategies.py          Pluggable strategies for what gets promoted to
                                long-term memory and how.
  memory_vector_db.py           Abstract vector-DB interface.
  memory_vector_db_factory.py   Concrete backend selection (Redis, Pinecone,
                                Chroma, Postgres, ...).
  working_memory_index.py       In-memory index used during working-memory
                                operations.
  prompt_security.py            Prompt-injection guards and safe-prompt helpers.
  docket_tasks.py               Background task definitions (Docket queue).
  tasks.py                      Task wiring helpers.
  migrations.py                 Schema migrations for stored memory records.
  healthcheck.py                /health endpoint.
  logging.py                    Structlog setup.
  llm/                          LiteLLM-based client package.
    client.py                   LLMClient with chat() and embed() methods.
    types.py                    ChatCompletionResponse, EmbeddingResponse, LLMBackend.
    embeddings.py               Embedding-specific helpers.
    exceptions.py               LLMClientError, ModelValidationError, APIKeyMissingError.
  utils/                        Cross-cutting helpers.
    redis.py                    Redis connection setup and pooling.
    keys.py                     Redis key naming conventions.
    api_keys.py                 API-key utilities.
    redis_query.py              RedisVL query construction helpers.
    recency.py                  Recency-boost ranking math.
    tag_codec.py                Tag encoding/decoding for filterable fields.
    datetime.py                 Timezone-aware datetime helpers.
  _aws/                         AWS-specific helpers (Bedrock, IAM, Cognito).
```

## Test layout

```
tests/
  test_api.py                   REST endpoint coverage.
  test_mcp.py                   MCP tool coverage.
  test_working_memory.py        Session memory + summarization.
  test_long_term_memory.py      Vector search, promotion, dedup.
  test_extraction.py            Topic + entity extraction.
  test_summarization.py         LLM summarization paths.
  test_filters.py               Filter compilation.
  test_auth.py                  JWT/JWKS validation.
  conftest.py                   testcontainers Redis fixture, async client fixture.
```

## Where features live

| Feature | Module(s) |
|---|---|
| Store/retrieve a working-memory session | `api.py` → `working_memory.py` → `utils/redis.py` |
| Search long-term memory | `api.py` → `long_term_memory.py` → `filters.py` → `utils/redis_query.py` |
| Promote working → long-term memory | `working_memory.py` → `memory_strategies.py` → `long_term_memory.py` |
| Summarize a conversation | `working_memory.py` → `summarization.py` → `llm/client.py` |
| Extract topics & entities | `extraction.py` → `llm/client.py` |
| Hydrate a prompt with relevant memories | `api.py` (`/v1/memory/prompt`) → `long_term_memory.py` |
| MCP tool surface | `mcp.py` → same business-logic modules as REST |
| Auth on every request | `dependencies.py` → `auth.py` |
| Background indexing & compaction | `docket_tasks.py` → `long_term_memory.py` |
| Choose a vector backend | `memory_vector_db_factory.py` → `memory_vector_db.py` |

## What to read before changing X

- **Search behavior.** Read `long_term_memory.py` for the query path, then
  `filters.py` for filter semantics, then `utils/redis_query.py` for the actual
  RedisVL builders. Project rule: never call `redis.ft().search()` directly —
  always use RedisVL `VectorQuery` / `FilterQuery`.
- **A new endpoint.** Add the route in `api.py`, the model in `models.py`, the
  business logic in the appropriate `*_memory.py` module. Mirror it in `mcp.py`
  if it should also be an MCP tool.
- **Authentication.** `auth.py` is the JWKS validator; `dependencies.py` is how
  routes opt in. `DISABLE_AUTH=true` is a development-only escape hatch.
- **A new LLM/embedding provider.** Configure via env (LiteLLM resolves it);
  no code changes unless you need provider-specific features — then extend
  `llm/client.py`.
- **A new vector-DB backend.** Implement the `memory_vector_db.py` interface
  and register it in `memory_vector_db_factory.py`.

## What is intentionally not exported

- `agent_memory_server._aws.*` — AWS-specific helpers used by the Bedrock path;
  not part of the public contract.
- `agent_memory_server.utils.*` — internal helpers; tests may import them but
  they are not stable across versions.
- `agent_memory_server.working_memory_index` — implementation detail of the
  working-memory store.
