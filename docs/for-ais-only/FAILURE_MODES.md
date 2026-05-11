# Failure Modes

Things that look like bugs but are intentional. Read this before "fixing" any
of them.

## Search uses RedisVL, never raw `redis.ft().search()`

If you see RedisVL `VectorQuery` / `FilterQuery` builders instead of direct
`redis.ft().search(...)` calls, that is deliberate. The project standardises on
RedisVL because it owns query construction, type coercion, and embedding
field handling. Replacing it with raw FT calls will break filter semantics and
break the project rule.

## Working memory automatically summarises on overflow

Long sessions appear to "lose" earlier messages — they have actually been
summarised into the working-memory summary field. This is by design: the
context window is bounded by the configured model. Removing the summarisation
step will cause downstream prompt-token explosions.

## Promotion to long-term memory is asynchronous

A memory written to working memory is not immediately searchable in
long-term memory. Promotion runs in the background via Docket. Tests that
expect synchronous promotion are wrong; use the explicit promotion call or
wait for the worker.

## Deduplication is content-hash based

Two memories with byte-identical content are intentionally collapsed. Tests
that insert "duplicates" and expect two records back are wrong. Vary the
content if you need two records.

## `DISABLE_AUTH=true` is a development-only escape hatch

The REST API is locked down by OAuth2/JWT in every other configuration. Tests
and local development use `DISABLE_AUTH=true`; production deployments must
not. Do not "fix" the auth middleware to be optional in production builds.

## All core operations are async

The codebase is async-first. Sync wrappers exist only on the client. Adding a
synchronous code path inside `agent_memory_server/*` will break the FastAPI +
Docket event loop assumptions.

## Inline imports are deliberate where they exist

Most imports live at the top of the module. The handful of inline imports
exist to avoid circular dependencies, optional-dependency import errors, or
a measurable startup cost. Hoisting them to the top will break one of those.

## Topic and entity extraction are best-effort

`extraction.py` may produce empty topic / entity lists for short or
ambiguous content. That is expected. Consumers must not assume a non-empty
result. Do not raise on empty output.

## Tests use `testcontainers` Redis 8

Integration tests start a real Redis 8 container. They require Docker. They
are slow on cold cache. Do not "speed them up" by mocking Redis — mocked
tests will not catch RedisVL query regressions.

## Coverage gate failures are not flakes

If CI reports coverage below the project bar, do not retry. The failure is
real. Either add a test or delete the unreachable branch.
