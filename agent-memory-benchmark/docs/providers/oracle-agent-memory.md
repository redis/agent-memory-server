# Oracle Agent Memory

Enterprise memory SDK (`oracleagentmemory`) on Oracle AI Database. Hybrid search plus LLM extraction.

Docs:

- [Oracle Help Center](https://docs.oracle.com/en/database/oracle/agent-memory/)
- [Get started](https://docs.oracle.com/en/database/oracle/agent-memory/26.6/guide/get-started.html)
- Product page: [oracle.com/database/ai-agent-memory](https://www.oracle.com/database/ai-agent-memory/)

CLI id: `oracle-agent-memory`. Extra: `oracle`.

## Install

```bash
uv sync --extra oracle
```

You need an Oracle AI Database (or Oracle Database 23ai Free) with the privileges the SDK documents. Local Free edition is enough for a smoke test if you accept disk and tablespace limits.

## Auth

```bash
OPENAI_API_KEY=replace-me
ORACLE_MEMORY_DB_USER=dmuser
ORACLE_MEMORY_DB_PASSWORD=REPLACE_ME
ORACLE_MEMORY_DB_CONNECT_STRING=localhost:1521/FREEPDB1
```

## Provision

1. Create a PDB and user as in Oracle Agent Memory get-started.
2. Grant the SDK the create-table and vector privileges listed there.
3. Confirm `sqlplus` or `oracledb` can connect with the DSN above.

The SDK creates and manages its own tables, so give it a dedicated PDB.

## Run

```bash
uv run memory-bench run \
  --provider oracle-agent-memory \
 --split oracle \
  --limit 1 \
  --run-name smoke-oracle \
  --provider-param model=gpt-4o \
  --provider-param embedder_model=text-embedding-3-small \
  --provider-param max_search_results=20

uv run memory-bench judge --experiment smoke-oracle --judge-model gpt-4o
```

## Cleanup

The wrapper deletes the ingest thread with `delete_thread` during reset, which
also removes that thread's messages and memories. Inspect the database after
interrupted runs and drop the dedicated PDB when finished.

Stop the database container if you used one.

## Notes

- The SDK is synchronous. The wrapper runs it in worker threads.
- The wrapper passes the answer `model` as the extraction LLM and
  `embedder_model` (default `text-embedding-3-small`) as the embedder.
- Oracle Free has a small tablespace cap. Full Small-split ingest may need a larger database.
- This recipe comes from public Oracle docs and SDK environment variables. It was not run against a live database.
