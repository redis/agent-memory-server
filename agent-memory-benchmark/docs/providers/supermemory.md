# Supermemory

Hosted memory API (`supermemory` SDK). Sessions become documents. Search returns hybrid hits for the answer model.

Docs:

- [Quickstart](https://supermemory.ai/docs/quickstart)
- [How it works](https://supermemory.ai/docs/concepts/how-it-works)
- [Developer console](https://console.supermemory.ai)
- [github.com/supermemoryai/supermemory](https://github.com/supermemoryai/supermemory)

CLI id: `supermemory`. Extra: `supermemory`.

## Install

```bash
uv sync --extra supermemory
```

## Auth

Create a key in the [developer console](https://console.supermemory.ai) under API Keys. A consumer-app login is not an API credential.

```bash
OPENAI_API_KEY=replace-me
SUPERMEMORY_API_KEY=replace-me
```

## Provision

No local cluster. Isolation uses container tags derived from the example `user_id`.

## Run

```bash
uv run memory-bench run \
  --provider supermemory \
 --split oracle \
  --limit 1 \
  --run-name smoke-supermemory \
  --provider-param model=gpt-4o \
  --provider-param search_limit=20

uv run memory-bench judge --experiment smoke-supermemory --judge-model gpt-4o
```

## Cleanup

Delete documents and container tags for `bench-*` users in the console or via the documents API.

## Notes

- Ingest extraction is asynchronous. The wrapper polls until documents finish processing (timeout is configurable).
- Server-side ingest and search tokens are not in harness `TokenUsage`. Only the local answer LLM is counted.
- Quotas and rate limits apply. Use `--limit` on a new account.
