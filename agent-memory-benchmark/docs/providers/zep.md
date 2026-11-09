# Zep Cloud

Hosted temporal knowledge graph memory (`zep-cloud`).

Docs:

- [Zep quickstart](https://help.getzep.com/quick-start-guide)
- [Zep Cloud](https://www.getzep.com)
- SDK: [zep-cloud on PyPI](https://pypi.org/project/zep-cloud/)

CLI id: `zep`. Extra: `zep`.

## Install

```bash
uv sync --extra zep
```

## Auth

Create a project and API key in the Zep console ([app.getzep.com](https://app.getzep.com/)).

```bash
OPENAI_API_KEY=replace-me
ZEP_API_KEY=replace-me
```

## Provision

No local graph DB. Each run creates a Zep user and threads. Graph processing is asynchronous on Zep's side. The wrapper waits for usable search results before answering.

## Run

```bash
uv run memory-bench run \
  --provider zep \
 --split oracle \
  --limit 1 \
  --run-name smoke-zep \
  --provider-param model=gpt-4o \
  --provider-param search_limit=10

uv run memory-bench judge --experiment smoke-zep --judge-model gpt-4o
```

## Cleanup

Delete benchmark users and threads in the Zep console, or via the Zep user-delete API. Leave no long-lived `bench-*` users in a shared org.

## Notes

- Zep enforces message size and batch limits. The wrapper batches messages in
  groups of 30 but does not truncate them.
- Graph extraction tokens are not visible to the harness. OpenAI charges cover the final answer call only.
- Cloud spend and quotas apply. Start with `--limit`.
