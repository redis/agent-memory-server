# Mem0

Open-source Mem0 fact extraction and vector retrieval via the `mem0ai` SDK.

Docs:

- [docs.mem0.ai](https://docs.mem0.ai/)
- [Open source Python quickstart](https://docs.mem0.ai/open-source/python-quickstart)
- [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)

CLI id: `mem0`. Extra: `mem0`.

## Install

```bash
uv sync --extra mem0
```

Default config uses an in-process store and OpenAI embeddings. The Python
adapter accepts a Mem0 config dictionary, but the scalar CLI
`--provider-param` syntax does not encode nested dictionaries. Use the Python
API when configuring Qdrant, Chroma, or another backend.

## Auth

```bash
OPENAI_API_KEY=replace-me
```

Hosted Mem0 platform keys are optional. This wrapper targets the open-source client unless you change config.

## Provision

No extra cluster is required for the default in-memory backend. For a durable vector DB, create the collection in that product first.

## Run

```bash
uv run memory-bench run \
  --provider mem0 \
 --split oracle \
  --limit 1 \
  --run-name smoke-mem0 \
  --provider-param model=gpt-4o

uv run memory-bench judge --experiment smoke-mem0 --judge-model gpt-4o
```

## Cleanup

In-memory state dies with the process. For an external vector DB, delete the collection or namespace used by `user_id=benchmark`.

## Notes

- Extraction uses the installed Mem0 package and its default configuration
  unless you supply a public Mem0 config.
- Ingest LLM tokens are billed by the model Mem0 calls. They may be incomplete in harness totals.
