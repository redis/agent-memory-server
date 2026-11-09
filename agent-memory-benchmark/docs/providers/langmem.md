# LangMem

LangChain LangMem memory manager: LLM extract, update, and delete, then embedding retrieval over extracted texts.

Docs:

- [langchain-ai.github.io/langmem](https://langchain-ai.github.io/langmem/)
- [github.com/langchain-ai/langmem](https://github.com/langchain-ai/langmem)

CLI id: `langmem`. Extra: `langmem`.

## Install

```bash
uv sync --extra langmem
```

## Auth

```bash
OPENAI_API_KEY=replace-me
```

## Provision

No server. State is in-process for the run.

## Run

```bash
uv run memory-bench run \
  --provider langmem \
 --split oracle \
  --limit 1 \
  --run-name smoke-langmem \
  --provider-param model=gpt-4o

uv run memory-bench judge --experiment smoke-langmem --judge-model gpt-4o
```

## Cleanup

None. Restart the process to drop memories.

## Notes

- Extraction calls OpenAI (or the chat model you set). Retrieval uses OpenAI embeddings and cosine similarity.
- This is a functional LangMem core (`create_memory_manager`), not a full LangGraph deployment.
