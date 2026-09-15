# Agent Memory Benchmark

This directory is an open **LongMemEval** v1 harness for agent-memory systems.
It ingests sessions, retrieves context for a test question, generates an answer
through a shared answering step, and records a task-specific LLM judgment.

## Why this exists

Agent-memory benchmark numbers are easy to publish and hard to audit. Results can
move with the dataset revision, model snapshot, prompt, retrieval limit, provider
configuration, hosted-service version, or wait policy. A static leaderboard hides
most of those choices.

Redis is publishing this harness so the community can inspect the evaluation
path, reproduce or dispute a result, contribute an adapter, and compare memory
designs under stated conditions. It is not included to assert that Redis is
automatically best. It joins the open work around the upstream
[LongMemEval implementation](https://github.com/xiaowu0162/LongMemEval), including
provider-published harnesses such as [Zep's](https://github.com/getzep/zep/tree/main/benchmarks/longmemeval).

The harness standardizes the outer loop, not the systems under test:

| Held in common by the harness | Remains provider-specific |
|---|---|
| Dataset and split | Extraction model and memory schema |
| Per-example isolation and reset | Indexing, consolidation, and retrieval |
| Answer prompt and answer model | Hosted-service version and processing time |
| Task-specific judge and result schema | Provider cost, quotas, and infrastructure |

Adapters use public SDKs or APIs and provider-native extraction where available.
That makes a run an end-to-end observation of the configured system, not a pure
comparison of retrieval algorithms.

## The protocol

This harness targets the LongMemEval setup used in
[Building and evaluating long-term conversational memory](https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/):

| Item | Value |
|------|--------|
| Dataset | LongMemEval **Small** |
| Size | **500** questions |
| Coverage | **six** task types |
| Models | **gpt-4o** defaults for answers and judging; configurable |
| Metric | Task-averaged accuracy |

The built-in judge uses the official LongMemEval task-specific binary prompts
from `evaluate_qa.py`. Each run also writes `hypotheses.jsonl` (`question_id` +
`hypothesis`) so you can re-grade with the
[upstream evaluation script](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py).
For a result you intend to publish, pin exact model snapshots and keep the
complete output directory.

## Quick start

You need Python 3.10–3.12, [uv](https://docs.astral.sh/uv/), and an OpenAI key.

```bash
cp .env.example .env    # set OPENAI_API_KEY
uv sync --extra langmem --group dev
uv run memory-bench providers
```

The cheapest smoke run uses LongMemEval **oracle** (short haystacks) and one question. LangMem needs no extra server.

```bash
uv run memory-bench run \
  --provider langmem \
  --split oracle \
  --limit 1 \
  --run-name smoke-langmem \
  --provider-param model=gpt-4o-mini
```

Judge that run (use `gpt-4o` to mirror the Redis research configuration):

```bash
uv run memory-bench judge \
  --experiment smoke-langmem \
  --judge-model gpt-4o-mini
```

`memory-bench` reads `.env` from the working directory or a parent, so keep yours in `agent-memory-benchmark/`. The first `run` downloads the split into `~/.cache/memory-bench/`. Later runs reuse the cache.

Put `-v` before the subcommand if you want debug logs (`memory-bench -v run ...`). Debug logs can include conversation text.

For the Redis research configuration, use `--split small`, no `--limit` (500
questions), and `model=gpt-4o` for both `run` and `judge`. That is expensive.
For a new published comparison, use exact model snapshots when available and
record any deviation from that configuration.

## How a run works

For each example:

1. Ingest prior sessions into the selected provider.
2. Wait for provider-specific extraction readiness or count stability (timeouts
   vary; see the provider recipe).
3. Retrieve memory for the question and generate an answer.
4. Run `memory-bench judge` (yes/no LLM judge, task-averaged accuracy).

Re-run the same command after a crash. Completed question ids are skipped.

The default answer and judge model is `gpt-4o`. Override it with
`--provider-param model=...` and `--judge-model`.

## Supported providers

Install extras from `pyproject.toml`. Pass `--provider` to `memory-bench run`.

| CLI id | Extra | Recipe |
|--------|-------|--------|
| `redis-agent-memory` | — | [Redis Agent Memory](docs/providers/redis-agent-memory.md) |
| `mem0` | `mem0` | [Mem0](docs/providers/mem0.md) |
| `langmem` | `langmem` | [LangMem](docs/providers/langmem.md) |
| `zep` | `zep` | [Zep](docs/providers/zep.md) |
| `graphiti` | `graphiti` | [Graphiti](docs/providers/graphiti.md) |
| `supermemory` | `supermemory` | [Supermemory](docs/providers/supermemory.md) |
| `vertex-memory-bank` | `google` | [Google Vertex Memory Bank](docs/providers/google-vertex-memory-bank.md) |
| `bedrock-agentcore` | `aws` | [AWS Bedrock AgentCore](docs/providers/aws-bedrock-agentcore.md) |
| `oracle-agent-memory` | `oracle` | [Oracle Agent Memory](docs/providers/oracle-agent-memory.md) |

Index: [docs/providers/README.md](docs/providers/README.md).

## Dataset

Public files: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). Upstream: [LongMemEval](https://github.com/xiaowu0162/LongMemEval).

Splits: `oracle`, `small`, `medium`. Default CLI split is `small`.

Repeat `--provider-param KEY=VALUE` for wrapper options. See each provider page.

## Outputs

Each run writes `experiment_results/<run-name>/`:

| File | Contents |
|------|----------|
| `answers.jsonl` | Per-question answer, retrieved context, prompt, latency, and token fields |
| `hypotheses.jsonl` | Same answers in LongMemEval's `question_id` / `hypothesis` schema |
| `judgments.jsonl` | Per-question score and raw judge response |
| `metadata.json` | Provider and benchmark configuration, Git state, models, and completion state |
| `errors.jsonl` | Failed examples |
| `metrics.json` | Aggregates after a judge run |

Provider parameters whose names look like keys, tokens, secrets, or passwords are redacted in `metadata.json`.

The output directory is the unit of review. A result without its `metadata.json`,
per-question answers and judgments, any generated error log, and run date should
not be added to a comparison. Treat a run with failed or missing questions as
incomplete.

To verify judgments with LongMemEval's upstream evaluator, pass the run's
`hypotheses.jsonl` and the same cleaned dataset file to
[`evaluate_qa.py`](https://github.com/xiaowu0162/LongMemEval#testing-your-system).
Judge calls can be nondeterministic even at temperature zero, so retain both
sets of judgments if they differ.

## Interpreting and sharing results

Before comparing or publishing a run, report at least:

- provider and service/package version, configuration, and run date;
- dataset revision and split, question count, and any failed examples;
- answer and judge model snapshots, prompts, retrieval limit, and wait policy;
- whether extraction and managed-service costs are included; and
- the complete redacted experiment artifacts needed to audit the score.

Compare only like-for-like runs. A provider's own published number is context,
not a reproduced result, until the harness configuration and artifacts match.
Pull requests that correct protocol behavior, add provider adapters, or publish
reproducible artifacts with these conditions are welcome.

## Adapter extension

1. Implement `MemoryStore` in `src/agent_memory_benchmark/memory/<id>_store.py`.
2. Register the CLI id in `memory/__init__.py`.
3. Add an optional extra in `pyproject.toml` if you need a vendor SDK.
4. Add `docs/providers/<id>.md`.

## Costs

Prices below use OpenAI list rates checked September 2, 2026:
[gpt-4o](https://developers.openai.com/api/docs/models/gpt-4o) at $2.50 / $10
per million input / output tokens and
[gpt-5.6-luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) at
$0.20 / $1.20. They cover visible LLM calls from this harness, not
hosted-memory invoices. Recalculate against current pricing before publishing a
new cost claim.

The Small split is 500 questions and about 61 million tokens of haystack chat. Oracle is the same 500 questions with about 3 million tokens of haystack.

| Split | What you pay for | gpt-4o | gpt-5.6-luna |
|-------|------------------|--------|--------------|
| Small | Answer + judge only | ~$5 | ~$0.50 |
| Small | Extract every session, then answer + judge | ~$200 | ~$20 |
| Oracle | Extract every session, then answer + judge | ~$15 | ~$1.50 |

The middle row is the LangMem path: each session goes through the same chat model you pass as `model=`. Providers that extract on their own side sit closer to the first row here plus their own bill.

`--limit 1 --split oracle` is cents. Medium is an order of magnitude above Small. Managed cloud resources keep billing until you delete them; each provider page has the teardown steps.

## License

Apache License 2.0.
