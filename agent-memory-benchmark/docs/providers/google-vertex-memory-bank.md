# Google Vertex Memory Bank

Vertex AI Agent Engine Memory Bank. Gemini extracts and consolidates memories per `user_id` scope.

Docs:

- [Memory Bank overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank)
- [Generate memories](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories)
- [API quickstart](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/api-quickstart)
- [Preview announcement](https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-memory-bank-in-public-preview)

CLI id: `vertex-memory-bank`. Extra: `google`.

## Install

```bash
uv sync --extra google
```

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) if you need application-default credentials.

## Auth

```bash
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_GCP_PROJECT_ID
```

Enable the Vertex AI / Agent Engine APIs listed in the Memory Bank setup docs for your project.

```bash
OPENAI_API_KEY=replace-me
GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
GOOGLE_CLOUD_LOCATION=us-central1
```

Answer generation still uses OpenAI unless you change `model`. Memory Bank extraction is billed by Google.

## Provision

Create an Agent Engine with Memory Bank enabled by following the public setup
pages. The wrapper requires its existing resource name:

```bash
--provider-param agent_engine_name=projects/YOUR_GCP_PROJECT_ID/locations/us-central1/reasoningEngines/YOUR_ENGINE_ID
```

Per-example isolation uses `scope.user_id`, not separate engines.

## Run

```bash
uv run memory-bench run \
  --provider vertex-memory-bank \
 --split oracle \
  --limit 1 \
  --run-name smoke-vertex-mb \
  --provider-param model=gpt-4o \
  --provider-param search_limit=25 \
  --provider-param project=YOUR_GCP_PROJECT_ID \
  --provider-param location=us-central1 \
  --provider-param agent_engine_name=projects/YOUR_GCP_PROJECT_ID/locations/us-central1/reasoningEngines/YOUR_ENGINE_ID

uv run memory-bench judge --experiment smoke-vertex-mb --judge-model gpt-4o
```

## Cleanup

Delete the Agent Engine (and Memory Bank data) in Cloud Console or with `gcloud` / the Agent Engine API. An idle engine keeps billing.

## Notes

- `generate_memories` is a long-running operation managed by the Google SDK.
- Gemini extraction tokens are not in harness totals.
- API names and SKUs move as Memory Bank leaves preview. Prefer current Google docs over this page if they disagree.
- This recipe was written from public docs. It was not run against a live Agent Engine.
