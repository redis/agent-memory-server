# AWS Bedrock Models

The Redis Agent Memory Server supports [Amazon Bedrock](https://aws.amazon.com/bedrock/) for both **embedding models** and **LLM generation models**. This allows you to use AWS-native AI models while keeping your data within the AWS ecosystem.

> **See also:** [LLM Providers](llm-providers.md#aws-bedrock) for a broader overview of all supported providers, and [Embedding Providers](embedding-providers.md) for embedding-specific configuration.

## Quick Start — Run a Full Bedrock-Backed Instance

Follow these steps to get the memory server running entirely on AWS Bedrock in under five minutes.

### 1. Install the `[aws]` extra

The server's core install does **not** include AWS SDK libraries. You must install the `[aws]` extra so that `boto3` (AWS SDK for Python) and `botocore` (low-level AWS client library) are available at runtime. Without these packages, any Bedrock operation will fail with an import error.

```bash
# With pip
pip install agent-memory-server[aws]

# With uv (used by this project)
uv sync --extra aws
```

### 2. Export environment variables

```bash
# ── AWS credentials ──────────────────────────────────────────────
# (or use an IAM role / AWS CLI profile instead — see "AWS Credentials" below)
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export REGION_NAME=us-east-1          # Used by the server's own model-validation client
export AWS_REGION_NAME=us-east-1      # Used by LiteLLM for Bedrock API calls

# ── Bedrock embedding model (bedrock/ prefix REQUIRED) ──────────
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
export REDISVL_VECTOR_DIMENSIONS=1024   # Must match the embedding model

# ── Bedrock generation models (NO prefix needed) ────────────────
export GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
export FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0

# ── Redis ────────────────────────────────────────────────────────
export REDIS_URL=redis://localhost:6379
```

### 3. Start Redis and the server

```bash
# Start Redis (requires Docker)
docker-compose up redis -d

# Start the memory server
uv run agent-memory api
```

The REST API is now available at `http://localhost:8000` (docs at `/docs`).

> **Tip:** To also run background tasks (memory extraction, compaction, etc.), start a worker in a second terminal:
> ```bash
> uv run agent-memory task-worker
> ```

---

## Why Two Region Variables?

The server reads the AWS region in **two** places:

| Variable | Read by | Purpose |
|----------|---------|---------|
| `REGION_NAME` | The server's Settings (pydantic-settings) | Creating `boto3` sessions for model-existence validation |
| `AWS_REGION_NAME` | LiteLLM | Making the actual Bedrock inference API calls |

Set **both** to the same value to avoid surprises. If you rely solely on an IAM role or AWS CLI profile, `boto3` and LiteLLM may auto-detect the region from the instance metadata or `~/.aws/config`, but explicitly setting the variables is recommended.

---

## Understanding the `bedrock/` Prefix

All LLM operations in the server go through [LiteLLM](https://docs.litellm.ai/), which uses a provider prefix to route requests to the correct backend.

| Model type | Prefix required? | Example |
|------------|------------------|---------|
| **Embedding** | **Yes** — must include `bedrock/` | `bedrock/amazon.titan-embed-text-v2:0` |
| **Generation (chat)** | **No** — Bedrock model IDs are recognized automatically | `anthropic.claude-sonnet-4-5-20250929-v1:0` |

**Why the difference?** LiteLLM can infer the provider for generation models from the Bedrock-style model ID (e.g., `anthropic.claude-*`). For embedding models, however, the `bedrock/` prefix is the only way LiteLLM distinguishes a Bedrock embedding call from other providers. If you omit the prefix on an embedding model, the server will auto-add it and emit a **deprecation warning** — but this behaviour will be removed in a future release.

---

## Supported Models

### Embedding Models

> **Important:** Always use the `bedrock/` prefix for embedding models.

| Model ID | Provider | Dimensions | Description |
|----------|----------|------------|-------------|
| `bedrock/amazon.titan-embed-text-v2:0` | Amazon | 1024 | Latest Titan text embedding (recommended) |
| `bedrock/amazon.titan-embed-text-v1` | Amazon | 1536 | Original Titan text embedding |
| `bedrock/amazon.titan-embed-image-v1` | Amazon | 1024 | Titan multimodal (text + image) embedding |
| `bedrock/cohere.embed-english-v3` | Cohere | 1024 | English-focused embeddings |
| `bedrock/cohere.embed-multilingual-v3` | Cohere | 1024 | Multilingual embeddings |
| `bedrock/cohere.embed-v4:0` | Cohere | 1024 | Cohere Embed v4 — text + image embedding |

### Generation Models (Anthropic Claude on Bedrock)

The following Anthropic Claude models are available on Bedrock. Models marked **pre-configured** have entries in `MODEL_CONFIGS` (in `config.py`) with validated token limits; the others are fully usable by setting the corresponding environment variable — LiteLLM routes the request based on the Bedrock model ID.

| Model ID | Description | Pre-configured |
|----------|-------------|:--------------:|
| `anthropic.claude-opus-4-6-v1` | Claude Opus 4.6 — latest and most capable | |
| `anthropic.claude-sonnet-4-6` | Claude Sonnet 4.6 | |
| `anthropic.claude-opus-4-5-20251101-v1:0` | Claude Opus 4.5 | ✓ |
| `anthropic.claude-sonnet-4-5-20250929-v1:0` | Claude Sonnet 4.5 | ✓ |
| `anthropic.claude-haiku-4-5-20251001-v1:0` | Claude Haiku 4.5 — fast & cost-effective | ✓ |
| `anthropic.claude-opus-4-1-20250805-v1:0` | Claude Opus 4.1 | |
| `anthropic.claude-sonnet-4-20250514-v1:0` | Claude Sonnet 4 | |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Claude 3.5 Haiku | |
| `anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku | |

> **Tip:** For the recommended quick-start configuration, use `anthropic.claude-sonnet-4-5-20250929-v1:0` as `GENERATION_MODEL` and `anthropic.claude-haiku-4-5-20251001-v1:0` as `FAST_MODEL`.

## Installation

AWS Bedrock support requires additional dependencies. Install the `[aws]` extra:

```bash
# With pip
pip install agent-memory-server[aws]

# With uv (recommended — used by this project)
uv sync --extra aws
```

This installs:

- **`boto3`** (`>=1.42.1,<2.0.0`) — AWS SDK for Python
- **`botocore`** (`>=1.42.1,<2.0.0`) — Low-level AWS client library

> **Without these packages**, any attempt to use Bedrock models will fail at import time. The standard install (`pip install agent-memory-server`) does **not** include them.

## Configuration

### Environment Variables

Configure the following environment variables to use Bedrock models:

```bash
# Required: AWS region where Bedrock is available
REGION_NAME=us-east-1            # For the server's own boto3 sessions
AWS_REGION_NAME=us-east-1        # For LiteLLM's Bedrock calls

# For Bedrock Embedding Models (bedrock/ prefix REQUIRED)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
REDISVL_VECTOR_DIMENSIONS=1024   # Must match the embedding model's output dimensions

# For Bedrock LLM Generation Models (no prefix needed)
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0

# AWS Credentials (choose one method below)
```

### AWS Credentials

There are several ways to provide AWS credentials:

#### Option 1: Environment Variables (Explicit)

```bash
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_SESSION_TOKEN=your-session-token  # Optional, for temporary credentials
```

#### Option 2: AWS Credentials File

The server will automatically use credentials from `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
```

#### Option 3: IAM Role (Recommended for AWS deployments)

When running on AWS infrastructure (EC2, ECS, Lambda, etc.), use IAM roles for automatic credential management. No explicit credentials are needed — `boto3` and LiteLLM will discover credentials from the instance metadata service.

#### Option 4: AWS SSO / AWS CLI Profile

If you've configured AWS SSO or named profiles:

```bash
# First, login via SSO
aws sso login --profile your-profile

# The server will use the default profile, or set explicitly
export AWS_PROFILE=your-profile
```

### Docker Configuration

The Dockerfile provides two build targets (multi-stage):

- **`standard`** (default) — OpenAI / Anthropic support only
- **`aws`** — Includes `boto3` and `botocore` for AWS Bedrock support

#### Building the AWS-enabled Image

```bash
# Build directly with Docker
docker build --target aws -t agent-memory-server:aws .
```

#### Docker Compose

The `docker-compose.yml` ships with a dedicated **`aws` profile** that uses pre-built AWS images (`redislabs/agent-memory-server-aws`). Activate it with `--profile aws`:

```bash
# Start the full AWS stack (API + MCP + task worker + Redis)
docker-compose --profile aws up

# Or start only the API and Redis
docker-compose --profile aws up api-aws redis
```

> **Note:** The `aws` profile services (`api-aws`, `mcp-aws`, `task-worker-aws`) are separate from the standard services. Do **not** mix profiles — run either `docker-compose up` (standard) or `docker-compose --profile aws up`.

Create a `.env` file with your credentials and model configuration. The Docker Compose AWS services read this file automatically:

```bash
# AWS credentials
REGION_NAME=us-east-1
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_SESSION_TOKEN=your-session-token  # Optional, for temporary credentials

# Embedding model (bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
REDISVL_VECTOR_DIMENSIONS=1024

# Generation models (override the defaults in docker-compose.yml if desired)
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
```

You can also pass AWS credentials at runtime with `docker run`:

```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e REGION_NAME=us-east-1 -e AWS_REGION_NAME=us-east-1 \
  -e EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0 \
  -e REDISVL_VECTOR_DIMENSIONS=1024 \
  -e GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0 \
  -p 8000:8000 redislabs/agent-memory-server-aws:latest
```

Or mount your AWS credentials directory:

```bash
docker run -v ~/.aws:/root/.aws:ro \
  -e AWS_PROFILE=my-profile \
  -e REGION_NAME=us-east-1 -e AWS_REGION_NAME=us-east-1 \
  -p 8000:8000 redislabs/agent-memory-server-aws:latest
```

## Required IAM Permissions

The AWS credentials must have permissions to invoke Bedrock models. Here's a minimal IAM policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```

For production, scope down the `Resource` to specific model ARNs:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "bedrock:ListFoundationModels",
            "Resource": "*"
        }
    ]
}
```

**Note**: When using Bedrock LLM models for generation tasks (memory extraction, summarization, topic modeling), ensure your IAM policy includes permissions for all the models you've configured (`GENERATION_MODEL`, `FAST_MODEL`, `SLOW_MODEL`, `TOPIC_MODEL`).

## Vector Dimensions

When using Bedrock embedding models, make sure to update the vector dimensions setting to match your chosen model:

```bash
# For Titan v2 and Cohere models (1024 dimensions)
REDISVL_VECTOR_DIMENSIONS=1024

# For Titan v1 (1536 dimensions)
REDISVL_VECTOR_DIMENSIONS=1536
```

## Complete Configuration Examples

### Example 1: Bedrock Embeddings with OpenAI Generation

```bash
# Embedding model (Bedrock — bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
REDISVL_VECTOR_DIMENSIONS=1024

# AWS Configuration
REGION_NAME=us-east-1
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# Generation model (OpenAI)
GENERATION_MODEL=gpt-4o
OPENAI_API_KEY=your-openai-key

# Other settings
REDIS_URL=redis://localhost:6379
```

### Example 2: Full Bedrock Stack (Recommended for AWS-only deployments)

```bash
# AWS Configuration
REGION_NAME=us-east-1
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# Embedding model (Bedrock Titan — bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
REDISVL_VECTOR_DIMENSIONS=1024

# Generation models (Bedrock Claude — no prefix needed)
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
SLOW_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0

# Other settings
REDIS_URL=redis://localhost:6379
```

### Example 3: OpenAI Embeddings with Bedrock Generation

```bash
# Embeddings via OpenAI
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-openai-key

# Generation via Bedrock
REGION_NAME=us-east-1
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0

REDIS_URL=redis://localhost:6379
```

### YAML Configuration

```yaml
# config.yaml - Full Bedrock Stack
region_name: us-east-1
embedding_model: bedrock/amazon.titan-embed-text-v2:0   # bedrock/ prefix required
redisvl_vector_dimensions: 1024
generation_model: anthropic.claude-sonnet-4-5-20250929-v1:0
fast_model: anthropic.claude-haiku-4-5-20251001-v1:0
slow_model: anthropic.claude-sonnet-4-5-20250929-v1:0
redis_url: redis://localhost:6379
```

## Model Validation

The server validates that the specified Bedrock embedding model exists in your configured region at startup. If the model is not found, you'll see an error like:

```
ValueError: Bedrock embedding model amazon.titan-embed-text-v2:0 not found in region us-east-1.
```

This helps catch configuration errors early. Common causes:

1. **Model not enabled**: You may need to enable the model in the Bedrock console
2. **Wrong region**: The model may not be available in your configured region
3. **Typo in model ID**: Double-check the model ID spelling

## Enabling Bedrock Models in AWS Console

Before using a Bedrock model, you must enable it in the AWS Console:

1. Navigate to **Amazon Bedrock** in the AWS Console
2. Select **Model access** from the left navigation
3. Click **Manage model access**
4. Enable the embedding models you want to use
5. Wait for access to be granted (usually immediate for Amazon models)

## Mixing Providers

You can mix and match providers for embeddings and generation independently. See the [Complete Configuration Examples](#complete-configuration-examples) section above for full `.env` snippets covering each combination.

This flexibility allows you to:

- Keep all data within AWS for compliance requirements
- Use the best model for each task (e.g., OpenAI embeddings + Bedrock generation)
- Optimize costs by choosing appropriate models for different operations

## Troubleshooting

### "AWS-related dependencies might be missing"

You need to install the `[aws]` extra. The standard install does not include `boto3`:

```bash
pip install agent-memory-server[aws]
# or with uv:
uv sync --extra aws
```

### "Missing environment variable 'REGION_NAME'"

The server's Settings class reads the region from `REGION_NAME`. Set it:

```bash
export REGION_NAME=us-east-1
export AWS_REGION_NAME=us-east-1   # Also set this for LiteLLM
```

### "Bedrock embedding model not found"

1. Verify the model ID is correct
2. Check the model is available in your region
3. Ensure the model is enabled in Bedrock console
4. Verify your IAM permissions include `bedrock:ListFoundationModels`

### "Bedrock LLM model not responding correctly"

1. Verify the model ID matches exactly (including version suffix like `:0`)
2. Check the model is enabled in your Bedrock console
3. Verify your IAM permissions include `bedrock:InvokeModel` for the specific model
4. Some models may have different regional availability - check AWS documentation

### "Error creating chat completion with Bedrock"

1. Check that the model ID is correct and the model is enabled
2. Verify your AWS credentials have `bedrock:InvokeModel` permission
3. Check the request isn't exceeding the model's token limits
4. Review CloudWatch logs for detailed error messages

### Credential Errors

If you see authentication errors:

1. Verify your AWS credentials are correctly set
2. Check the credentials have the required IAM permissions
3. If using temporary credentials, ensure they haven't expired
4. Try running `aws sts get-caller-identity` to verify your credentials work

### Model-Specific Issues

**Model IDs**: Ensure you're using the Bedrock-specific model IDs (e.g., `anthropic.claude-sonnet-4-5-20250929-v1:0`) not the direct provider API model IDs.

**Embedding Model Prefix**: Bedrock embedding models require the `bedrock/` prefix (e.g., `bedrock/amazon.titan-embed-text-v2:0`). Unprefixed names will work but emit a deprecation warning.

**LiteLLM Backend**: All Bedrock operations use [LiteLLM](https://docs.litellm.ai/) internally, which provides a unified interface for all Bedrock models. Model-specific formatting is handled automatically.

## Performance Considerations

- **Latency**: Bedrock API calls may have different latency characteristics than OpenAI
- **Rate limits**: Check your Bedrock service quotas for your region
- **Caching**: The server caches model existence checks for 1 hour to reduce API calls
- **Cost**: Review Bedrock pricing for your chosen embedding model

## Related Documentation

- [LLM Providers](llm-providers.md) - Comprehensive LLM provider guide (recommended)
- [Embedding Providers](embedding-providers.md) - Embedding model configuration
- [Configuration](configuration.md) - Full configuration reference
- [Custom Memory Vector Databases](custom-memory-vector-db.md) - Custom memory vector database setup
- [Getting Started](getting-started.md) - Initial setup guide
