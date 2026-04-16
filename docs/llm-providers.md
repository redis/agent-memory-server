# LLM Providers

The Redis Agent Memory Server uses [LiteLLM](https://docs.litellm.ai/) as a unified interface for all LLM operations. This enables support for 100+ LLM providers without code changes—just configure environment variables.

## Architecture Overview

All LLM operations go through a single `LLMClient` abstraction:

```
┌──────────────────────────────────────────────────────────┐
│                      LLMClient                           │
│  ┌─────────────────┐  ┌───────────────────────────────┐  │
│  │ Chat Completions│  │ Embeddings (LiteLLMEmbeddings)│  │
│  └────────┬────────┘  └──────────────┬────────────────┘  │
│           │                          │                   │
│           └──────────┬───────────────┘                   │
│                      ▼                                   │
│               ┌──────────────┐                           │
│               │   LLM Proxy  │                           │
│               └──────┬───────┘                           │
└──────────────────────┼───────────────────────────────────┘
                       ▼
    ┌──────────┬───────────────┬──────────────┐
    │  OpenAI  │   Anthropic   │   Bedrock    │  ... 100+ providers
    └──────────┴───────────────┴──────────────┘
```

**Benefits:**
- Single configuration point for all LLM operations
- Swap providers without code changes
- Consistent error handling and logging
- Automatic model validation at startup

## Quick Start

Set environment variables for your chosen provider:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...
export GENERATION_MODEL=gpt-4o
export EMBEDDING_MODEL=text-embedding-3-small

# Anthropic (requires a separate embedding provider — Anthropic has no embedding models)
export ANTHROPIC_API_KEY=sk-ant-...
export GENERATION_MODEL=claude-3-5-sonnet-20241022
export OPENAI_API_KEY=sk-...                    # Needed for embeddings
export EMBEDDING_MODEL=text-embedding-3-small   # Use OpenAI for embeddings

# AWS Bedrock (full stack — see "AWS Bedrock" section for details)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1                                  # Required: LiteLLM Bedrock calls
export REGION_NAME=us-east-1                                      # Optional: server-side boto3 utilities
export GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0 # No prefix for generation
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0       # bedrock/ prefix REQUIRED
export REDISVL_VECTOR_DIMENSIONS=1024                             # Must match embedding model
```

> **Bedrock users:** You must also install the `[aws]` extra (`pip install agent-memory-server[aws]` or `uv sync --extra aws`) to get `boto3` and `botocore`. See [AWS Bedrock](aws-bedrock.md) for a full walkthrough.

## Supported Providers

### Generation Models (Chat Completions)

| Provider | Model Format | Environment Variables | Example |
|----------|-------------|----------------------|---------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `claude-3-*` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |
| AWS Bedrock | `anthropic.claude-*` | AWS credentials + `AWS_REGION_NAME` | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Ollama | `ollama/<model>` | `OLLAMA_API_BASE` | `ollama/llama2` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE` | `azure/my-gpt4-deployment` |
| Google Gemini | `gemini/<model>` | `GEMINI_API_KEY` | `gemini/gemini-1.5-pro` |

### Embedding Models

See [Embedding Providers](embedding-providers.md) for complete embedding configuration.

**Quick reference:**
```bash
# OpenAI (default)
EMBEDDING_MODEL=text-embedding-3-small

# AWS Bedrock (use bedrock/ prefix)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0

# Ollama (local)
EMBEDDING_MODEL=ollama/nomic-embed-text
REDISVL_VECTOR_DIMENSIONS=768  # Required for Ollama
```

## Provider Configuration

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export GENERATION_MODEL=gpt-4o           # Primary generation model
export FAST_MODEL=gpt-4o-mini            # Fast tasks (topic extraction, etc.)
export EMBEDDING_MODEL=text-embedding-3-small
```

**Supported models:**
- `gpt-4o`, `gpt-4o-mini` (recommended)
- `gpt-4`, `gpt-4-32k`
- `o1`, `o1-mini`, `o3-mini` (reasoning models)
- `text-embedding-3-small`, `text-embedding-3-large`

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GENERATION_MODEL=claude-3-5-sonnet-20241022
export FAST_MODEL=claude-3-5-haiku-20241022

# Anthropic doesn't have embedding models - use OpenAI or another provider
export OPENAI_API_KEY=sk-...
export EMBEDDING_MODEL=text-embedding-3-small
```

**Supported models:**
- `claude-3-7-sonnet-latest`, `claude-3-7-sonnet-20250219`
- `claude-3-5-sonnet-latest`, `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-latest`, `claude-3-5-haiku-20241022`
- `claude-3-opus-latest`

> **Note:** Anthropic does not provide embedding models. Use OpenAI, Bedrock, or another provider for embeddings.

### AWS Bedrock

AWS Bedrock provides access to foundation models from multiple providers (Anthropic Claude, Amazon Titan, Cohere, etc.) through AWS infrastructure.

> **Full guide:** See [AWS Bedrock Models](aws-bedrock.md) for a complete walkthrough including a Quick Start demo, Docker Compose instructions, IAM policies, and troubleshooting.

#### Installation

AWS Bedrock support requires the `[aws]` extra, which installs `boto3` (`>=1.42.1`) and `botocore` (`>=1.42.1`). Without these packages, any Bedrock operation will fail at import time.

```bash
# With pip
pip install agent-memory-server[aws]

# With uv (recommended — used by this project)
uv sync --extra aws
```

#### Authentication

Bedrock uses standard AWS credentials. Configure using any of these methods:

```bash
# Option 1: Environment variables (recommended for development)
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1    # Required: LiteLLM's Bedrock API calls
export REGION_NAME=us-east-1        # Optional: server-side boto3 model-existence checks

# Option 2: AWS CLI profile
export AWS_PROFILE=my-profile
export AWS_REGION_NAME=us-east-1

# Option 3: IAM role (recommended for production on AWS)
# No explicit credentials needed — LiteLLM discovers them from instance metadata
export AWS_REGION_NAME=us-east-1

# Option 4: AWS SSO
aws sso login --profile your-profile
export AWS_PROFILE=your-profile
```

> **Why two region variables?** `AWS_REGION_NAME` is required for LiteLLM's Bedrock inference calls. `REGION_NAME` is only needed if you use the server's optional `boto3`-based model-existence checks (`_aws/utils.py`). When both are set, use the same value.

#### Generation Models

Generation models use Bedrock-native model IDs **without** a prefix — LiteLLM recognises them automatically.

```bash
# Claude models on Bedrock (no prefix needed)
export GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
export FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
```

**Anthropic Claude models on Bedrock** (models marked ✓ are pre-configured in `MODEL_CONFIGS`):

| Model ID | Description | Pre-configured |
|----------|-------------|:--------------:|
| `anthropic.claude-opus-4-6-v1` | Claude Opus 4.6 — latest | |
| `anthropic.claude-sonnet-4-6` | Claude Sonnet 4.6 | |
| `anthropic.claude-opus-4-5-20251101-v1:0` | Claude Opus 4.5 | ✓ |
| `anthropic.claude-sonnet-4-5-20250929-v1:0` | Claude Sonnet 4.5 | ✓ |
| `anthropic.claude-haiku-4-5-20251001-v1:0` | Claude Haiku 4.5 — fast & cost-effective | ✓ |
| `anthropic.claude-opus-4-1-20250805-v1:0` | Claude Opus 4.1 | |
| `anthropic.claude-sonnet-4-20250514-v1:0` | Claude Sonnet 4 | |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Claude 3.5 Haiku | |
| `anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku | |

Any Bedrock model can be used by setting the environment variable — LiteLLM routes based on the model ID convention.

#### Embedding Models

> **Important:** Bedrock embedding models **require** the `bedrock/` prefix.

LiteLLM needs the `bedrock/` prefix to distinguish Bedrock embeddings from other providers. If you omit it, the server auto-adds the prefix and emits a **deprecation warning** — this fallback will be removed in a future release.

```bash
# Correct — use bedrock/ prefix
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
export REDISVL_VECTOR_DIMENSIONS=1024  # Must match embedding model dimensions

# Deprecated — works but emits a warning
export EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
```

**Supported Bedrock embedding models:**

| Model ID | Dimensions | Description |
|----------|------------|-------------|
| `bedrock/amazon.titan-embed-text-v2:0` | 1024 | Latest Titan text embedding (recommended) |
| `bedrock/amazon.titan-embed-text-v1` | 1536 | Original Titan text embedding |
| `bedrock/amazon.titan-embed-image-v1` | 1024 | Titan multimodal (text + image) embedding * |
| `bedrock/cohere.embed-english-v3` | 1024 | English-focused |
| `bedrock/cohere.embed-multilingual-v3` | 1024 | Multilingual |
| `bedrock/cohere.embed-v4:0` | 1024 | Cohere Embed v4 — text + image * |

> \* Models marked with **\*** are not in `MODEL_CONFIGS`, so their dimensions cannot be auto-resolved. Set `REDISVL_VECTOR_DIMENSIONS=1024` explicitly when using them.

#### Enabling Bedrock Models

Before using a Bedrock model, enable it in the AWS Console:

1. Navigate to **Amazon Bedrock** in the AWS Console
2. Select **Model access** from the left navigation
3. Click **Manage model access**
4. Enable the models you need
5. Wait for access to be granted (usually immediate for Amazon models)

#### IAM Permissions

Your IAM role/user needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-*"
      ]
    }
  ]
}
```

#### Docker

The Dockerfile has a dedicated `aws` build target, and `docker-compose.yml` provides an `aws` profile with pre-built AWS images:

```bash
# Build the AWS-enabled image directly
docker build --target aws -t agent-memory-server:aws .

# Or use Docker Compose with the aws profile
docker-compose --profile aws up
```

Pass AWS credentials at runtime:

```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e REGION_NAME=us-east-1 -e AWS_REGION_NAME=us-east-1 \
  -e GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0 \
  -e EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0 \
  -e REDISVL_VECTOR_DIMENSIONS=1024 \
  -p 8000:8000 redislabs/agent-memory-server-aws:latest
```

#### Complete Example

Full Bedrock stack (keep all AI operations within AWS):

```bash
# AWS credentials
export REGION_NAME=us-east-1
export AWS_REGION_NAME=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Embeddings (bedrock/ prefix REQUIRED)
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
export REDISVL_VECTOR_DIMENSIONS=1024

# Generation (no prefix needed)
export GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
export FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0

# Start Redis and the server
docker-compose up redis -d
uv run agent-memory api
```

### Ollama (Local Models)

Run models locally with [Ollama](https://ollama.ai/):

```bash
# Start Ollama server
ollama serve

# Pull models
ollama pull llama2
ollama pull nomic-embed-text

# Configure agent-memory-server
export OLLAMA_API_BASE=http://localhost:11434
export GENERATION_MODEL=ollama/llama2
export EMBEDDING_MODEL=ollama/nomic-embed-text
export REDISVL_VECTOR_DIMENSIONS=768  # Required - Ollama models vary
```

**Common Ollama models:**
- Generation: `ollama/llama2`, `ollama/mistral`, `ollama/codellama`
- Embeddings: `ollama/nomic-embed-text` (768d), `ollama/mxbai-embed-large` (1024d)

> **Note:** Always set `REDISVL_VECTOR_DIMENSIONS` for Ollama embedding models.

### Azure OpenAI

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-resource.openai.azure.com/
export AZURE_API_VERSION=2024-02-15-preview

# Use azure/ prefix with your deployment name
export GENERATION_MODEL=azure/my-gpt4-deployment
export EMBEDDING_MODEL=azure/my-embedding-deployment
```

### Google Gemini

```bash
export GEMINI_API_KEY=...
export GENERATION_MODEL=gemini/gemini-1.5-pro
export FAST_MODEL=gemini/gemini-1.5-flash

# Note: Gemini API (gemini/ prefix) only supports generation models.
# For embeddings, use Vertex AI or another provider:
export EMBEDDING_MODEL=text-embedding-3-small  # OpenAI
# Or with Vertex AI:
# export EMBEDDING_MODEL=vertex_ai/text-embedding-004
```

## Model Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GENERATION_MODEL` | Primary model for AI tasks | `gpt-5` |
| `FAST_MODEL` | Fast model for topic extraction, etc. | `gpt-5-mini` |
| `SLOW_MODEL` | Slower, more capable model for complex tasks | `gpt-5` |
| `EMBEDDING_MODEL` | Model for vector embeddings | `text-embedding-3-small` |
| `REDISVL_VECTOR_DIMENSIONS` | Fallback/override for embedding dimensions when they cannot be resolved from `MODEL_CONFIGS` | `1536` |

### Model Validation

The server validates models at startup:
- Checks model exists in LiteLLM's model registry
- Verifies required API keys are set
- Logs warnings for deprecated model names

## Troubleshooting

### Common Issues

**"API key not found"**
```bash
# Check your API key is set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

**"Model not found"**
- Verify model name matches LiteLLM format
- Check provider prefix (e.g., `bedrock/`, `ollama/`, `azure/`)
- See [LiteLLM model list](https://docs.litellm.ai/docs/providers)

**"Embedding dimension mismatch"**
```bash
# Set dimensions explicitly
export REDISVL_VECTOR_DIMENSIONS=1024
```

**Bedrock "Access Denied"**
- Verify IAM permissions include `bedrock:InvokeModel`
- Check model is enabled in your AWS region
- Ensure both `REGION_NAME` and `AWS_REGION_NAME` are set correctly
- See [AWS Bedrock Troubleshooting](aws-bedrock.md#troubleshooting) for more details

**Bedrock "AWS-related dependencies might be missing"**
- Install the `[aws]` extra: `pip install agent-memory-server[aws]` or `uv sync --extra aws`

### Debug Logging

Enable debug logging to troubleshoot LLM issues:

```bash
export LOG_LEVEL=DEBUG
```

## Migration from Previous Versions

If upgrading from a version that used provider-specific embeddings:

### Bedrock Embedding Model Names

**Before (deprecated):**
```bash
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
```

**After (recommended):**
```bash
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
```

The server will auto-correct unprefixed Bedrock model names with a deprecation warning.

### Removed Dependencies

The following are no longer required:
- `langchain-aws` - Bedrock now uses LiteLLM
- `langchain-openai` - OpenAI embeddings now use LiteLLM

## See Also

- [AWS Bedrock Models](aws-bedrock.md) - Complete Bedrock guide with Quick Start, IAM policies, and Docker setup
- [Embedding Providers](embedding-providers.md) - Detailed embedding configuration
- [Configuration](configuration.md) - All environment variables
- [Query Optimization](query-optimization.md) - Model selection for query optimization
- [LiteLLM Documentation](https://docs.litellm.ai/) - Full provider list
