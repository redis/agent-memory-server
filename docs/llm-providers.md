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
│               │   LiteLLM    │                           │
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

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export GENERATION_MODEL=claude-3-5-sonnet-20241022
export EMBEDDING_MODEL=text-embedding-3-small  # Use OpenAI for embeddings

# AWS Bedrock
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1
export GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
```

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

#### Authentication

Bedrock uses standard AWS credentials. Configure using any of these methods:

```bash
# Option 1: Environment variables (recommended for development)
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1

# Option 2: AWS CLI profile
export AWS_PROFILE=my-profile
export AWS_REGION_NAME=us-east-1

# Option 3: IAM role (recommended for production on AWS)
# No credentials needed - uses instance/container role
export AWS_REGION_NAME=us-east-1
```

#### Generation Models

```bash
# Claude models on Bedrock (no prefix needed for generation)
export GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
export FAST_MODEL=anthropic.claude-3-5-haiku-20241022-v1:0

# Amazon Titan
export GENERATION_MODEL=amazon.titan-text-premier-v1:0
```

**Supported Bedrock generation models:**
- `anthropic.claude-sonnet-4-5-20250929-v1:0` (recommended)
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `amazon.titan-text-premier-v1:0`
- `amazon.titan-text-express-v1`

#### Embedding Models

> **Important:** Bedrock embedding models require the `bedrock/` prefix.

```bash
# Correct - use bedrock/ prefix
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0

# Deprecated - unprefixed names emit a warning
export EMBEDDING_MODEL=amazon.titan-embed-text-v2:0  # Works but shows deprecation warning
```

**Supported Bedrock embedding models:**
- `bedrock/amazon.titan-embed-text-v2:0` (1024 dimensions, recommended)
- `bedrock/amazon.titan-embed-text-v1` (1536 dimensions)
- `bedrock/cohere.embed-english-v3` (1024 dimensions)
- `bedrock/cohere.embed-multilingual-v3` (1024 dimensions)

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
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-*"
      ]
    }
  ]
}
```

#### Docker Configuration

When running in Docker, pass AWS credentials:

```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION_NAME \
  -e GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0 \
  -e EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0 \
  agent-memory-server
```

Or mount credentials:

```bash
docker run -v ~/.aws:/root/.aws:ro \
  -e AWS_PROFILE=my-profile \
  -e AWS_REGION_NAME=us-east-1 \
  agent-memory-server
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

# Gemini embeddings
export EMBEDDING_MODEL=gemini/text-embedding-004
```

## Model Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GENERATION_MODEL` | Primary model for AI tasks | `gpt-4o-mini` |
| `FAST_MODEL` | Fast model for topic extraction, etc. | Same as `GENERATION_MODEL` |
| `QUERY_OPTIMIZATION_MODEL` | Model for query optimization | Same as `GENERATION_MODEL` |
| `EMBEDDING_MODEL` | Model for vector embeddings | `text-embedding-3-small` |
| `REDISVL_VECTOR_DIMENSIONS` | Override embedding dimensions | Auto-detected |

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
- Ensure correct `AWS_REGION_NAME`

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

- [Embedding Providers](embedding-providers.md) - Detailed embedding configuration
- [Configuration](configuration.md) - All environment variables
- [Query Optimization](query-optimization.md) - Model selection for query optimization
- [LiteLLM Documentation](https://docs.litellm.ai/) - Full provider list
