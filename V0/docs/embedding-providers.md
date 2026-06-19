# Embedding Providers

The Redis Agent Memory Server uses LiteLLM for embeddings, enabling support for many embedding providers out of the box.

## Quick Start

Set the `EMBEDDING_MODEL` environment variable to your desired model:

```bash
# OpenAI (default)
export EMBEDDING_MODEL=text-embedding-3-small

# AWS Bedrock
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0

# Ollama (local)
export EMBEDDING_MODEL=ollama/nomic-embed-text
```

## Supported Providers

| Provider | Model Format | Environment Variables | Example |
|----------|-------------|----------------------|---------|
| OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` | `text-embedding-3-large` |
| AWS Bedrock | `bedrock/<model-id>` | AWS credentials | `bedrock/amazon.titan-embed-text-v2:0` |
| Ollama | `ollama/<model>` | `OLLAMA_API_BASE` | `ollama/nomic-embed-text` |
| HuggingFace | `huggingface/<org>/<model>` | `HUGGINGFACE_API_KEY` | `huggingface/BAAI/bge-large-en` |
| Cohere | `cohere/<model>` | `COHERE_API_KEY` | `cohere/embed-english-v3.0` |
| Vertex AI | `vertex_ai/<model>` | GCP credentials | `vertex_ai/text-embedding-004` |
| Mistral | `mistral/<model>` | `MISTRAL_API_KEY` | `mistral/mistral-embed` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE` | `azure/my-embedding-deployment` |

> **Note:** Google's embedding models (`text-embedding-004`, `text-embedding-005`) are available via Vertex AI, not the Gemini API. The `gemini/` prefix only supports generation models.

## Provider Configuration

### OpenAI

```bash
export EMBEDDING_MODEL=text-embedding-3-small
export OPENAI_API_KEY=sk-...
```

Available models:
- `text-embedding-3-small` (1536 dimensions, recommended)
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions, legacy)

### AWS Bedrock

```bash
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1
```

Available models:
- `bedrock/amazon.titan-embed-text-v2:0` (1024 dimensions, recommended)
- `bedrock/amazon.titan-embed-text-v1` (1536 dimensions)
- `bedrock/cohere.embed-english-v3` (1024 dimensions)
- `bedrock/cohere.embed-multilingual-v3` (1024 dimensions)

> **Note**: Always use the `bedrock/` prefix. Unprefixed Bedrock model names are deprecated.

### Ollama (Local)

```bash
export EMBEDDING_MODEL=ollama/nomic-embed-text
export OLLAMA_API_BASE=http://localhost:11434
export REDISVL_VECTOR_DIMENSIONS=768  # Required for Ollama models
```

Popular models:
- `ollama/nomic-embed-text` (768 dimensions)
- `ollama/mxbai-embed-large` (1024 dimensions)
- `ollama/all-minilm` (384 dimensions)

### HuggingFace

```bash
export EMBEDDING_MODEL=huggingface/BAAI/bge-large-en
export HUGGINGFACE_API_KEY=hf_...
export REDISVL_VECTOR_DIMENSIONS=1024  # Required for HuggingFace models
```

### Cohere

```bash
export EMBEDDING_MODEL=cohere/embed-english-v3.0
export COHERE_API_KEY=...
export REDISVL_VECTOR_DIMENSIONS=1024
```

## Embedding Dimensions

The server needs to know the embedding dimensions for the Redis vector index. Dimensions are resolved in this order:

1. **LiteLLM auto-detection** - Works for OpenAI and Bedrock models
2. **MODEL_CONFIGS lookup** - Pre-configured models in the server
3. **REDISVL_VECTOR_DIMENSIONS** - Explicit override (required for unknown models)

### Setting Dimensions Explicitly

For models not in our config (Ollama, HuggingFace, etc.), set dimensions explicitly:

```bash
export EMBEDDING_MODEL=ollama/nomic-embed-text
export REDISVL_VECTOR_DIMENSIONS=768
```

### Common Model Dimensions

| Model | Dimensions |
|-------|-----------|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `bedrock/amazon.titan-embed-text-v2:0` | 1024 |
| `bedrock/amazon.titan-embed-text-v1` | 1536 |
| `ollama/nomic-embed-text` | 768 |
| `ollama/mxbai-embed-large` | 1024 |
| `cohere/embed-english-v3.0` | 1024 |

## Migration Guide

### From OpenAI-only Setup

No changes required. The default `text-embedding-3-small` continues to work.

### From Bedrock (Legacy)

If you were using unprefixed Bedrock model names:

```bash
# Before (deprecated)
export EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# After (recommended)
export EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
```

The server will auto-add the prefix and emit a deprecation warning, but updating your config is recommended.

## Troubleshooting

### "Unknown embedding model" Error

The model isn't in our config and LiteLLM can't auto-detect dimensions. Set `REDISVL_VECTOR_DIMENSIONS`:

```bash
export REDISVL_VECTOR_DIMENSIONS=768
```

### Bedrock "Model not found" Error

1. Ensure the model is available in your AWS region
2. Check your IAM permissions include `bedrock:InvokeModel`
3. Use the `bedrock/` prefix: `bedrock/amazon.titan-embed-text-v2:0`

### Ollama Connection Error

Ensure Ollama is running and set the API base:

```bash
export OLLAMA_API_BASE=http://localhost:11434
```
