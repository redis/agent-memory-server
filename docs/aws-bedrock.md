# AWS Bedrock Models

> **Note:** This documentation has been consolidated into [LLM Providers](llm-providers.md#aws-bedrock).
> This page is kept for reference but the LLM Providers guide is the authoritative source.

The Redis Agent Memory Server supports [Amazon Bedrock](https://aws.amazon.com/bedrock/) for both **embedding models** and **LLM generation models**. This allows you to use AWS-native AI models while keeping your data within the AWS ecosystem.

## Quick Reference

For complete AWS Bedrock configuration, see [LLM Providers - AWS Bedrock](llm-providers.md#aws-bedrock).

**Key points:**
- All LLM operations use [LiteLLM](https://docs.litellm.ai/) internally
- Bedrock embedding models require the `bedrock/` prefix (e.g., `bedrock/amazon.titan-embed-text-v2:0`)
- Bedrock generation models do not need a prefix (e.g., `anthropic.claude-sonnet-4-5-20250929-v1:0`)
- The `[aws]` extra installs `boto3` and `botocore` for AWS authentication

## Overview

Amazon Bedrock provides access to a wide variety of foundation models from leading AI providers. The Redis Agent Memory Server supports using Bedrock for:

1. **Embedding Models** - For semantic search and memory retrieval
2. **LLM Generation Models** - For memory extraction, summarization, and topic modeling

### Supported Embedding Models

> **Important:** Use the `bedrock/` prefix for embedding models.

| Model ID | Provider | Dimensions | Description |
|----------|----------|------------|-------------|
| `bedrock/amazon.titan-embed-text-v2:0` | Amazon | 1024 | Latest Titan embedding model |
| `bedrock/amazon.titan-embed-text-v1` | Amazon | 1536 | Original Titan embedding model |
| `bedrock/cohere.embed-english-v3` | Cohere | 1024 | English-focused embeddings |
| `bedrock/cohere.embed-multilingual-v3` | Cohere | 1024 | Multilingual embeddings |

### Pre-configured LLM Generation Models

The following models are pre-configured in the codebase:

| Model ID | Provider | Max Tokens | Description |
|----------|----------|------------|-------------|
| `anthropic.claude-sonnet-4-5-20250929-v1:0` | Anthropic | 200,000 | Claude 4.5 Sonnet |
| `anthropic.claude-haiku-4-5-20251001-v1:0` | Anthropic | 200,000 | Claude 4.5 Haiku |
| `anthropic.claude-opus-4-5-20251101-v1:0` | Anthropic | 200,000 | Claude 4.5 Opus |

## Installation

AWS Bedrock support requires additional dependencies. Install them with:

```bash
pip install agent-memory-server[aws]
```

This installs:

- `boto3` - AWS SDK for Python
- `botocore` - Low-level AWS client library

## Configuration

### Environment Variables

Configure the following environment variables to use Bedrock models:

```bash
# Required: AWS region where Bedrock is available
AWS_REGION_NAME=us-east-1

# For Bedrock Embedding Models (note: bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0

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

When running on AWS infrastructure (EC2, ECS, Lambda, etc.), use IAM roles for automatic credential management. No explicit credentials are needed.

#### Option 4: AWS SSO / AWS CLI Profile

If you've configured AWS SSO or named profiles:

```bash
# First, login via SSO
aws sso login --profile your-profile

# The server will use the default profile, or set explicitly
export AWS_PROFILE=your-profile
```

### Docker Configuration

The Docker image supports two build targets:

- **`standard`** (default): OpenAI/Anthropic support only
- **`aws`**: Includes AWS Bedrock embedding models support

#### Building the AWS-enabled Image

```bash
# Build directly with Docker
docker build --target aws -t agent-memory-server:aws .

# Or use Docker Compose with the DOCKER_TARGET variable
DOCKER_TARGET=aws docker-compose up --build
```

#### Docker Compose Configuration

When using Docker Compose, set the `DOCKER_TARGET` environment variable to `aws`:

```bash
# Start with AWS Bedrock support
DOCKER_TARGET=aws docker-compose up --build

# Or for the production-like setup
DOCKER_TARGET=aws docker-compose -f docker-compose-task-workers.yml up --build
```

Create a `.env` file with your credentials and configuration:

```bash
# Docker build target
DOCKER_TARGET=aws

# Embedding model (note: bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0

# AWS credentials
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_SESSION_TOKEN=your-session-token  # Optional
```

The Docker Compose files already include the AWS environment variables, so you only need to set them in your `.env` file or environment.

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
# Embedding model (Bedrock - note: bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0

# AWS Configuration
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# Embedding dimensions (must match embedding model)
REDISVL_VECTOR_DIMENSIONS=1024

# Generation model (OpenAI)
GENERATION_MODEL=gpt-4o
OPENAI_API_KEY=your-openai-key

# Other settings
REDIS_URL=redis://localhost:6379
```

### Example 2: Full Bedrock Stack (Recommended for AWS-only deployments)

```bash
# AWS Configuration
AWS_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# Embedding model (Bedrock Titan - note: bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
REDISVL_VECTOR_DIMENSIONS=1024

# Generation models (Bedrock Claude - no prefix needed)
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
SLOW_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
TOPIC_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0

# Other settings
REDIS_URL=redis://localhost:6379
```

### YAML Configuration

```yaml
# config.yaml - Full Bedrock Stack
region_name: us-east-1
embedding_model: amazon.titan-embed-text-v2:0
redisvl_vector_dimensions: 1024
generation_model: anthropic.claude-sonnet-4-5-20250929-v1:0
fast_model: anthropic.claude-haiku-4-5-20251001-v1:0
slow_model: anthropic.claude-sonnet-4-5-20250929-v1:0
topic_model: anthropic.claude-haiku-4-5-20251001-v1:0
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

You can mix and match providers for different use cases:

### Bedrock Embeddings with OpenAI Generation

```bash
# Embeddings via Bedrock (note: bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
AWS_REGION_NAME=us-east-1

# Generation via OpenAI
GENERATION_MODEL=gpt-4o
OPENAI_API_KEY=your-openai-key
```

### Full Bedrock Stack (Embeddings + Generation)

```bash
# All AWS - keep everything within your AWS environment
AWS_REGION_NAME=us-east-1

# Embeddings via Bedrock (note: bedrock/ prefix required)
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
REDISVL_VECTOR_DIMENSIONS=1024

# Generation via Bedrock Claude (no prefix needed)
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
FAST_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
```

### OpenAI Embeddings with Bedrock Generation

```bash
# Embeddings via OpenAI
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-openai-key

# Generation via Bedrock
AWS_REGION_NAME=us-east-1
GENERATION_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
```

This flexibility allows you to:
- Keep all data within AWS for compliance requirements
- Use the best model for each task
- Optimize costs by choosing appropriate models for different operations

## Troubleshooting

### "AWS-related dependencies might be missing"

Install the AWS extras:

```bash
pip install agent-memory-server[aws]
```

### "Missing environment variable 'AWS_REGION_NAME'"

Set the AWS region:

```bash
export AWS_REGION_NAME=us-east-1
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
