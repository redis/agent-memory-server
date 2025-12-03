# AWS Bedrock Embedding Models

The Redis Agent Memory Server supports [Amazon Bedrock](https://aws.amazon.com/bedrock/) embedding models as an alternative to OpenAI embeddings. This allows you to use AWS-native embedding models while keeping your data within the AWS ecosystem.

## Overview

Amazon Bedrock provides access to several embedding models that can be used for semantic search and memory retrieval:

| Model ID | Provider | Dimensions | Description |
|----------|----------|------------|-------------|
| `amazon.titan-embed-text-v2:0` | Amazon | 1024 | Latest Titan embedding model |
| `amazon.titan-embed-text-v1` | Amazon | 1536 | Original Titan embedding model |
| `cohere.embed-english-v3` | Cohere | 1024 | English-focused embeddings |
| `cohere.embed-multilingual-v3` | Cohere | 1024 | Multilingual embeddings |

## Installation

AWS Bedrock support requires additional dependencies. Install them with:

```bash
pip install agent-memory-server[aws]
```

This installs:

- `langchain-aws` - LangChain integration for AWS services
- `boto3` - AWS SDK for Python
- `botocore` - Low-level AWS client library

## Configuration

### Environment Variables

Configure the following environment variables to use Bedrock embeddings:

```bash
# Required: Set the embedding model to a Bedrock model ID
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Required: AWS region where Bedrock is available
REGION_NAME=us-east-1

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

# Embedding model
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# AWS credentials
REGION_NAME=us-east-1
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
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
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

## Vector Dimensions

When using Bedrock embedding models, make sure to update the vector dimensions setting to match your chosen model:

```bash
# For Titan v2 and Cohere models (1024 dimensions)
REDISVL_VECTOR_DIMENSIONS=1024

# For Titan v1 (1536 dimensions)
REDISVL_VECTOR_DIMENSIONS=1536
```

## Complete Configuration Example

Here's a complete example using Amazon Titan embeddings:

### Environment Variables

```bash
# Embedding model
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# AWS Configuration
REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# Vector store dimensions (must match embedding model)
REDISVL_VECTOR_DIMENSIONS=1024

# Other settings
REDIS_URL=redis://localhost:6379
GENERATION_MODEL=gpt-4o  # Generation still uses OpenAI/Anthropic
```

### YAML Configuration

```yaml
# config.yaml
embedding_model: amazon.titan-embed-text-v2:0
region_name: us-east-1
redisvl_vector_dimensions: 1024
redis_url: redis://localhost:6379
generation_model: gpt-4o
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

You can use Bedrock for embeddings while using OpenAI or Anthropic for text generation:

```bash
# Embeddings via Bedrock
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
REGION_NAME=us-east-1

# Generation via OpenAI
GENERATION_MODEL=gpt-4o
OPENAI_API_KEY=your-openai-key
```

This is a common pattern when you want to keep embedding data within AWS but leverage the latest generation models.

## Troubleshooting

### "AWS-related dependencies might be missing"

Install the AWS extras:

```bash
pip install agent-memory-server[aws]
```

### "Missing environment variable 'REGION_NAME'"

Set the AWS region:

```bash
export REGION_NAME=us-east-1
```

### "Bedrock embedding model not found"

1. Verify the model ID is correct
2. Check the model is available in your region
3. Ensure the model is enabled in Bedrock console
4. Verify your IAM permissions include `bedrock:ListFoundationModels`

### Credential Errors

If you see authentication errors:

1. Verify your AWS credentials are correctly set
2. Check the credentials have the required IAM permissions
3. If using temporary credentials, ensure they haven't expired
4. Try running `aws sts get-caller-identity` to verify your credentials work

## Performance Considerations

- **Latency**: Bedrock API calls may have different latency characteristics than OpenAI
- **Rate limits**: Check your Bedrock service quotas for your region
- **Caching**: The server caches model existence checks for 1 hour to reduce API calls
- **Cost**: Review Bedrock pricing for your chosen embedding model

## Related Documentation

- [Configuration](configuration.md) - Full configuration reference
- [Vector Store Backends](vector-store-backends.md) - Custom vector store setup
- [Getting Started](getting-started.md) - Initial setup guide

