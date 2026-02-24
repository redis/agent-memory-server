# Operations Guide

Configure, secure, and deploy your Redis Agent Memory Server in production. This guide covers server configuration, authentication, LLM providers, and infrastructure setup.

## Server Configuration

<div class="grid cards" markdown>

-   ⚙️ **Configuration**

    ---

    Environment variables, YAML config files, and all server settings

    [Configuration Reference →](configuration.md)

-   🔐 **Authentication**

    ---

    OAuth2/JWT, token-based auth, and multi-provider setup

    [Authentication Guide →](authentication.md)

-   🛡️ **Security**

    ---

    Security considerations for custom prompts and production deployments

    [Security Guide →](security-custom-prompts.md)

</div>

## AI Provider Setup

<div class="grid cards" markdown>

-   🤖 **LLM Providers**

    ---

    Configure OpenAI, Anthropic, AWS Bedrock, Ollama, and 100+ providers via LiteLLM

    [LLM Providers →](llm-providers.md)

-   📐 **Embedding Providers**

    ---

    Set up embedding models for semantic search

    [Embedding Providers →](embedding-providers.md)

-   🗄️ **Vector Store Backends**

    ---

    Configure Redis, Pinecone, Chroma, or other vector stores

    [Vector Store Backends →](vector-store-backends.md)

</div>

## Quick Reference

| Topic | Description |
|-------|-------------|
| [Configuration](configuration.md) | All environment variables and YAML settings |
| [Authentication](authentication.md) | OAuth2, token auth, and development mode |
| [Security](security-custom-prompts.md) | Custom prompt security and best practices |
| [LLM Providers](llm-providers.md) | Generation models including AWS Bedrock |
| [Embedding Providers](embedding-providers.md) | Embedding models and dimensions |
| [Vector Store Backends](vector-store-backends.md) | Storage backend configuration |

## Where to Start

**Deploying to production?** Start with [Configuration](configuration.md) to understand all server settings.

**Setting up authentication?** See [Authentication](authentication.md) for OAuth2 or token-based auth.

**Using AWS?** The [LLM Providers](llm-providers.md) guide covers AWS Bedrock setup for both generation and embedding models.

**Customizing storage?** Check [Vector Store Backends](vector-store-backends.md) for Redis, Pinecone, and other options.
