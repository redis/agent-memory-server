---
description: How-to guides for Agent Memory Server.
---

# How-To Guides

Task-oriented recipes. Each one assumes you already know the basics from
the numbered tutorials and answers a single "how do I..." question.

## Configure

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } **[Configuration](configuration.md)**

    ---

    Environment variables and runtime options.

-   :material-shield-key:{ .lg .middle } **[Authentication](authentication.md)**

    ---

    OAuth2/JWT setup with Auth0, Cognito, Okta, Azure AD.

-   :material-shield-lock:{ .lg .middle } **[Custom prompt security](security.md)**

    ---

    Custom prompts, injection protection, hardening.

</div>

## Connect providers

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **[LLM providers](llm_providers.md)**

    ---

    OpenAI, Anthropic, and other LiteLLM-supported chat models.

-   :material-vector-line:{ .lg .middle } **[Embedding providers](embedding_providers.md)**

    ---

    Pick and configure an embedding model for semantic search.

-   :material-aws:{ .lg .middle } **[AWS Bedrock](aws_bedrock.md)**

    ---

    Use Bedrock-hosted Claude, Titan, and Cohere models.

</div>

## Vector storage

<div class="grid cards" markdown>

-   :material-database-search:{ .lg .middle } **[Advanced vector DB](advanced_vector_db.md)**

    ---

    Tune index parameters, choose algorithms, manage migrations.

-   :material-database-edit:{ .lg .middle } **[Custom vector DB](custom_vector_db.md)**

    ---

    Plug in your own vector backend behind the memory layer.

-   :material-tune:{ .lg .middle } **[Query optimization](query_optimization.md)**

    ---

    Lower latency and improve recall on large memory stores.

-   :material-pencil:{ .lg .middle } **[Memory editing](memory_editing.md)**

    ---

    Update, correct, and enrich existing long-term memories via REST, MCP, or the Python client.

</div>

## Integrate

<div class="grid cards" markdown>

-   :material-puzzle:{ .lg .middle } **[Integration patterns](integration_patterns.md)**

    ---

    Recipes for plugging the server into common agent frameworks.

-   :material-toolbox:{ .lg .middle } **[Development](development.md)**

    ---

    Local dev loop, debugging, and contributing back.

</div>

## Operations Quick Reference

| Topic | Description |
|-------|-------------|
| [Configuration](configuration.md) | All environment variables and YAML settings |
| [Worker timeout tuning](configuration.md#worker-lease-and-task-timeout) | How to size task timeout vs redelivery timeout for Docket workers |
| [Authentication](authentication.md) | OAuth2, token auth, and development mode |
| [Custom prompt security](security.md) | Custom prompt security and best practices |
| [LLM Providers](llm_providers.md) | Generation models including AWS Bedrock |
| [Embedding Providers](embedding_providers.md) | Embedding models and dimensions |
| [Custom Vector DB](custom_vector_db.md) | Storage backend configuration |
| [Advanced Vector DB](advanced_vector_db.md) | Tune index parameters and algorithms |

## Where to Start

**Deploying to production?** Start with [Configuration](configuration.md) to understand all server settings.

**Setting up authentication?** See [Authentication](authentication.md) for OAuth2 or token-based auth.

**Using AWS?** The [LLM Providers](llm_providers.md) and [AWS Bedrock](aws_bedrock.md) guides cover Bedrock setup for both generation and embedding models.

**Customizing storage?** Check [Custom Vector DB](custom_vector_db.md) for Redis and custom backend options.
