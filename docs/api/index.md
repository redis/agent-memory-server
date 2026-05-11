---
description: API reference for Agent Memory Server.
---

# API Reference

Every interface for talking to the memory server: HTTP endpoints, the MCP
toolset for AI agents, the CLI for operators, and three client SDKs.
Pick the one that matches your stack — they all sit on top of the same
core memory operations.

## Server Interfaces

<div class="grid cards" markdown>

-   :material-web:{ .lg .middle } **[REST API](rest.md)**

    ---

    OpenAPI-described HTTP endpoints for working memory, long-term memory, and prompts.

-   :material-protocol:{ .lg .middle } **[MCP server](mcp.md)**

    ---

    Model Context Protocol tools for AI agent integration (Claude Desktop, etc.).

-   :material-console:{ .lg .middle } **[CLI](cli.md)**

    ---

    Command-line interface for running, indexing, migrating, and scheduling tasks.

</div>

## Client SDKs

<div class="grid cards" markdown>

-   :material-language-python:{ .lg .middle } **[Python SDK](python_sdk.md)**

    ---

    Async-first Python client with tool schemas for OpenAI and Anthropic.

-   :material-language-typescript:{ .lg .middle } **[TypeScript SDK](typescript_sdk.md)**

    ---

    Type-safe client for Node.js and browser applications.

-   :material-coffee:{ .lg .middle } **[Java SDK](java_sdk.md)**

    ---

    JVM client for Java and Kotlin applications.

</div>

## Source Reference

<div class="grid cards" markdown>

-   :material-package:{ .lg .middle } **[Server package](server/index.md)**

    ---

    Auto-generated `agent_memory_server` Python package reference, built from source via [mkdocstrings](https://mkdocstrings.github.io/).

</div>

## Interface Comparison

| Interface | Best For | Authentication |
|-----------|----------|----------------|
| REST API | Applications, backends, custom integrations | OAuth2/JWT or token |
| MCP Server | Claude Desktop, MCP-compatible AI agents | Environment config |
| CLI | Server administration, development | Local access |
| Python SDK | Python applications with LLM tool integration | Token or OAuth2 |
| TypeScript SDK | Node.js, browser, and TypeScript applications | Token or OAuth2 |
| Java SDK | JVM-based applications | Token or OAuth2 |

## Feature Cross-Reference

| Feature | REST API | MCP Server | CLI | Documentation |
|---------|----------|------------|-----|---------------|
| **Memory Search** (semantic, keyword, hybrid) | ✅ `/v1/long-term-memory/search` | ✅ `search_long_term_memory` | ✅ `agent-memory search` | [REST](rest.md), [MCP](mcp.md), [CLI](cli.md) |
| **Memory Editing** | ✅ `PATCH /v1/long-term-memory/{id}` | ✅ `edit_long_term_memory` | ❌ | [Memory Editing](../user_guide/how_to_guides/memory_editing.md) |
| **Query Optimization** | ✅ `optimize_query` param | ✅ `optimize_query` param | ❌ | [Query Optimization](../user_guide/how_to_guides/query_optimization.md) |
| **Recency Boost** | ✅ Default enabled | ✅ Available | ❌ | [Recency Boost](../concepts/recency_boost.md) |
| **Authentication** | ✅ JWT/Token | ✅ Inherited | ✅ Token management | [Authentication](../user_guide/how_to_guides/authentication.md) |
| **Background Tasks** | ✅ Automatic | ✅ Automatic | ✅ Worker management | [Configuration](../user_guide/how_to_guides/configuration.md) |

## By Interface Preference

**REST API users** → [REST API Documentation](rest.md) → [Authentication](../user_guide/how_to_guides/authentication.md)
**MCP/Claude users** → [MCP Server](mcp.md) → [Memory Editing](../user_guide/how_to_guides/memory_editing.md)
**CLI management** → [CLI Reference](cli.md) → [Configuration](../user_guide/how_to_guides/configuration.md)

!!! tip "Interactive API Docs"
    When running the server locally, visit `http://localhost:8000/docs` for
    interactive Swagger documentation where you can try endpoints directly.
