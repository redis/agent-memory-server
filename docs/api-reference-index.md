# API Reference

Complete reference documentation for all Redis Agent Memory Server interfaces and client SDKs.

## Server Interfaces

<div class="grid cards" markdown>

-   🌐 **REST API**

    ---

    HTTP endpoints for memory operations with complete examples

    [REST API Reference →](api.md)

-   🤖 **MCP Server**

    ---

    Model Context Protocol tools for AI agents (Claude Desktop, etc.)

    [MCP Reference →](mcp.md)

-   💻 **CLI Reference**

    ---

    Command-line interface for server management

    [CLI Reference →](cli.md)

</div>

## Client SDKs

<div class="grid cards" markdown>

-   🐍 **Python SDK**

    ---

    Async-first client with tool schemas for OpenAI and Anthropic

    [Python SDK →](python-sdk.md)

-   📘 **TypeScript SDK**

    ---

    Type-safe client for Node.js and browser applications

    [TypeScript SDK →](typescript-sdk.md)

-   ☕ **Java SDK**

    ---

    Java client for JVM applications

    [Java SDK →](java-sdk.md)

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

## Interactive API Docs

When running the server locally, visit `http://localhost:8000/docs` for interactive Swagger documentation where you can try endpoints directly.
