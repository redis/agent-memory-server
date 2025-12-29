# API Reference

Complete reference documentation for all Redis Agent Memory Server interfaces.

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

## Interface Comparison

| Interface | Best For | Authentication |
|-----------|----------|----------------|
| REST API | Applications, backends, custom integrations | OAuth2/JWT or token |
| MCP Server | Claude Desktop, MCP-compatible AI agents | Environment config |
| CLI | Server administration, development | Local access |

## Interactive API Docs

When running the server locally, visit `http://localhost:8000/docs` for interactive Swagger documentation where you can try endpoints directly.
