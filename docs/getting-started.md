# Getting Started

## Installation

First, you'll need to download this repository. After you've downloaded it, you can install and run the servers.

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Install uv:

```bash
pip install uv
```

2. Install the package and required dependencies:

```bash
uv sync
```

3. Set up environment variables (see [Configuration](configuration.md) section)

## Running

The easiest way to start the worker, REST API server, and MCP server is to use Docker Compose. See the Docker Compose section below for more details.

But you can also run these components via the CLI commands. Here's how you
run the REST API server:

```bash
# Development mode (no separate worker needed)
uv run agent-memory api --no-worker

# Production mode (requires separate worker process)
uv run agent-memory api
```

Or the MCP server:

```bash
# Stdio mode (recommended for Claude Desktop)
uv run agent-memory mcp

# SSE mode for development
uv run agent-memory mcp --mode sse --no-worker

# SSE mode for production
uv run agent-memory mcp --mode sse
```

### Using uvx in MCP clients

When configuring MCP-enabled apps (e.g., Claude Desktop), prefer `uvx` so the app can run the server without a local checkout:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["--from", "agent-memory-server", "agent-memory", "mcp"],
      "env": {
        "DISABLE_AUTH": "true",
        "REDIS_URL": "redis://localhost:6379",
        "OPENAI_API_KEY": "<your-openai-key>"
      }
    }
  }
}
```

Notes:
- API keys: Default models use OpenAI. Set `OPENAI_API_KEY`. To use Anthropic instead, set `ANTHROPIC_API_KEY` and also `GENERATION_MODEL` to an Anthropic model (e.g., `claude-3-5-haiku-20241022`).
- Make sure your MCP host can find `uvx` (on its PATH or by using an absolute command path). macOS: `brew install uv`. If not on PATH, set `"command"` to an absolute path (e.g., `/opt/homebrew/bin/uvx` on Apple Silicon, `/usr/local/bin/uvx` on Intel macOS).
- For production, remove `DISABLE_AUTH` and configure auth.


**For production deployments**, you'll need to run a separate worker process:

```bash
uv run agent-memory task-worker
```

**For development**, use the `--no-worker` flag to run tasks inline without needing a separate worker process.

**NOTE:** With uv, prefix the command with `uv`, e.g.: `uv run agent-memory --mode sse`. If you installed from source, you'll probably need to add `--directory` to tell uv where to find the code: `uv run --directory <path/to/checkout> run agent-memory --mode stdio`.

## Docker Compose

To start the API using Docker Compose, follow these steps:

1. Ensure that Docker and Docker Compose are installed on your system.

2. Open a terminal in the project root directory (where the docker-compose.yml file is located).

3. (Optional) Set up your environment variables (such as OPENAI_API_KEY and ANTHROPIC_API_KEY) either in a .env file or by modifying the docker-compose.yml as needed.

4. Build and start the containers by running:
   ```bash
   docker-compose up --build
   ```

5. Once the containers are up, the REST API will be available at http://localhost:8000. You can also access the interactive API documentation at http://localhost:8000/docs. The MCP server will be available at http://localhost:9000/sse.

6. To stop the containers, press Ctrl+C in the terminal and then run:
   ```bash
   docker-compose down
   ```
