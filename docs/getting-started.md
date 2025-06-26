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
uv run agent-memory api
```

Or the MCP server:

```bash
uv run agent-memory mcp --mode <stdio|sse>
```

Both servers require a worker to be running, which you can start like this:

```bash
uv run agent-memory task-worker
```

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
