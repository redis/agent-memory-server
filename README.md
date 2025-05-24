# 🔮 Redis Agent Memory Server

A Redis-powered memory server built for AI agents and applications. It manages both conversational context and long-term memories, offering semantic search, automatic summarization, and flexible APIs through both REST and MCP interfaces.

## Features

- **Working Memory**
  - Session-scoped storage for messages, structured memories, context, and metadata
  - Automatically summarizes conversations when they exceed the window size
  - Client model-aware token limit management (adapts to the context window of the client's LLM)
  - Supports all major OpenAI and Anthropic models
  - Automatic promotion of structured memories to long-term storage

- **Long-Term Memory**
  - Persistent storage for memories across sessions
  - Semantic search to retrieve memories with advanced filtering system
  - Filter by session, namespace, topics, entities, timestamps, and more
  - Supports both exact match and semantic similarity search
  - Automatic topic modeling for stored memories with BERTopic or configured LLM
  - Automatic Entity Recognition using BERT
  - Memory deduplication and compaction

- **Other Features**
  - Namespace support for session and working memory isolation
  - Both a REST interface and MCP server
  - Background task processing for memory indexing and promotion
  - Unified search across working memory and long-term memory

## System Diagram
![System Diagram](diagram.png)

## Project Status and Roadmap
### Project Status: In Development, Pre-Release

This project is under active development and is **pre-release** software. Think of it as an early beta!

### Roadmap
- [x] Long-term memory deduplication and compaction
- [x] Use a background task system instead of `BackgroundTask`
- [ ] Configurable strategy for moving working memory to long-term memory
- [ ] Authentication/authorization hooks
- [ ] Separate Redis connections for long-term and working memory

## REST API Endpoints

The following endpoints are available:

- **GET /health**
  A simple health check endpoint returning the current server time.
  Example Response:
  ```json
  {"now": 1616173200}
  ```

- **GET /sessions/**
  Retrieves a paginated list of session IDs.
  _Query Parameters:_
  - `limit` (int): Number of sessions per page (default: 10)
  - `offset` (int): Number of sessions to skip (default: 0)
  - `namespace` (string, optional): Filter sessions by namespace.

- **GET /sessions/{session_id}/memory**
  Retrieves working memory for a session, including messages, structured memories,
  context, and metadata.
  _Query Parameters:_
  - `namespace` (string, optional): The namespace to use for the session
  - `window_size` (int, optional): Number of messages to include in the response (default from config)
  - `model_name` (string, optional): The client's LLM model name to determine appropriate context window size
  - `context_window_max` (int, optional): Direct specification of max context window tokens (overrides model_name)

- **PUT /sessions/{session_id}/memory**
  Sets working memory for a session, replacing any existing memory.
  Automatically summarizes conversations that exceed the window size.
  _Request Body Example:_
  ```json
  {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there"}
    ],
    "memories": [
      {
        "id": "mem-123",
        "text": "User prefers direct communication",
        "memory_type": "semantic"
      }
    ],
    "context": "Previous conversation summary...",
    "session_id": "session-123",
    "namespace": "default"
  }
  ```

- **DELETE /sessions/{session_id}/memory**
  Deletes all working memory (messages, context, structured memories, metadata) for a session.

- **POST /long-term-memory**
  Creates long-term memories directly, bypassing working memory.
  _Request Body Example:_
  ```json
  {
    "memories": [
      {
        "id": "mem-456",
        "text": "User is interested in AI and machine learning",
        "memory_type": "semantic",
        "session_id": "session-123",
        "namespace": "default"
      }
    ]
  }
  ```

- **POST /long-term-memory/search**
  Performs vector search on long-term memories with advanced filtering options.
  _Request Body Example:_
  ```json
  {
    "text": "Search query text",
    "limit": 10,
    "offset": 0,
    "session_id": {"eq": "session-123"},
    "namespace": {"eq": "default"},
    "topics": {"any": ["AI", "Machine Learning"]},
    "entities": {"all": ["OpenAI", "Claude"]},
    "created_at": {"gte": 1672527600, "lte": 1704063599},
    "last_accessed": {"gt": 1704063600},
    "user_id": {"eq": "user-456"}
  }
  ```

- **POST /memory-prompt**
  Generates prompts enriched with relevant memory context from both working
  memory and long-term memory. Useful for retrieving context before answering questions.
  _Request Body Example:_
  ```json
  {
    "query": "What did we discuss about AI?",
    "session": {
      "session_id": "session-123",
      "namespace": "default",
      "window_size": 10
    },
    "long_term_search": {
      "text": "AI discussion",
      "limit": 5,
      "namespace": {"eq": "default"}
    }
  }
  ```

  _Filter options for search endpoints:_
  - Tag filters (session_id, namespace, topics, entities, user_id):
    - `eq`: Equals this value
    - `ne`: Not equals this value
    - `any`: Contains any of these values
    - `all`: Contains all of these values

  - Numeric filters (created_at, last_accessed):
    - `gt`: Greater than
    - `lt`: Less than
    - `gte`: Greater than or equal
    - `lte`: Less than or equal
    - `eq`: Equals
    - `ne`: Not equals
    - `between`: Between two values

## MCP Server Interface
Agent Memory Server offers an MCP (Model Context Protocol) server interface powered by FastMCP, providing tool-based memory management for LLMs and agents:

- **set_working_memory**: Set working memory for a session (like PUT /sessions/{id}/memory API). Stores structured memory records and JSON data in working memory with automatic promotion to long-term storage.
- **create_long_term_memories**: Create long-term memories directly, bypassing working memory. Useful for bulk memory creation.
- **search_long_term_memory**: Perform semantic search across long-term memories with advanced filtering options.
- **memory_prompt**: Generate prompts enriched with session context and long-term memories. Essential for retrieving relevant context before answering questions.

## Command Line Interface

The `agent-memory-server` provides a command-line interface (CLI) for managing the server and related tasks. You can access the CLI using the `agent-memory` command (assuming the package is installed in a way that makes the script available in your PATH, e.g., via `pip install ...`).

### Available Commands

Here's a list of available commands and their functions:

#### `version`
Displays the current version of `agent-memory-server`.
```bash
agent-memory version
```

#### `api`
Starts the REST API server.
```bash
agent-memory api [OPTIONS]
```
**Options:**
*   `--port INTEGER`: Port to run the server on. (Default: value from `settings.port`, usually 8000)
*   `--host TEXT`: Host to run the server on. (Default: "0.0.0.0")
*   `--reload`: Enable auto-reload for development.

Example:
```bash
agent-memory api --port 8080 --reload
```

#### `mcp`
Starts the Model Context Protocol (MCP) server.
```bash
agent-memory mcp [OPTIONS]
```
**Options:**
*   `--port INTEGER`: Port to run the MCP server on. (Default: value from `settings.mcp_port`, usually 9000)
*   `--sse`: Run the MCP server in Server-Sent Events (SSE) mode. If not provided, it runs in stdio mode.

Example (SSE mode):
```bash
agent-memory mcp --port 9001 --sse
```
Example (stdio mode):
```bash
agent-memory mcp --port 9001
```

#### `schedule-task`
Schedules a background task to be processed by a Docket worker.
```bash
agent-memory schedule-task <TASK_PATH> [OPTIONS]
```
**Arguments:**
*   `TASK_PATH`: The Python import path to the task function. For example: `"agent_memory_server.long_term_memory.compact_long_term_memories"`

**Options:**
*   `--args TEXT` / `-a TEXT`: Arguments to pass to the task in `key=value` format. Can be specified multiple times. Values are automatically converted to boolean, integer, or float if possible, otherwise they remain strings.

Example:
```bash
agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories" -a limit=500 -a namespace=my_namespace -a compact_semantic_duplicates=false
```

#### `task-worker`
Starts a Docket worker to process background tasks from the queue. This worker uses the Docket name configured in settings.
```bash
agent-memory task-worker [OPTIONS]
```
**Options:**
*   `--concurrency INTEGER`: Number of tasks to process concurrently. (Default: 10)
*   `--redelivery-timeout INTEGER`: Seconds to wait before a task is redelivered to another worker if the current worker fails or times out. (Default: 30)

Example:
```bash
agent-memory task-worker --concurrency 5 --redelivery-timeout 60
```

#### `rebuild_index`
Rebuilds the search index for Redis Memory Server.
```bash
agent-memory rebuild_index
```

#### `migrate_memories`
Runs data migrations. Migrations are reentrant.
```bash
agent-memory migrate_memories
```

To see help for any command, you can use `--help`:
```bash
agent-memory --help
agent-memory api --help
agent-memory mcp --help
# etc.
```

## Getting Started

### Installation

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

2. Set up environment variables (see Configuration section)

### Running

The easiest way to start the REST and MCP servers is to use Docker Compose. See the Docker Compose section of this file for more details.

But you can also run these servers via the CLI commands. Here's how you
run the REST API server:
```bash
uv run agent-memory api
```

And the MCP server:
```
uv run agent-memory mcp --mode <stdio|sse>
```

**NOTE:** With uv, prefix the command with `uv`, e.g.: `uv run agent-memory --mode sse`. If you installed from source, you'll probably need to add `--directory` to tell uv where to find the code: `uv run --directory <path/to/checkout> run agent-memory --mode stdio`.

### Docker Compose

To start the API using Docker Compose, follow these steps:

1. Ensure that Docker and Docker Compose are installed on your system.

2. Open a terminal in the project root directory (where the docker-compose.yml file is located).

3. (Optional) Set up your environment variables (such as OPENAI_API_KEY and ANTHROPIC_API_KEY) either in a .env file or by modifying the docker-compose.yml as needed.

4. Build and start the containers by running:
   docker-compose up --build

5. Once the containers are up, the REST API will be available at http://localhost:8000. You can also access the interactive API documentation at http://localhost:8000/docs. The MCP server will be available at http://localhost:9000/sse.

6. To stop the containers, press Ctrl+C in the terminal and then run:
   docker-compose down

## Using the MCP Server with Claude Desktop, Cursor, etc.
You can use the MCP server that comes with this project in any application or SDK that supports MCP tools.

### Claude
<img src="claude.png">

For example, with Claude, use the following configuration:
  ```json
  {
    "mcpServers": {
      "redis-memory-server": {
          "command": "uv",
          "args": [
              "--directory",
              "/ABSOLUTE/PATH/TO/REPO/DIRECTORY/agent-memory-server",
              "run",
              "agent-memory",
              "-mcp",
              "--mode",
              "stdio"
          ]
      }
    }
  }
  ```
**NOTE:** On a Mac, this configuration requires that you use `brew install uv` to install uv. Probably any method that makes the `uv`
command globally accessible, so Claude can find it, would work.

### Cursor

<img src="cursor.png">

Cursor's MCP config is similar to Claude's, but it also supports SSE servers, so you can run the server in SSE mode and pass in the URL:

  ```json
  {
    "mcpServers": {
      "redis-memory-server": {
        "url": "http://localhost:9000/sse"
      }
    }
  }
  ```


## Configuration

You can configure the MCP and REST servers and task worker using environment
variables. See the file `config.py` for all the available settings.

The names of the settings map directly to an environment variable, so for
example, you can set the `openai_api_key` setting with the `OPENAI_API_KEY`
environment variable.

## Running the Background Task Worker

The Redis Memory Server uses Docket for background task management. There are two ways to run the worker:

### 1. Using the Docket CLI

After installing the package, you can run the worker using the Docket CLI command:

```bash
docket worker --tasks agent_memory_server.docket_tasks:task_collection --docket memory-server
```

You can customize the concurrency and redelivery timeout:

```bash
docket worker --tasks agent_memory_server.docket_tasks:task_collection --concurrency 5 --redelivery-timeout 60 --docket memory-server
```

**NOTE:** The name passed with `--docket` is effectively the name of a task queue where
the worker will look for work. This name should match the docket name your API server
is using, configured with the `docket_name` setting via environment variable
or directly in `agent_memory_server.config.Settings`.

## Memory Compaction

The memory compaction functionality optimizes storage by merging duplicate and semantically similar memories. This improves retrieval quality and reduces storage costs.

### Running Compaction

Memory compaction is available as a task function in `agent_memory_server.long_term_memory.compact_long_term_memories`. You can trigger it manually
by running the `agent-memory schedule-task` command:

```bash
agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories"
```

### Key Features

- **Hash-based Deduplication**: Identifies and merges exact duplicate memories using content hashing
- **Semantic Deduplication**: Finds and merges memories with similar meaning using vector search
- **LLM-powered Merging**: Uses language models to intelligently combine memories

## Running Migrations
When the data model changes, we add a migration in `migrations.py`. You can run
these to make sure your schema is up to date, like so:

```bash
uv run agent-memory migrate-memories
```

## Development

### Running Tests
```bash
uv run pytest
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
