# Command Line Interface

The `agent-memory-server` provides a command-line interface (CLI) for managing the server and related tasks. You can access the CLI using the `agent-memory` command (assuming the package is installed in a way that makes the script available in your PATH, e.g., via `pip install ...`).

## Available Commands

Here's a list of available commands and their functions:

### `version`

Displays the current version of `agent-memory-server`.

```bash
agent-memory version
```

### `api`

Starts the REST API server.

```bash
agent-memory api [OPTIONS]
```

**Options:**

- `--port INTEGER`: Port to run the server on. (Default: value from `settings.port`, usually 8000)
- `--host TEXT`: Host to run the server on. (Default: "0.0.0.0")
- `--reload`: Enable auto-reload for development.

Example:

```bash
agent-memory api --port 8080 --reload
```

### `mcp`

Starts the Model Context Protocol (MCP) server.

```bash
agent-memory mcp [OPTIONS]
```

**Options:**

- `--port INTEGER`: Port to run the MCP server on. (Default: value from `settings.mcp_port`, usually 9000)
- `--sse`: Run the MCP server in Server-Sent Events (SSE) mode. If not provided, it runs in stdio mode.

Example (SSE mode):

```bash
agent-memory mcp --port 9001 --sse
```

Example (stdio mode):

```bash
agent-memory mcp --port 9001
```

### `schedule-task`

Schedules a background task to be processed by a Docket worker.

```bash
agent-memory schedule-task <TASK_PATH> [OPTIONS]
```

**Arguments:**

- `TASK_PATH`: The Python import path to the task function. For example: `"agent_memory_server.long_term_memory.compact_long_term_memories"`

**Options:**

- `--args TEXT` / `-a TEXT`: Arguments to pass to the task in `key=value` format. Can be specified multiple times. Values are automatically converted to boolean, integer, or float if possible, otherwise they remain strings.

Example:

```bash
agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories" -a limit=500 -a namespace=my_namespace -a compact_semantic_duplicates=false
```

### `task-worker`

Starts a Docket worker to process background tasks from the queue. This worker uses the Docket name configured in settings.

```bash
agent-memory task-worker [OPTIONS]
```

**Options:**

- `--concurrency INTEGER`: Number of tasks to process concurrently. (Default: 10)
- `--redelivery-timeout INTEGER`: Seconds to wait before a task is redelivered to another worker if the current worker fails or times out. (Default: 30)

Example:

```bash
agent-memory task-worker --concurrency 5 --redelivery-timeout 60
```

### `rebuild_index`

Rebuilds the search index for Redis Memory Server.

```bash
agent-memory rebuild_index
```

### `migrate_memories`

Runs data migrations. Migrations are reentrant.

```bash
agent-memory migrate_memories
```

## Getting Help

To see help for any command, you can use `--help`:

```bash
agent-memory --help
agent-memory api --help
agent-memory mcp --help
# etc.
```
