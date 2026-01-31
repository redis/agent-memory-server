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
- `--task-backend [asyncio|docket]`: Background task backend. `docket` (default) uses Docket-based background workers (requires a running `agent-memory task-worker` for non-blocking tasks). `asyncio` runs tasks inline in the API process and does **not** require a separate worker.
- `--no-worker` (**deprecated**): Backwards-compatible alias for `--task-backend=asyncio`. Maintained for older scripts; prefer `--task-backend`.

**Examples:**

```bash
# Development mode (no separate worker needed, asyncio backend)
agent-memory api --port 8080 --reload --task-backend asyncio

# Production mode (default Docket backend; requires separate worker process)
agent-memory api --port 8080
```

!!! warning "Limitations of `--task-backend=asyncio`"
    The `asyncio` backend is suitable for development and simple use cases, but has important limitations:

    - **Periodic tasks don't run**: Scheduled maintenance tasks like memory compaction, periodic forgetting, and summary view refresh only execute when a Docket worker is running. These tasks use Docket's `Perpetual` scheduler.
    - **No task persistence**: If the server restarts, pending background tasks are lost.
    - **No distributed processing**: All tasks run in the API process; you cannot scale workers independently.

    For production deployments, use the default `docket` backend with a separate `agent-memory task-worker` process.

### `mcp`

Starts the Model Context Protocol (MCP) server.

```bash
agent-memory mcp [OPTIONS]
```

**Options:**

- `--port INTEGER`: Port to run the MCP server on. (Default: value from `settings.mcp_port`, usually 9000)
- `--mode [stdio|sse]`: Run the MCP server in stdio or SSE mode. (Default: stdio)
- `--task-backend [asyncio|docket]`: Background task backend. `asyncio` (default) runs tasks inline in the MCP process with no separate worker. `docket` sends tasks to a Docket queue, which requires running `agent-memory task-worker`.

**Examples:**

```bash
# Stdio mode (recommended for Claude Desktop) - default asyncio backend
agent-memory mcp

# SSE mode for development (no separate worker needed)
agent-memory mcp --mode sse --port 9001

# SSE mode for production (requires separate worker process)
agent-memory mcp --mode sse --port 9001 --task-backend docket
```

**Note:** Stdio mode is designed for tools like Claude Desktop and, by default, uses the asyncio backend (no worker). Use `--task-backend docket` if you want MCP to enqueue background work into a shared Docket worker.

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

### `token` Commands

Manages authentication tokens for token-based authentication. The token command group provides subcommands for creating, listing, viewing, and removing API tokens.

#### `token add`

Creates a new authentication token.

```bash
agent-memory token add --description "DESCRIPTION" [--expires-days DAYS] [--format text|json] [--token TOKEN_VALUE]
```

**Options:**

- `--description TEXT` / `-d TEXT`: **Required**. Description for the token (e.g., "API access for service X")
- `--expires-days INTEGER` / `-e INTEGER`: **Optional**. Number of days until token expires. If not specified, token never expires.
- `--format [text|json]`: **Optional**. Output format. `text` (default) is human-readable; `json` is machine-readable and recommended for CI or scripting.
- `--token TEXT`: **Optional**. Use a pre-generated token value instead of having the CLI generate one. The CLI will hash and store the provided token; make sure you've stored the plaintext securely in your secrets manager or CI system.

**Examples:**

```bash
# Create a token that expires in 30 days
agent-memory token add --description "API access token" --expires-days 30

# Create a permanent token (no expiration)
agent-memory token add --description "Service account token"

# Create a token and return JSON (for CI/scripts)
agent-memory token add --description "CI token" --expires-days 30 --format json

# Register a pre-generated token (e.g., from a secrets manager)
agent-memory token add --description "Terraform bootstrap" --token "$MY_TOKEN" --format json
```

**Security Note:** The generated token is displayed only once. Store it securely as it cannot be retrieved again.

#### `token list`

Lists all authentication tokens, showing masked token hashes, descriptions, and expiration dates.

```bash
agent-memory token list [--format text|json]
```

When `--format json` is used, the command prints a JSON array of token summaries suitable for scripting and CI pipelines. The default `text` format produces human-readable output like the example below.
**JSON Output Example:**
```json
[
  {
    "hash": "abc12345def67890xyz",
    "description": "API access token",
    "created_at": "2025-07-10T18:30:00.000000+00:00",
    "expires_at": "2025-08-09T18:30:00.000000+00:00",
    "status": "Active"
  },
  {
    "hash": "def09876uvw54321...",
    "description": "Service account token",
    "created_at": "2025-07-10T19:00:00.000000+00:00",
    "expires_at": null,
    "status": "Never Expires"
  }
]
```

**Example Output:**
```
Authentication Tokens:
==================================================
Token: abc12345...xyz67890
Description: API access token
Created: 2025-07-10T18:30:00.000000+00:00
Expires: 2025-08-09T18:30:00.000000+00:00
------------------------------
Token: def09876...uvw54321
Description: Service account token
Created: 2025-07-10T19:00:00.000000+00:00
Expires: Never
------------------------------
```

#### `token show`

Shows detailed information about a specific token. Supports partial hash matching for convenience.

```bash
agent-memory token show TOKEN_HASH [--format text|json]
```

When `--format json` is used, the command prints a JSON object with token details (including status) suitable for scripting and CI pipelines. The default `text` format produces human-readable output.
**JSON Output Example:**
```json
{
  "hash": "abc12345def67890xyz",
  "description": "API access token",
  "created_at": "2025-07-10T18:30:00.000000+00:00",
  "expires_at": "2025-08-09T18:30:00.000000+00:00",
  "status": "Active"
}
```

**Arguments:**

- `TOKEN_HASH`: The token hash (or partial hash) to display. Can be the full hash or just the first few characters.

**Examples:**

```bash
# Show token details using full hash
agent-memory token show abc12345def67890xyz

# Show token details using partial hash (must be unique)
agent-memory token show abc123
```

#### `token remove`

Removes an authentication token. By default, asks for confirmation before removal.

```bash
agent-memory token remove TOKEN_HASH [--force]
```

**Arguments:**

- `TOKEN_HASH`: The token hash (or partial hash) to remove. Can be the full hash or just the first few characters.

**Options:**

- `--force` / `-f`: Remove the token without asking for confirmation.

**Examples:**

```bash
# Remove token with confirmation prompt
agent-memory token remove abc123

# Remove token without confirmation
agent-memory token remove abc123 --force
```

**Security Features:**

- All tokens are hashed using bcrypt before storage
- Tokens automatically expire based on Redis TTL if expiration is set
- Server never stores plaintext tokens
- Partial hash matching for CLI convenience

## Getting Help

To see help for any command, you can use `--help`:

```bash
agent-memory --help
agent-memory api --help
agent-memory mcp --help
# etc.
```
