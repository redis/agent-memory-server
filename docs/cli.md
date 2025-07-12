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

### `token` Commands

Manages authentication tokens for token-based authentication. The token command group provides subcommands for creating, listing, viewing, and removing API tokens.

#### `token add`

Creates a new authentication token.

```bash
agent-memory token add --description "DESCRIPTION" [--expires-days DAYS]
```

**Options:**

- `--description TEXT` / `-d TEXT`: **Required**. Description for the token (e.g., "API access for service X")
- `--expires-days INTEGER` / `-e INTEGER`: **Optional**. Number of days until token expires. If not specified, token never expires.

**Examples:**

```bash
# Create a token that expires in 30 days
agent-memory token add --description "API access token" --expires-days 30

# Create a permanent token (no expiration)
agent-memory token add --description "Service account token"
```

**Security Note:** The generated token is displayed only once. Store it securely as it cannot be retrieved again.

#### `token list`

Lists all authentication tokens, showing masked token hashes, descriptions, and expiration dates.

```bash
agent-memory token list
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
agent-memory token show TOKEN_HASH
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
