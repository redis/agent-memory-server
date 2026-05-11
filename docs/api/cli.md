# Command Line Interface

The `agent-memory-server` package provides a command-line interface (CLI) called `agent-memory` for running the REST API and MCP servers, scheduling and processing background tasks, running migrations, and managing authentication tokens.

## Installation

The `agent-memory` command is included when you install the server package.

```bash
pip install agent-memory-server
```

Verify the installation by running:

```bash
agent-memory version
```

## Getting help

All commands and command groups support the `--help` flag.

| Flag | Description |
|---|---|
| `--help` | Display usage information for the command or command group |

**Examples**

```bash
agent-memory --help            # Top-level help
agent-memory api --help        # Help for a single command
agent-memory token --help      # Help for a command group
```

## `version`

Display the installed version of `agent-memory-server`.

**Syntax**

```bash
agent-memory version
```

---

## `api`

Start the REST API server.

**Syntax**

```bash
agent-memory api [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--port` | integer | `settings.port` (usually `8000`) | Port to run the server on |
| `--host` | string | `0.0.0.0` | Host to bind the server to |
| `--reload` | flag | disabled | Enable auto-reload for development |
| `--task-backend` | `asyncio` \| `docket` | `docket` | Background task backend. `docket` requires a running `agent-memory task-worker`; `asyncio` runs tasks inline in the API process. |
| `--no-worker` *(deprecated)* | flag | — | Backwards-compatible alias for `--task-backend=asyncio`. Prefer `--task-backend`. |

**Examples**

```bash
# Development mode (no separate worker needed)
agent-memory api --port 8080 --reload --task-backend asyncio

# Production mode (default Docket backend; requires a separate worker)
agent-memory api --port 8080
```

!!! warning "Limitations of `--task-backend=asyncio`"

    The `asyncio` backend is suitable for development and simple use cases, but has important limitations:

    - **Periodic tasks don't run**: Scheduled maintenance tasks like memory compaction, periodic forgetting, and summary view refresh only execute when a Docket worker is running. These tasks use Docket's `Perpetual` scheduler.
    - **No task persistence**: If the server restarts, pending background tasks are lost.
    - **No distributed processing**: All tasks run in the API process; you cannot scale workers independently.

    For production deployments, use the default `docket` backend with a separate `agent-memory task-worker` process.

---

## `mcp`

Start the Model Context Protocol (MCP) server.

**Syntax**

```bash
agent-memory mcp [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--port` | integer | `settings.mcp_port` (usually `9000`) | Port to run the MCP server on |
| `--mode` | `stdio` \| `sse` \| `streamable-http` | `stdio` | MCP transport mode |
| `--task-backend` | `asyncio` \| `docket` | `asyncio` | Background task backend. `docket` requires a running `agent-memory task-worker`. |

**Examples**

```bash
# Stdio mode (recommended for Claude Desktop)
agent-memory mcp

# SSE mode for development (no separate worker needed)
agent-memory mcp --mode sse --port 9001

# Streamable HTTP mode for network deployments (e.g. Kubernetes)
agent-memory mcp --mode streamable-http --port 9000

# SSE mode for production (requires a separate worker process)
agent-memory mcp --mode sse --port 9001 --task-backend docket
```

!!! note "Choosing a mode"

    Stdio mode is designed for tools like Claude Desktop and uses the asyncio backend by default (no worker required). Streamable HTTP mode is suited for deploying the MCP server as a network service where HTTP clients (like Claude Code) connect over the network. Use `--task-backend docket` if you want MCP to enqueue background work into a shared Docket worker.

---

## `task-worker`

Start a Docket worker to process background tasks from the queue. The worker uses the Docket name configured in settings.

**Syntax**

```bash
agent-memory task-worker [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--concurrency` | integer | `10` | Number of tasks to process concurrently |
| `--redelivery-timeout` | integer | `2 × llm_task_timeout_minutes` (`600` seconds with default config) | Seconds to wait before a task is redelivered to another worker if the current worker fails or times out |

**Example**

```bash
agent-memory task-worker --concurrency 5 --redelivery-timeout 600
```

---

## `schedule-task`

Schedule a background task to be processed by a Docket worker.

**Syntax**

```bash
agent-memory schedule-task <TASK_PATH> [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `TASK_PATH` | Python import path to the task function (e.g. `agent_memory_server.long_term_memory.compact_long_term_memories`) |

**Options**

| Option | Description |
|---|---|
| `-a`, `--args TEXT` | Arguments to pass to the task in `key=value` format. Repeatable. Values are auto-converted to bool, integer, or float when possible; otherwise they remain strings. |

**Example**

```bash
agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories" \
  -a limit=500 \
  -a namespace=my_namespace \
  -a compact_semantic_duplicates=false
```

---

## `rebuild_index`

Rebuild the Redis search index.

**Syntax**

```bash
agent-memory rebuild_index
```

---

## `migrate-memories`

Run the built-in long-term memory migrations. Safe to rerun. Use this after upgrading if you need to backfill fields on existing memory records.

**Syntax**

```bash
agent-memory migrate-memories
```

---

## `migrate-working-memory`

Migrate legacy `working_memory:*` string keys to Redis JSON. Use this if you are upgrading from an older working-memory format or if you want to remove deprecated legacy `sessions` / `sessions:*` sorted sets from the old session-listing path.

**Syntax**

```bash
agent-memory migrate-working-memory [OPTIONS]
```

**Options**

| Option | Description |
|---|---|
| `--dry-run` | Report what would be migrated without making any changes |

**Examples**

```bash
# Inspect what would change
agent-memory migrate-working-memory --dry-run

# Run the migration
agent-memory migrate-working-memory
```

---

## `token`

Manage authentication tokens for token-based authentication. This command group provides subcommands for creating, listing, viewing, and removing API tokens.

**Syntax**

```bash
agent-memory token <subcommand> [OPTIONS]
```

**Subcommands**

| Subcommand | Description |
|---|---|
| [`add`](#token-add) | Create a new authentication token |
| [`list`](#token-list) | List all authentication tokens |
| [`show`](#token-show) | Show detailed information about a specific token |
| [`remove`](#token-remove) | Remove an authentication token |

**Security features**

- All tokens are hashed using bcrypt before storage
- Tokens automatically expire based on Redis TTL if an expiration is set
- The server never stores plaintext tokens
- Partial hash matching is supported for CLI convenience

### `token add`

Create a new authentication token.

**Syntax**

```bash
agent-memory token add --description <text> [OPTIONS]
```

**Required options**

| Option | Description |
|---|---|
| `-d`, `--description TEXT` | Description for the token (e.g. "API access for service X") |

**Optional options**

| Option | Default | Description |
|---|---|---|
| `-e`, `--expires-days INTEGER` | never expires | Number of days until the token expires |
| `--format` `text` \| `json` | `text` | Output format. `json` is recommended for CI and scripting. |
| `--token TEXT` | auto-generated | Use a pre-generated token value instead of letting the CLI generate one. The CLI hashes and stores the provided token; store the plaintext securely in your secrets manager or CI system. |

**Examples**

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

!!! warning "Token shown only once"

    The generated token is displayed only once. Store it securely; it cannot be retrieved again.

### `token list`

List all authentication tokens, showing masked token hashes, descriptions, and expiration dates.

**Syntax**

```bash
agent-memory token list [OPTIONS]
```

**Options**

| Option | Default | Description |
|---|---|---|
| `--format` `text` \| `json` | `text` | Output format. `json` returns a machine-readable array suitable for CI pipelines. |

**Text output example**

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

**JSON output example**

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

### `token show`

Show detailed information about a specific token. Supports partial hash matching for convenience.

**Syntax**

```bash
agent-memory token show <TOKEN_HASH> [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `TOKEN_HASH` | The token hash (or partial hash) to display. Can be the full hash or just the first few characters; partial hashes must be unique. |

**Options**

| Option | Default | Description |
|---|---|---|
| `--format` `text` \| `json` | `text` | Output format. `json` returns a machine-readable object including status. |

**Examples**

```bash
# Show token details using full hash
agent-memory token show abc12345def67890xyz

# Show token details using partial hash (must be unique)
agent-memory token show abc123
```

**JSON output example**

```json
{
  "hash": "abc12345def67890xyz",
  "description": "API access token",
  "created_at": "2025-07-10T18:30:00.000000+00:00",
  "expires_at": "2025-08-09T18:30:00.000000+00:00",
  "status": "Active"
}
```

### `token remove`

Remove an authentication token. By default, asks for confirmation before removal.

**Syntax**

```bash
agent-memory token remove <TOKEN_HASH> [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `TOKEN_HASH` | The token hash (or partial hash) to remove. Can be the full hash or just the first few characters. |

**Options**

| Option | Description |
|---|---|
| `-f`, `--force` | Remove the token without asking for confirmation |

**Examples**

```bash
# Remove token with confirmation prompt
agent-memory token remove abc123

# Remove token without confirmation
agent-memory token remove abc123 --force
```
