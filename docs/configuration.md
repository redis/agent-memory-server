# Configuration

You can configure the MCP and REST servers and task worker using environment
variables. See the file `config.py` for all the available settings.

The names of the settings map directly to an environment variable, so for
example, you can set the `openai_api_key` setting with the `OPENAI_API_KEY`
environment variable.

## Running the Background Task Worker

The Redis Memory Server uses Docket for background task management. You can run a worker instance like this:

```bash
uv run agent-memory task-worker
```

You can customize the concurrency and redelivery timeout:

```bash
uv run agent-memory task-worker --concurrency 5 --redelivery-timeout 60
```

## Memory Compaction

The memory compaction functionality optimizes storage by merging duplicate and semantically similar memories. This improves retrieval quality and reduces storage costs.

### Running Compaction

Memory compaction is available as a task function in `agent_memory_server.long_term_memory.compact_long_term_memories`. You can trigger it manually
by running the `agent-memory schedule-task` command:

```bash
uv run agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories"
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

## Authentication Configuration

The Redis Memory Server supports multiple authentication modes configured via environment variables:

### Authentication Mode Settings

```bash
# Primary authentication mode setting
AUTH_MODE=disabled  # Options: disabled, token, oauth2 (default: disabled)

# Legacy setting (for backward compatibility)
DISABLE_AUTH=true   # Set to true to bypass all authentication

# Alternative token auth setting
TOKEN_AUTH_ENABLED=true  # Alternative way to enable token authentication
```

### OAuth2/JWT Settings

Required when `AUTH_MODE=oauth2`:

```bash
OAUTH2_ISSUER_URL=https://your-auth-provider.com
OAUTH2_AUDIENCE=your-api-audience
OAUTH2_JWKS_URL=https://your-auth-provider.com/.well-known/jwks.json  # Optional
OAUTH2_ALGORITHMS=["RS256"]  # Supported algorithms (default: ["RS256"])
```

### Token Authentication

When using `AUTH_MODE=token`:

- Tokens are managed via CLI commands (`agent-memory token`)
- No additional environment variables required
- Tokens are stored securely in Redis with bcrypt hashing
- Optional expiration dates supported

### Examples

**Development (No Authentication):**
```bash
export AUTH_MODE=disabled
# OR
export DISABLE_AUTH=true
```

**Production with Token Authentication:**
```bash
export AUTH_MODE=token
```

**Production with OAuth2:**
```bash
export AUTH_MODE=oauth2
export OAUTH2_ISSUER_URL=https://your-domain.auth0.com/
export OAUTH2_AUDIENCE=https://your-api.com
```

For detailed authentication setup and usage, see the [Authentication Documentation](authentication.md).
