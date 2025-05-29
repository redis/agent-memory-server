# Configuration

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
