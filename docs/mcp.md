# MCP Server Interface

Agent Memory Server offers an MCP (Model Context Protocol) server interface powered by FastMCP, providing tool-based memory management for LLMs and agents:

- **set_working_memory**: Set working memory for a session (like PUT /sessions/{id}/memory API). Stores structured memory records and JSON data in working memory with automatic promotion to long-term storage.
- **create_long_term_memories**: Create long-term memories directly, bypassing working memory. Useful for bulk memory creation.
- **search_long_term_memory**: Perform semantic search across long-term memories with advanced filtering options.
- **edit_long_term_memory**: Update existing long-term memories with new or corrected information. Allows partial updates to specific fields while preserving other data.
- **delete_long_term_memories**: Remove specific long-term memories by ID. Useful for cleaning up outdated or incorrect information.
- **get_long_term_memory**: Retrieve specific memories by ID for detailed inspection or verification before editing.
- **memory_prompt**: Generate prompts enriched with working memory session and long-term memories. Essential for retrieving relevant context before answering questions.

## Available MCP Tools

The MCP server provides the following tools that AI agents can use to manage memories:

### Memory Search and Retrieval

**search_long_term_memory**
- Search for memories using semantic similarity
- Supports advanced filtering by user, session, namespace, topics, entities, and timestamps
- Configurable query optimization and recency boost
- Returns ranked results with relevance scores

**get_long_term_memory**
- Retrieve specific memories by their unique ID
- Useful for inspecting memory details before editing
- Returns complete memory record with all metadata

**memory_prompt**
- Generate AI prompts enriched with relevant memory context
- Combines working memory and long-term memory search results
- Essential for providing context to AI agents before responses

### Memory Management

**create_long_term_memories**
- Create new persistent memories directly
- Bypasses working memory for bulk operations
- Supports all memory types (semantic, episodic, message)
- Automatic indexing and embedding generation

**edit_long_term_memory**
- Update existing memories with corrections or new information
- Supports partial updates (only change specific fields)
- Automatic re-indexing and embedding regeneration
- Preserves memory ID and creation timestamp

**delete_long_term_memories**
- Remove specific memories by ID
- Supports batch deletion of multiple memories
- Useful for cleanup and data management

### Working Memory

**set_working_memory**
- Manage session-specific conversation state
- Store messages, structured memories, and arbitrary data
- Automatic promotion of memories to long-term storage
- TTL-based expiration for session cleanup

## Using the MCP Server with Claude Desktop, Cursor, etc.

You can use the MCP server that comes with this project in any application or SDK that supports MCP tools.

### Claude

<img src="../claude.png">

For Claude, the easiest way is to use uvx (recommended):

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
- Make sure your MCP host can find `uvx` (on its PATH or by using an absolute command path).
  - macOS: `brew install uv`
  - If not on PATH, set `"command"` to an absolute path (e.g., `/opt/homebrew/bin/uvx` on Apple Silicon, `/usr/local/bin/uvx` on Intel macOS). On Linux, `~/.local/bin/uvx` is common. See https://docs.astral.sh/uv/getting-started/ for distro specifics
- Set `DISABLE_AUTH=false` in production and configure proper auth per the Authentication guide.

If you’re running from a local checkout instead of PyPI, you can use `uv run` with a directory:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/REPO/DIRECTORY/agent-memory-server",
        "run",
        "agent-memory",
        "mcp"
      ]
    }
  }
}
```

Alternative (Anthropic):

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["--from", "agent-memory-server", "agent-memory", "mcp"],
      "env": {
        "DISABLE_AUTH": "true",
        "REDIS_URL": "redis://localhost:6379",
        "ANTHROPIC_API_KEY": "<your-anthropic-key>",
        "GENERATION_MODEL": "claude-3-5-haiku-20241022"
      }
    }
  }
}
```

### Cursor

<img src="../cursor.png">

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
