# Redis Agent Memory Server

A memory layer for AI agents using Redis as the vector database.

## Features

- **Dual Interface**: REST API and Model Context Protocol (MCP) server
- **Two-Tier Memory**: Working memory (session-scoped) and long-term memory (persistent)
- **Semantic Search**: Vector-based similarity search with metadata filtering
- **Flexible Backends**: Pluggable vector store factory system
- **AI Integration**: Automatic topic extraction, entity recognition, and conversation summarization
- **Python SDK**: Easy integration with AI applications

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install uv
uv install --all-extras

# Start Redis
docker-compose up redis

# Start the server
uv run agent-memory api
```

### 2. Python SDK

```bash
# Install the client
pip install agent-memory-client
```

```python
from agent_memory_client import MemoryAPIClient

# Connect to server
client = MemoryAPIClient(base_url="http://localhost:8000")

# Store memories
await client.create_long_term_memories([
    {
        "text": "User prefers morning meetings",
        "user_id": "user123",
        "memory_type": "preference"
    }
])

# Search memories
results = await client.search_long_term_memory(
    text="What time does the user like meetings?",
    user_id="user123"
)
```

### 3. MCP Integration

```bash
# Start MCP server
uv run agent-memory mcp

# Or with SSE mode
uv run agent-memory mcp --mode sse --port 9000
```

## Documentation

📚 **[Full Documentation](https://redis.github.io/agent-memory-server/)** - Complete guides, API reference, and examples

### Key Documentation Sections:

- **[Quick Start Guide](docs/quick-start.md)** - Get up and running in minutes
- **[Python SDK](docs/python-sdk.md)** - Complete SDK reference with examples
- **[Vector Store Backends](docs/vector-store-backends.md)** - Configure different vector databases
- **[Authentication](docs/authentication.md)** - OAuth2/JWT setup for production
- **[Memory Types](docs/memory-types.md)** - Understanding semantic vs episodic memory
- **[API Reference](docs/api.md)** - REST API endpoints
- **[MCP Protocol](docs/mcp.md)** - Model Context Protocol integration

## Architecture

```
Working Memory (Session-scoped)  →  Long-term Memory (Persistent)
    ↓                                      ↓
- Messages                          - Semantic search
- Context                          - Topic modeling
- Structured memories              - Entity recognition
- Metadata                         - Deduplication
```

## Use Cases

- **AI Assistants**: Persistent memory across conversations
- **Customer Support**: Context from previous interactions
- **Personal AI**: Learning user preferences and history
- **Research Assistants**: Accumulating knowledge over time
- **Chatbots**: Maintaining context and personalization

## Development

```bash
# Install dependencies
uv install --all-extras

# Run tests
uv run pytest

# Format code
uv run ruff format
uv run ruff check

# Start development stack
docker-compose up
```

## Production Deployment

- **Authentication**: OAuth2/JWT with multiple providers (Auth0, AWS Cognito, etc.)
- **Redis**: Requires Redis with RediSearch module (RedisStack recommended)
- **Scaling**: Supports Redis clustering and background task processing
- **Monitoring**: Structured logging and health checks included

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see the [development documentation](docs/development.md) for guidelines.
