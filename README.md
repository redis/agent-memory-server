# Redis Agent Memory Server

A memory layer for AI agents using Redis as the vector database.

## Features

- **Dual Interface**: REST API and Model Context Protocol (MCP) server
- **Two-Tier Memory**: Working memory (session-scoped) and long-term memory (persistent)
- **Configurable Memory Strategies**: Customize how memories are extracted (discrete, summary, preferences, custom)
- **Semantic Search**: Vector-based similarity search with metadata filtering
- **Flexible Backends**: Pluggable vector store factory system
- **AI Integration**: Automatic topic extraction, entity recognition, and conversation summarization
- **Python SDK**: Easy integration with AI applications

## Quick Start

### 1. Installation

#### Using Docker

Pre-built Docker images are available from:
- **Docker Hub**: [redislabs/agent-memory-server](https://hub.docker.com/r/redislabs/agent-memory-server)
- **GitHub Packages**: [ghcr.io/redis/agent-memory-server](https://github.com/redis/agent-memory-server/pkgs/container/agent-memory-server)

**Quick Start (Development Mode)**:
```bash
# Start with docker-compose (includes Redis, API, MCP, and worker)
docker-compose up

# Or run just the API server (requires separate Redis)
docker run -p 8000:8000 \
  -e REDIS_URL=redis://your-redis:6379 \
  -e OPENAI_API_KEY=your-key \
  redislabs/agent-memory-server:latest
```

The default image runs in development mode (`--no-worker`), which is perfect for testing and development.

**Production Deployment**:

For production, run separate containers for the API and background workers:

```bash
# API Server (without background worker)
docker run -p 8000:8000 \
  -e REDIS_URL=redis://your-redis:6379 \
  -e OPENAI_API_KEY=your-key \
  -e DISABLE_AUTH=false \
  redislabs/agent-memory-server:latest \
  agent-memory api --host 0.0.0.0 --port 8000

# Background Worker (separate container)
docker run \
  -e REDIS_URL=redis://your-redis:6379 \
  -e OPENAI_API_KEY=your-key \
  redislabs/agent-memory-server:latest \
  agent-memory task-worker --concurrency 10

# MCP Server (if needed)
docker run -p 9000:9000 \
  -e REDIS_URL=redis://your-redis:6379 \
  -e OPENAI_API_KEY=your-key \
  redislabs/agent-memory-server:latest \
  agent-memory mcp --mode sse --port 9000
```

#### From Source

```bash
# Install dependencies
pip install uv
uv install --all-extras

# Start Redis
docker-compose up redis

# Start the server (development mode)
uv run agent-memory api --no-worker
```

### 2. Python SDK

Allowing the server to extract memories from working memory is easiest. However, you can also manually create memories:

```bash
# Install the client
pip install agent-memory-client

# For LangChain integration
pip install agent-memory-client langchain-core
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

> **Note**: While you can call client functions directly as shown above, using **MCP or SDK-provided tool calls** is recommended for AI agents as it provides better integration, automatic context management, and follows AI-native patterns. For the best performance, you can add messages to working memory and allow the server to extract memories in the background. See **[Memory Integration Patterns](https://redis.github.io/agent-memory-server/memory-integration-patterns/)** for guidance on when to use each approach.


#### LangChain Integration

For LangChain users, the SDK provides automatic conversion of memory client tools to LangChain-compatible tools, eliminating the need for manual wrapping with `@tool` decorators.

```python
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Get LangChain-compatible tools automatically
memory_client = await create_memory_client("http://localhost:8000")
tools = get_memory_tools(
    memory_client=memory_client,
    session_id="my_session",
    user_id="alice"
)

# Create prompt and agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with memory."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Use the agent
result = await executor.ainvoke({"input": "Remember that I love pizza"})
```

### 3. MCP Integration

```bash
# Start MCP server (stdio mode - recommended for Claude Desktop)
uv run agent-memory mcp

# Or with SSE mode (development mode)
uv run agent-memory mcp --mode sse --port 9000 --no-worker
```

## Documentation

📚 **[Full Documentation](https://redis.github.io/agent-memory-server/)** - Complete guides, API reference, and examples

### Key Documentation Sections:

- **[Quick Start Guide](https://redis.github.io/agent-memory-server/quick-start/)** - Get up and running in minutes
- **[Python SDK](https://redis.github.io/agent-memory-server/python-sdk/)** - Complete SDK reference with examples
- **[LangChain Integration](https://redis.github.io/agent-memory-server/langchain-integration/)** - Automatic tool conversion for LangChain
- **[Vector Store Backends](https://redis.github.io/agent-memory-server/vector-store-backends/)** - Configure different vector databases
- **[Authentication](https://redis.github.io/agent-memory-server/authentication/)** - OAuth2/JWT setup for production
- **[Memory Types](https://redis.github.io/agent-memory-server/memory-types/)** - Understanding semantic vs episodic memory
- **[API Reference](https://redis.github.io/agent-memory-server/api/)** - REST API endpoints
- **[MCP Protocol](https://redis.github.io/agent-memory-server/mcp/)** - Model Context Protocol integration

## Architecture

```
Working Memory (Session-scoped)  →  Long-term Memory (Persistent)
    ↓                                      ↓
- Messages                         - Semantic search
- Structured memories              - Topic modeling
- Summary of past messages         - Entity recognition
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
## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see the [development documentation](docs/development.md) for guidelines.
