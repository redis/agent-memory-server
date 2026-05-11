---
description: Agent Memory Server documentation. Session Memory and Long-Term Memory for AI Agents.
---

<div class="rds-hero" markdown>

![Redis](assets/redis-logo-script-red.svg){ .rds-hero__logo }

# Agent Memory Server

Session Memory and Long-Term Memory for AI Agents
{: .rds-hero__tagline }

</div>

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](user_guide/01_quick_start.md)**

    ---

    Get a memory-enabled agent running in 5 minutes.

-   :material-target:{ .lg .middle } **[Use Cases](examples/use_cases.md)**

    ---

    See what you can build: support bots, tutors, personal assistants.

-   :material-language-python:{ .lg .middle } **[Python SDK](api/python_sdk.md)**

    ---

    Async-first client with OpenAI and Anthropic tool integration.

</div>

---

## Explore the Docs

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **[Concepts](concepts/index.md)**

    ---

    The memory model. Working memory, long-term memory, lifecycle, extraction, summarization.

-   :material-rocket-launch:{ .lg .middle } **[User Guide](user_guide/index.md)**

    ---

    Tutorials and how-to recipes. Installation, auth, providers, vector DB, integration patterns.

-   :material-lightbulb-on:{ .lg .middle } **[Examples](examples/index.md)**

    ---

    Runnable agent patterns: travel agent, AI tutor, memory editor, LangChain integration.

-   :material-api:{ .lg .middle } **[API Reference](api/index.md)**

    ---

    REST, MCP, CLI, SDKs, plus the full `agent_memory_server` Python package.

</div>

## What is Agent Memory Server?

Agent Memory Server is a production-ready memory system for AI agents and applications that:

- **🧠 Remembers everything**: Stores conversation history, user preferences, and important facts across sessions
- **🔍 Finds relevant context**: Uses semantic, keyword, and hybrid search to surface the right information at the right time
- **📈 Gets smarter over time**: Automatically extracts, organizes, and deduplicates memories from interactions
- **🔌 Works with any AI model**: REST API and MCP interfaces compatible with OpenAI, Anthropic, and others
- **🌐 Multi-provider support**: Use [100+ LLM providers](user_guide/how_to_guides/llm_providers.md) via LiteLLM (OpenAI, Anthropic, AWS Bedrock, Ollama, Azure, Gemini, and more)

## Why Use It?

=== "For AI Applications"

    - Never lose conversation context across sessions
    - Provide personalized responses based on user history
    - Build agents that learn and improve from interactions
    - Scale from prototypes to production with authentication and multi-tenancy

=== "For Developers"

    - Drop-in memory solution with REST API and MCP support
    - Works with existing AI frameworks and models
    - Production-ready with authentication, background processing, and vector storage
    - Extensively documented with examples and tutorials

## Quick Example

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

client = MemoryAPIClient(MemoryClientConfig(base_url="http://localhost:8000"))

# Store a user preference
await client.create_long_term_memory([{
    "text": "User prefers morning meetings and hates scheduling calls after 4 PM",
    "memory_type": "semantic",
    "topics": ["scheduling", "preferences"],
    "user_id": "alice"
}])

# Later, search for relevant context
results = await client.search_long_term_memory(
    text="when does user prefer meetings",
    limit=3
)

print(f"Found: {results.memories[0].text}")
# Output: "User prefers morning meetings and hates scheduling calls after 4 PM"
```

## Core Features

### 🧠 Two-Tier Memory System

!!! info "Working Memory (Session-scoped)"
    - Current conversation state and context
    - Automatic summarization when conversations get long
    - Durable by default, optional TTL expiration

!!! success "Long-Term Memory (Persistent)"
    - User preferences, facts, and important information
    - Flexible search: semantic (vector embeddings), keyword (full-text), and hybrid (combined)
    - Advanced filtering by time, topics, entities, users

### 🔍 Intelligent Search

- **Multiple search modes**: Semantic (vector similarity), keyword (full-text), and hybrid (combined) search
- **Advanced filters**: Search by user, session, time, topics, entities
- **Query optimization**: AI-powered query refinement for better results
- **Recency boost**: Time-aware ranking that surfaces relevant recent information

### ✨ Smart Memory Management

- **Automatic extraction**: Pull important facts from conversations
- **Contextual grounding**: Resolve pronouns and references ("he" → "John")
- **Deduplication**: Prevent duplicate memories with content hashing
- **Memory editing**: Update, correct, or enrich existing memories

### 🚀 Production Ready

- **Multiple interfaces**: REST API, MCP server, Python client
- **Authentication**: OAuth2/JWT, token-based, or disabled for development
- **Scalable storage**: Redis (default), Pinecone, Chroma, PostgreSQL, and more
- **Background processing**: Async tasks for heavy operations
- **Multi-tenancy**: User and namespace isolation

## Reader Paths

**👋 New to memory systems?** → [Quick Start](user_guide/01_quick_start.md) → [Use Cases](examples/use_cases.md) → [Long-Term Memory](concepts/long_term_memory.md)
**🔧 Ready to integrate?** → [Installation](user_guide/02_installation.md) → [REST API](api/rest.md) → [Configuration](user_guide/how_to_guides/configuration.md) → [Authentication](user_guide/how_to_guides/authentication.md)
**🤖 Building an AI agent?** → [MCP Server](api/mcp.md) → [Memory Lifecycle](concepts/memory_lifecycle.md) → [Query Optimization](user_guide/how_to_guides/query_optimization.md)

## Get Started

<div class="grid" markdown>

[:material-rocket-launch: New to memory systems?](user_guide/01_quick_start.md){ .md-button .md-button--primary }
[:material-api: Ready to integrate?](user_guide/index.md){ .md-button }

</div>

## For AI agents

If you are an AI agent reading these docs, start with
[`AGENTS.md`](https://github.com/redis/agent-memory-server/blob/main/AGENTS.md)
at the repo root for usage notes, or
[For AI Agents](for-ais-only/index.md) for an internal map of the source
tree. A flat [`llms.txt`](https://ai.redis.io/agent-memory/llms.txt)
index of every doc page is generated at build time.

---

## Community & Support

- **💻 Source Code**: [GitHub Repository](https://github.com/redis/agent-memory-server)
- **🐳 Docker Images**: [Docker Hub](https://hub.docker.com/r/redislabs/agent-memory-server)
- **🐛 Issues**: [Report Issues](https://github.com/redis/agent-memory-server/issues)
- **📖 Examples**: [Complete Examples](https://github.com/redis/agent-memory-server/tree/main/examples)
