# Redis Agent Memory Server

**Give your AI agents persistent memory and context that gets smarter over time.**

Transform your AI agents from goldfish 🐠 into elephants 🐘 with Redis-powered memory that automatically learns, organizes, and recalls information across conversations and sessions.

<div class="grid cards" markdown>

-   :rocket:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes with our step-by-step guide

    [:octicons-arrow-right-24: Quick Start Guide](quick-start.md)

-   :brain:{ .lg .middle } **Use Cases**

    ---

    See real-world examples across industries and applications

    [:octicons-arrow-right-24: Explore Use Cases](use-cases.md)

-   :material-sdk:{ .lg .middle } **Python SDK**

    ---

    Easy integration with tool abstractions for OpenAI and Anthropic

    [:octicons-arrow-right-24: SDK Documentation](python-sdk.md)

-   :sparkles:{ .lg .middle } **New Features**

    ---

    Advanced features in v0.10.0: query optimization, memory editing, and more

    [:octicons-arrow-right-24: Advanced Features](query-optimization.md)

</div>

## What is Redis Agent Memory Server?

Redis Agent Memory Server is a production-ready memory system for AI agents and applications that:

- **:brain: Remembers everything**: Stores conversation history, user preferences, and important facts across sessions
- **:mag: Finds relevant context**: Uses semantic search to surface the right information at the right time
- **:chart_with_upwards_trend: Gets smarter over time**: Automatically extracts, organizes, and deduplicates memories from interactions
- **:electric_plug: Works with any AI model**: REST API and MCP interfaces compatible with OpenAI, Anthropic, and others

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
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Store a user preference
await client.create_long_term_memories([{
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

### :brain: Two-Tier Memory System

!!! info "Working Memory (Session-scoped)"
    - Current conversation state and context
    - Automatic summarization when conversations get long
    - TTL-based expiration (1 hour default)

!!! success "Long-Term Memory (Persistent)"
    - User preferences, facts, and important information
    - Semantic search with vector embeddings
    - Advanced filtering by time, topics, entities, users

### :mag: Intelligent Search
- **Semantic similarity**: Find memories by meaning, not just keywords
- **Advanced filters**: Search by user, session, time, topics, entities
- **Query optimization**: AI-powered query refinement for better results
- **Recency boost**: Time-aware ranking that surfaces relevant recent information

### :sparkles: Smart Memory Management
- **Automatic extraction**: Pull important facts from conversations
- **Contextual grounding**: Resolve pronouns and references ("he" → "John")
- **Deduplication**: Prevent duplicate memories with content hashing
- **Memory editing**: Update, correct, or enrich existing memories

### :rocket: Production Ready
- **Multiple interfaces**: REST API, MCP server, Python client
- **Authentication**: OAuth2/JWT, token-based, or disabled for development
- **Scalable storage**: Redis (default), Pinecone, Chroma, PostgreSQL, and more
- **Background processing**: Async tasks for heavy operations
- **Multi-tenancy**: User and namespace isolation

## Get Started

Ready to give your AI agents perfect memory?

<div class="grid" markdown>

<div markdown>
**New to memory systems?**

Start with our quick tutorial to understand the basics and see immediate results.

[Quick Start Guide :material-rocket-launch:](quick-start.md){ .md-button .md-button--primary }
</div>

<div markdown>
**Ready to integrate?**

Jump into the API documentation and start building with REST or MCP interfaces.

[API Documentation :material-api:](api.md){ .md-button }
</div>

</div>

---

## What's New in v0.10.0

<div class="grid cards" markdown>

-   :brain:{ .lg .middle } **Query Optimization**

    ---

    AI-powered query refinement with configurable models for better search accuracy

    [:octicons-arrow-right-24: Learn More](query-optimization.md)

-   :link:{ .lg .middle } **Contextual Grounding**

    ---

    Resolve pronouns and references in extracted memories for clearer context

    [:octicons-arrow-right-24: Learn More](contextual-grounding.md)

-   :pencil2:{ .lg .middle } **Memory Editing**

    ---

    Update and correct existing memories through REST API and MCP tools

    [:octicons-arrow-right-24: Learn More](memory-editing.md)

-   :clock1:{ .lg .middle } **Recency Boost**

    ---

    Time-aware memory ranking that surfaces relevant recent information

    [:octicons-arrow-right-24: Learn More](recency-boost.md)

</div>

## Community & Support

- **:material-github: Source Code**: [GitHub Repository](https://github.com/redis/redis-memory-server)
- **:material-docker: Docker Images**: [Docker Hub](https://hub.docker.com/r/andrewbrookins510/agent-memory-server)
- **:material-bug: Issues**: [Report Issues](https://github.com/redis/redis-memory-server/issues)
- **:material-book-open: Examples**: [Complete Examples](examples/)

---

**Ready to transform your AI agents?** Start with the [Quick Start Guide](quick-start.md) and build smarter agents in minutes! :brain::sparkles:
