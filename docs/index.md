# Redis Agent Memory Server

**Give your AI agents persistent memory and context that gets smarter over time.**

Transform your AI agents from goldfish 🐠 into elephants 🐘 with Redis-powered memory that automatically learns, organizes, and recalls information across conversations and sessions.

<div class="grid cards" markdown>

-   🚀 **Quick Start**

    ---

    Get up and running in 5 minutes with our step-by-step guide

    [Quick Start Guide →](quick-start.md)

-   🧠 **Use Cases**

    ---

    See real-world examples across industries and applications

    [Explore Use Cases →](use-cases.md)

-   🐍 **Python SDK**

    ---

    Easy integration with tool abstractions for OpenAI and Anthropic

    [SDK Documentation →](python-sdk.md)


</div>

## What is Redis Agent Memory Server?

Redis Agent Memory Server is a production-ready memory system for AI agents and applications that:

- **🧠 Remembers everything**: Stores conversation history, user preferences, and important facts across sessions
- **🔍 Finds relevant context**: Uses semantic, keyword, and hybrid search to surface the right information at the right time
- **📈 Gets smarter over time**: Automatically extracts, organizes, and deduplicates memories from interactions
- **🔌 Works with any AI model**: REST API and MCP interfaces compatible with OpenAI, Anthropic, and others
- **🌐 Multi-provider support**: Use [100+ LLM providers](llm-providers.md) via LiteLLM (OpenAI, Anthropic, AWS Bedrock, Ollama, Azure, Gemini, and more)

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

## Get Started

Ready to give your AI agents perfect memory?

<div class="grid" markdown>

<div markdown>
**New to memory systems?**

Start with our quick tutorial to understand the basics and see immediate results.

[🚀 Quick Start Guide](quick-start.md){ .md-button .md-button--primary }
</div>

<div markdown>
**Ready to integrate?**

Jump into the Developer Guide for memory patterns and integration strategies.

[🧠 Developer Guide](memory-integration-patterns.md){ .md-button }
</div>

</div>

---

## Community & Support

- **💻 Source Code**: [GitHub Repository](https://github.com/redis/agent-memory-server)
- **🐳 Docker Images**: [Docker Hub](https://hub.docker.com/r/redislabs/agent-memory-server)
- **🐛 Issues**: [Report Issues](https://github.com/redis/agent-memory-server/issues)
- **📖 Examples**: [Complete Examples](https://github.com/redis/agent-memory-server/tree/main/examples)

---

**Ready to transform your AI agents?** Start with the [Quick Start Guide](quick-start.md) and build smarter agents in minutes! 🧠✨
