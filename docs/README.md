# Redis Agent Memory Server Documentation

Comprehensive documentation for building AI agents with persistent, intelligent memory.

## 🚀 Getting Started

**New to Redis Agent Memory Server?** Start here:

- **[Quick Start Guide](quick-start.md)** - Get up and running in 5 minutes
- **[Getting Started](getting-started.md)** - Complete installation and setup guide
- **[Use Cases](use-cases.md)** - Real-world examples and implementation patterns

## 🧠 Core Concepts

Understand the fundamentals:

- **[Memory Types](memory-types.md)** - Working vs Long-term memory explained with examples
- **[Authentication](authentication.md)** - OAuth2/JWT, token-based, and development setup
- **[Configuration](configuration.md)** - Environment variables, settings, and deployment options

## ✨ Advanced Features

**New in v0.10.0** - Powerful features for intelligent memory management:

- **[Query Optimization](query-optimization.md)** - AI-powered query refinement for better search accuracy
- **[Contextual Grounding](contextual-grounding.md)** - Resolve pronouns and references in extracted memories
- **[Memory Editing](memory-editing.md)** - Update, correct, and enrich existing memories
- **[Recency Boost](recency-boost.md)** - Time-aware memory ranking and intelligent scoring
- **[Vector Store Backends](vector-store-backends.md)** - Alternative storage backends (Pinecone, Chroma, etc.)

## 🔌 API Reference

Choose your integration approach:

- **[REST API](api.md)** - HTTP endpoints with complete examples and curl commands
- **[MCP Server](mcp.md)** - Model Context Protocol tools for AI agents (Claude, etc.)
- **[CLI](cli.md)** - Command-line interface for server management and administration

## 🛠️ Development

For contributors and advanced users:

- **[Development Guide](development.md)** - Local setup, testing, and contributing guidelines
- **[System Architecture](../diagram.png)** - Visual overview of system components

## 📚 Additional Resources

- **[Manual OAuth Testing](../manual_oauth_qa/README.md)** - Comprehensive Auth0 testing guide
- **[Main Project README](../README.md)** - Project overview and quick reference
- **[Examples Directory](../examples/)** - Complete working examples and demos

## Navigation Tips

### By Experience Level

**👋 New Users**: Quick Start → Use Cases → Memory Types
**🔧 Developers**: Getting Started → REST API → Configuration
**🤖 AI Agent Builders**: MCP Server → Memory Editing → Query Optimization
**🏗️ System Admins**: Authentication → Configuration → CLI

### By Use Case

**Building a chatbot?** → Quick Start → Memory Types → MCP Server
**Adding memory to existing app?** → REST API → Authentication → Configuration
**Research/content assistant?** → Use Cases → Query Optimization → Contextual Grounding
**Production deployment?** → Authentication → Vector Store Backends → Development

### By Interface Preference

**REST API users** → [API Documentation](api.md) → [Authentication](authentication.md)
**MCP/Claude users** → [MCP Server](mcp.md) → [Memory Editing](memory-editing.md)
**CLI management** → [CLI Reference](cli.md) → [Configuration](configuration.md)

## Feature Cross-Reference

| Feature | REST API | MCP Server | CLI | Documentation |
|---------|----------|------------|-----|---------------|
| **Memory Search** | ✅ `/v1/long-term-memory/search` | ✅ `search_long_term_memory` | ❌ | [REST API](api.md), [MCP](mcp.md) |
| **Memory Editing** | ✅ `PATCH /v1/long-term-memory/{id}` | ✅ `edit_long_term_memory` | ❌ | [Memory Editing](memory-editing.md) |
| **Query Optimization** | ✅ `optimize_query` param | ✅ `optimize_query` param | ❌ | [Query Optimization](query-optimization.md) |
| **Recency Boost** | ✅ Default enabled | ✅ Available | ❌ | [Recency Boost](recency-boost.md) |
| **Authentication** | ✅ JWT/Token | ✅ Inherited | ✅ Token management | [Authentication](authentication.md) |
| **Background Tasks** | ✅ Automatic | ✅ Automatic | ✅ Worker management | [Configuration](configuration.md) |

---

**Need help?** Check the [Quick Start Guide](quick-start.md) or explore [real-world examples](use-cases.md) to see Redis Agent Memory Server in action! 🧠✨
