# Developer Guide

Learn how to integrate memory into your AI applications. This guide covers integration patterns, memory types, extraction strategies, and memory lifecycle management.

## Core Concepts

<div class="grid cards" markdown>

-   🔄 **Memory Integration Patterns**

    ---

    Three patterns for using memory: LLM-driven, code-driven, and background extraction

    [Integration Patterns →](memory-integration-patterns.md)

-   📝 **Working Memory**

    ---

    Session-scoped storage for active conversation state

    [Working Memory →](working-memory.md)

-   🧠 **Long-term Memory**

    ---

    Persistent, cross-session storage for knowledge that should be retained

    [Long-term Memory →](long-term-memory.md)

-   🎯 **Memory Extraction Strategies**

    ---

    Configure how memories are extracted: discrete, summary, preferences, or custom

    [Extraction Strategies →](memory-extraction-strategies.md)

</div>

## Additional Topics

| Topic | Description |
|-------|-------------|
| [Summary Views](summary-views.md) | Pre-computed memory summaries for efficient context |
| [Memory Lifecycle](memory-lifecycle.md) | How memories are created, updated, and managed over time |
| [LangChain Integration](langchain-integration.md) | Use memory with LangChain agents and chains |

## Where to Start

**Building a chatbot?** Start with [Memory Integration Patterns](memory-integration-patterns.md) to understand your options.

**Need to understand the data model?** Read [Working Memory](working-memory.md) and [Long-term Memory](long-term-memory.md).

**Configuring extraction behavior?** See [Memory Extraction Strategies](memory-extraction-strategies.md).

**Looking for server configuration?** See the [Operations Guide](operations-guide-index.md) for authentication, LLM providers, and deployment.
