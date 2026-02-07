# Developer Guide

Learn how to integrate memory into your AI applications. This guide covers integration patterns, memory types, extraction strategies, and production considerations.

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
| [Memory Lifecycle](memory-lifecycle.md) | How memories are created, updated, and managed over time |
| [LLM Providers](llm-providers.md) | Configure OpenAI, Anthropic, AWS Bedrock, Ollama, and more |
| [Embedding Providers](embedding-providers.md) | Configure embedding models for semantic search |
| [Custom Memory Vector Databases](custom-memory-vector-db.md) | Configure Redis or custom memory vector databases |
| [AWS Bedrock](aws-bedrock.md) | AWS-specific setup for Bedrock models |
| [Authentication](authentication.md) | OAuth2/JWT and token-based authentication |
| [Security](security-custom-prompts.md) | Security considerations for custom prompts |

## Where to Start

**Building a chatbot?** Start with [Memory Integration Patterns](memory-integration-patterns.md) to understand your options.

**Need to understand the data model?** Read [Working Memory](working-memory.md) and [Long-term Memory](long-term-memory.md).

**Configuring extraction behavior?** See [Memory Extraction Strategies](memory-extraction-strategies.md).
