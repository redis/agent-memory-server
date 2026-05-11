---
description: Foundational concepts for Agent Memory Server.
---

# Concepts

Memory in this system has **two tiers**. **Working memory** holds the live
conversation for a single session. **Long-term memory** holds anything
worth keeping across conversations — facts, preferences, decisions. The
**lifecycle** pages cover how content moves between the two tiers
(extraction, promotion, deduplication, eventual forgetting). Everything
else on this page is a mechanism that supports those two tiers.

If you only read two pages, read [Working memory](working_memory.md) and
[Long-term memory](long_term_memory.md). Then come back for the rest.

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **[Working memory](working_memory.md)**

    ---

    Session-scoped conversation state with automatic summarization and context tracking.

-   :material-database:{ .lg .middle } **[Long-term memory](long_term_memory.md)**

    ---

    Persistent storage with semantic, keyword, and hybrid search across sessions.

-   :material-chart-box:{ .lg .middle } **[Summary views](summary_views.md)**

    ---

    Aggregated views of memory content for dashboards and analytics.

-   :material-refresh:{ .lg .middle } **[Memory lifecycle](memory_lifecycle.md)**

    ---

    How memories are created, promoted, deduplicated, and eventually forgotten.

-   :material-magnify:{ .lg .middle } **[Memory extraction](memory_extraction.md)**

    ---

    How important facts are automatically extracted from conversations.

-   :material-link-variant:{ .lg .middle } **[Contextual grounding](contextual_grounding.md)**

    ---

    How retrieved memories are injected into prompts at the right moment.

-   :material-sort-clock-descending:{ .lg .middle } **[Recency boost](recency_boost.md)**

    ---

    Tunable scoring that biases newer memories without losing semantic relevance.

</div>

## Related Topics

| Topic | Description |
|-------|-------------|
| [LangChain Integration](../examples/langchain.md) | Use memory with LangChain agents and chains |
| [Custom Memory Vector Databases](../user_guide/how_to_guides/custom_vector_db.md) | Configure Redis or a custom vector backend |
| [Use cases](../examples/use_cases.md) | What people actually build with this — read this if you're still deciding |

## When to Use Advanced Features

| Feature | Use When |
|---------|----------|
| [Query Optimization](../user_guide/how_to_guides/query_optimization.md) | Search results aren't matching user intent well |
| [Recency Boost](recency_boost.md) | Recent memories should rank higher than older ones |
| [Advanced Vector Config](../user_guide/how_to_guides/advanced_vector_db.md) | You need to tune performance or use custom distance metrics |
| [Contextual Grounding](contextual_grounding.md) | Extracted memories contain unresolved pronouns like "he" or "it" |
| [Memory Editing](../user_guide/how_to_guides/memory_editing.md) | You need to correct, enrich, or update stored memories after creation |

## Where to Start

**Building a chatbot?** Start with [Memory Integration Patterns](../user_guide/how_to_guides/integration_patterns.md) to understand your options.

**Need to understand the data model?** Read [Working Memory](working_memory.md) and [Long-term Memory](long_term_memory.md).

**Configuring extraction behavior?** See [Memory Extraction](memory_extraction.md).

**Looking for server configuration?** See the [How-To Guides](../user_guide/how_to_guides/index.md) for authentication, LLM providers, and deployment.
