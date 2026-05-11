---
description: Reference for the agent_memory_server Python package.
---

# Server package reference

Auto-generated reference for the `agent_memory_server` Python package.
Module pages are generated from Google-style docstrings via
[mkdocstrings](https://mkdocstrings.github.io/).

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **[Models](models.md)**

    ---

    Pydantic data models for memory records, sessions, and search payloads.

-   :material-memory:{ .lg .middle } **[Working memory](working_memory.md)**

    ---

    Session-scoped memory APIs for messages, structured memories, and metadata.

-   :material-database-search:{ .lg .middle } **[Long-term memory](long_term_memory.md)**

    ---

    Persistent memory with semantic search, deduplication, and compaction.

-   :material-strategy:{ .lg .middle } **[Memory strategies](memory_strategies.md)**

    ---

    Pluggable strategies for promoting working memory into long-term storage.

-   :material-vector-line:{ .lg .middle } **[Vector DB](memory_vector_db.md)**

    ---

    Vector database abstraction and factory used by the long-term memory layer.

-   :material-text-search:{ .lg .middle } **[Extraction](extraction.md)**

    ---

    Topic and entity extraction (LLM and BERT-based) over message streams.

-   :material-text-box-outline:{ .lg .middle } **[Summarization](summarization.md)**

    ---

    Conversation summarization when context windows are exceeded.

-   :material-filter:{ .lg .middle } **[Filters](filters.md)**

    ---

    Filter primitives (session, namespace, topics, entities, timestamps).

-   :material-shield-key:{ .lg .middle } **[Auth](auth.md)**

    ---

    OAuth2/JWT verification and JWKS handling.

-   :material-cog:{ .lg .middle } **[Config](config.md)**

    ---

    Settings loaded from environment variables.

-   :material-robot:{ .lg .middle } **[LLM client](llm.md)**

    ---

    LiteLLM-backed chat and embedding client.

</div>
