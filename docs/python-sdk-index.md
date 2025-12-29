# Python SDK

The Python SDK provides a simple, async-first interface for integrating memory into your AI applications. It includes client libraries, tool schemas for LLMs, and framework integrations.

<div class="grid cards" markdown>

-   🐍 **SDK Documentation**

    ---

    Complete API reference for the Python client library

    [SDK Reference →](python-sdk.md)

-   🦜 **LangChain Integration**

    ---

    Use memory with LangChain agents and chains

    [LangChain Guide →](langchain-integration.md)

-   ⚙️ **Configuration**

    ---

    Environment variables and configuration options

    [Configuration →](configuration.md)

</div>

## Installation

```bash
pip install agent-memory-client
```

## Quick Example

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Get memory-enriched context for your LLM
context = await client.memory_prompt(
    query="What restaurants should I try?",
    session_id="chat_123",
    long_term_search={"text": "food preferences", "limit": 5}
)
```

See the [SDK Documentation](python-sdk.md) for complete usage examples.
