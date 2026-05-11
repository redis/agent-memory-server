# 📊 Recent Messages Limit Demo

**File**: [`examples/recent_messages_limit_demo.py`](https://github.com/redis/agent-memory-server/blob/main/examples/recent_messages_limit_demo.py)

Demonstrates the `recent_messages_limit` parameter for efficiently retrieving only the most recent N messages from working memory.

## Core Concept

When working memory grows large, retrieving all messages is expensive. The `recent_messages_limit` parameter lets you fetch only the N most recent messages, useful for context windows and UI displays.

## Key Features

- **Efficient Retrieval**: Fetch only the messages you need instead of the full history
- **SDK & HTTP Integration**: Uses `agent_memory_client` to store working memory and raw HTTP requests to retrieve it
- **Multiple Scenarios**: Tests various limits (3, 5, 20, 2) to show how `recent_messages_limit` changes the results
- **Direct API Verification**: Uses the raw HTTP API for retrieval so you can inspect the exact responses from the server

## Usage Examples

```bash
cd examples
python recent_messages_limit_demo.py
```

## Key Implementation Pattern

```python
from agent_memory_client import create_memory_client

client = await create_memory_client(base_url="http://localhost:8000")

# Get only the 3 most recent messages
_created, memory = await client.get_or_create_working_memory(
    session_id="my-session",
    namespace="demo",
    context_window_max=3
)
```
