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

The `recent_messages_limit` parameter is not yet exposed in the high-level Python SDK. Use the REST API directly via `httpx`:

```python
import httpx

async with httpx.AsyncClient(base_url="http://localhost:8000") as http:
    # Get only the 3 most recent messages
    resp = await http.get(
        f"/v1/working-memory/{session_id}",
        params={"namespace": "demo", "recent_messages_limit": 3},
    )
    resp.raise_for_status()
    messages = resp.json()["messages"]
```

The equivalent raw request is:

```
GET /v1/working-memory/{session_id}?namespace=demo&recent_messages_limit=3
```
