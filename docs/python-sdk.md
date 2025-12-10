# Python SDK

The Python SDK (`agent-memory-client`) provides the easiest way to integrate memory into your AI applications. It includes high-level abstractions, tool integration for OpenAI and Anthropic, and automatic function call resolution.

## Installation

**Requirements**: Python 3.10 or higher

```bash
pip install agent-memory-client
```

## Quick Start

```python
from agent_memory_client import MemoryAPIClient

# Connect to your memory server
client = MemoryAPIClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"  # Optional if auth disabled
)

# Store a memory
await client.create_long_term_memories([{
    "text": "User prefers morning meetings and hates scheduling calls after 4 PM",
    "memory_type": "semantic",
    "topics": ["scheduling", "preferences"],
    "user_id": "alice"
}])

# Search memories
results = await client.search_long_term_memory(
    text="when does user prefer meetings",
    limit=5
)
```

## Client Configuration

### Basic Setup

```python
from agent_memory_client import MemoryAPIClient

# Minimal configuration (development)
client = MemoryAPIClient(base_url="http://localhost:8000")

# Production configuration
client = MemoryAPIClient(
    base_url="https://your-memory-server.com",
    api_key="your-api-token",
    timeout=30.0,
    session_id="user-session-123",
    user_id="user-456",
    namespace="production"
)
```

### Authentication

```python
# Token authentication
client = MemoryAPIClient(
    base_url="https://your-server.com",
    api_key="your-token-here"
)

# OAuth2/JWT authentication
client = MemoryAPIClient(
    base_url="https://your-server.com",
    bearer_token="your-jwt-token"
)

# Development (no auth)
client = MemoryAPIClient(base_url="http://localhost:8000")
```

## Tool Integration

### OpenAI Integration

The SDK provides automatic tool schemas and function call resolution for OpenAI:

```python
import openai
from agent_memory_client import MemoryAPIClient

# Setup clients
memory_client = MemoryAPIClient(base_url="http://localhost:8000")
openai_client = openai.AsyncClient()

# Get tool schemas for OpenAI
memory_tools = MemoryAPIClient.get_all_memory_tool_schemas()

async def chat_with_memory(message: str, session_id: str):
    # Make request with memory tools
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}],
        tools=memory_tools,
        tool_choice="auto"
    )

    # Process tool calls automatically
    if response.choices[0].message.tool_calls:
        # Resolve all tool calls
        results = []
        for tool_call in response.choices[0].message.tool_calls:
            result = await memory_client.resolve_tool_call(
                tool_call=tool_call,
                session_id=session_id
            )
            if result["success"]:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result["formatted_response"]
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": f"Error: {result['error']}"
                })

        # Continue conversation with results
        messages = [
            {"role": "user", "content": message},
            response.choices[0].message,
            *results
        ]

        final_response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        return final_response.choices[0].message.content

    return response.choices[0].message.content
```

### Anthropic Integration

Similar tool integration for Anthropic Claude:

```python
import anthropic
from agent_memory_client import MemoryAPIClient

# Setup clients
memory_client = MemoryAPIClient(base_url="http://localhost:8000")
anthropic_client = anthropic.AsyncClient()

# Get tool schemas for Anthropic
memory_tools = MemoryAPIClient.get_all_memory_tool_schemas_anthropic()

async def chat_with_memory(message: str, session_id: str):
    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": message}],
        tools=memory_tools,
        max_tokens=1000
    )

    # Process tool calls
    if response.stop_reason == "tool_use":
        results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                result = await memory_client.resolve_tool_call(
                    tool_call={
                        "type": "tool_use",
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    },
                    session_id=session_id
                )
                if result["success"]:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result["formatted_response"]
                    })
                else:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error: {result['error']}"
                    })

        # Continue conversation
        messages = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.content + results}
        ]

        final_response = await anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            max_tokens=1000
        )

        return final_response.content[0].text

    return response.content[0].text
```

### Available Tools

The SDK provides these tools for LLM integration:

1. **`create_long_term_memory`** - Eagerly create long-term memories by making an API request
2. **`add_memory_to_working_memory`** - Lazily create memories by adding them to working memory (promoted to long-term storage later)
3. **`search_memory`** - Search with semantic similarity across long-term memories
4. **`edit_long_term_memory`** - Update existing long-term memories
5. **`delete_long_term_memories`** - Remove long-term memories
6. **`get_or_create_working_memory`** - Retrieve or create a working memory session
7. **`update_working_memory_data`** - Update session-specific data in working memory
8. **`get_current_datetime`** - Get current UTC datetime for grounding relative time expressions

**Note:** The following tool names have been deprecated for clarity:
- `create_long_term_memories` (deprecated) → use `eagerly_create_long_term_memory`
- `add_memory_to_working_memory` (deprecated) → use `lazily_create_long_term_memory`

### Customizing Tool Descriptions

The SDK provides `ToolSchema` and `ToolSchemaCollection` wrapper classes that allow you to customize tool descriptions, names, and parameter descriptions before passing them to LLMs. This is useful for:

- Adjusting descriptions to match your application's tone or domain
- Renaming tools to avoid conflicts with other tools
- Adding context-specific information to parameter descriptions

#### Basic Customization

```python
from agent_memory_client import MemoryAPIClient

# Get a tool schema and customize it
schema = MemoryAPIClient.get_memory_search_tool_schema()
schema.set_description("Search through the user's personal knowledge base")
schema.set_name("search_knowledge_base")

# Customize parameter descriptions
schema.set_parameter_description("query", "Natural language search query")

# Use with LLM
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=[schema.to_dict()]
)
```

#### Method Chaining

All setter methods return `self` for fluent method chaining:

```python
schema = (MemoryAPIClient.get_memory_search_tool_schema()
    .set_description("Find relevant information from memory")
    .set_name("find_info")
    .set_parameter_description("query", "What to search for"))
```

#### Bulk Customization with Collections

When working with all tools, use `ToolSchemaCollection` for bulk operations:

```python
# Get all tools as a collection
all_tools = MemoryAPIClient.get_all_memory_tool_schemas()

# Customize specific tools by name
all_tools.set_description("search_memory", "Find relevant memories")
all_tools.set_name("search_memory", "find_memories")

# Get a specific tool for detailed customization
search_tool = all_tools.get_by_name("find_memories")
if search_tool:
    search_tool.set_parameter_description("limit", "Max results to return")

# List all tool names
print(all_tools.names())  # ['find_memories', 'get_or_create_working_memory', ...]

# Convert to list for LLM consumption
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=all_tools.to_list()
)
```

#### Creating Independent Copies

Use `copy()` to create independent copies that won't affect the original:

```python
# Create a copy for customization
custom_schema = MemoryAPIClient.get_memory_search_tool_schema().copy()
custom_schema.set_description("Custom description")

# Original is unchanged
original = MemoryAPIClient.get_memory_search_tool_schema()
assert original.get_description() != custom_schema.get_description()
```

#### Anthropic Format

The same customization API works for Anthropic tool schemas:

```python
# Anthropic format
schema = MemoryAPIClient.get_memory_search_tool_schema_anthropic()
schema.set_description("Custom Anthropic description")

# Check the format
print(schema.format)  # "anthropic"

# Use with Anthropic
response = await anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    tools=[schema.to_dict()]
)
```

#### ToolSchema API Reference

| Method | Description |
|--------|-------------|
| `set_description(text)` | Set the tool description |
| `set_name(name)` | Set the tool name |
| `set_parameter_description(param, text)` | Set a parameter's description |
| `get_description()` | Get the current description |
| `get_name()` | Get the current name |
| `get_parameter_description(param)` | Get a parameter's description |
| `to_dict()` | Convert to dict (returns deep copy) |
| `copy()` | Create an independent copy |
| `format` | Property: "openai" or "anthropic" |

#### ToolSchemaCollection API Reference

| Method | Description |
|--------|-------------|
| `get_by_name(name)` | Get a specific tool by name |
| `set_description(name, text)` | Set description for a tool by name |
| `set_name(old_name, new_name)` | Rename a tool |
| `names()` | Get list of all tool names |
| `to_list()` | Convert to list of dicts |
| `copy()` | Create an independent copy |
| `len(collection)` | Get number of tools |
| `collection[index]` | Access tool by index |
| `for tool in collection` | Iterate over tools |

## Memory Operations

### Creating Memories

```python
# Create multiple memories
memories = [
    {
        "text": "User works as a software engineer at TechCorp",
        "memory_type": "semantic",
        "topics": ["career", "work", "company"],
        "entities": ["TechCorp", "software engineer"],
        "user_id": "alice"
    },
    {
        "text": "User prefers Python and TypeScript for development",
        "memory_type": "semantic",
        "topics": ["programming", "preferences", "languages"],
        "entities": ["Python", "TypeScript"],
        "user_id": "alice"
    }
]

result = await client.create_long_term_memories(memories)
print(f"Created {len(result.memories)} memories")
```

### Searching Memories

```python
# Basic semantic search
results = await client.search_long_term_memory(
    text="user programming experience",
    limit=10
)

# Advanced filtering
results = await client.search_long_term_memory(
    text="user preferences",
    user_id="alice",
    topics=["programming", "food"],
    limit=5,
    min_relevance_score=0.7
)

# Time-based filtering
from datetime import datetime, timedelta

week_ago = datetime.now() - timedelta(days=7)
results = await client.search_long_term_memory(
    text="recent updates",
    created_after=week_ago,
    limit=10
)

# Process results
for memory in results.memories:
    print(f"Relevance: {memory.relevance_score:.2f}")
    print(f"Text: {memory.text}")
    print(f"Topics: {', '.join(memory.topics or [])}")
```

### Memory Editing

```python
# Update a memory
await client.edit_memory(
    memory_id="memory-123",
    updates={
        "text": "User works as a senior software engineer at TechCorp",
        "topics": ["career", "work", "company", "senior"],
        "entities": ["TechCorp", "senior software engineer"]
    }
)

# Add context to existing memory
await client.edit_memory(
    memory_id="memory-456",
    updates={
        "text": "User prefers Python and TypeScript for development. Recently started learning Rust.",
        "topics": ["programming", "preferences", "languages", "rust"],
        "entities": ["Python", "TypeScript", "Rust"]
    }
)
```

### Working Memory

```python
# Store conversation context
conversation = {
    "messages": [
        {"role": "user", "content": "I'm planning a trip to Italy"},
        {"role": "assistant", "content": "That sounds exciting! What cities are you thinking of visiting?"},
        {"role": "user", "content": "Rome and Florence, maybe Venice too"}
    ],
    "memories": [
        {
            "text": "User is planning a trip to Italy, considering Rome, Florence, and Venice",
            "memory_type": "semantic",
            "topics": ["travel", "italy", "vacation"],
            "entities": ["Italy", "Rome", "Florence", "Venice"]
        }
    ]
}

await client.set_working_memory("session-123", conversation)

# Retrieve or create working memory
created, memory = await client.get_or_create_working_memory("session-123")
if created:
    print("Created new session")
else:
    print("Found existing session")
print(f"Session has {len(memory.messages)} messages")
```

## Memory-Enhanced Conversations

### Context Injection

The SDK provides a powerful `memory_prompt` method that automatically enriches your prompts with relevant context:

```python
async def get_contextualized_response(user_message: str, session_id: str, user_id: str):
    # Get memory-enriched context
    context = await client.memory_prompt(
        query=user_message,
        session={
            "session_id": session_id,
            "user_id": user_id,
            "model_name": "gpt-4o"
        },
        long_term_search={
            "text": user_message,
            "limit": 5,
            "user_id": user_id
        }
    )

    # Send to LLM
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=context.messages
    )

    return response.choices[0].message.content
```

### Automatic Memory Storage

```python
async def chat_with_auto_memory(message: str, session_id: str):
    # Get contextualized prompt
    context = await client.memory_prompt(
        query=message,
        session={"session_id": session_id, "model_name": "gpt-4o"}
    )

    # Generate response
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=context.messages + [{"role": "user", "content": message}]
    )

    # Store the conversation
    conversation = {
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.choices[0].message.content}
        ]
    }

    await client.set_working_memory(session_id, conversation)

    return response.choices[0].message.content
```

## Batch Operations

### Bulk Memory Creation

```python
# Process large datasets efficiently
async def import_user_data(user_data: list, user_id: str):
    batch_size = 50

    for i in range(0, len(user_data), batch_size):
        batch = user_data[i:i + batch_size]

        memories = [
            {
                "text": item["description"],
                "memory_type": "semantic",
                "topics": item.get("categories", []),
                "entities": item.get("entities", []),
                "user_id": user_id,
                "metadata": {"source": item["source"]}
            }
            for item in batch
        ]

        result = await client.create_long_term_memories(memories)
        print(f"Imported batch {i//batch_size + 1}, {len(result.memories)} memories")
```

### Bulk Search Operations

```python
# Search multiple queries efficiently
async def multi_search(queries: list[str], user_id: str):
    results = {}

    # Use asyncio.gather for concurrent searches
    search_tasks = [
        client.search_long_term_memory(
            text=query,
            user_id=user_id,
            limit=3
        )
        for query in queries
    ]

    search_results = await asyncio.gather(*search_tasks)

    for query, result in zip(queries, search_results):
        results[query] = [memory.text for memory in result.memories]

    return results
```

## Error Handling

### Robust Client Usage

```python
from agent_memory_client import MemoryAPIClient, MemoryError
import asyncio
import logging

async def robust_memory_operation(client: MemoryAPIClient):
    try:
        # Attempt memory operation
        results = await client.search_long_term_memory(
            text="user preferences",
            limit=5
        )

        return results.memories

    except MemoryError as e:
        if e.status_code == 401:
            logging.error("Authentication failed - check API key")
        elif e.status_code == 429:
            logging.warning("Rate limited - waiting before retry")
            await asyncio.sleep(5)
            return await robust_memory_operation(client)
        else:
            logging.error(f"Memory API error: {e}")
            return []

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []
```

### Connection Management

```python
import httpx
from agent_memory_client import MemoryAPIClient

# Custom timeout and retry configuration
async with httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
) as http_client:

    client = MemoryAPIClient(
        base_url="http://localhost:8000",
        http_client=http_client
    )

    # Perform operations
    results = await client.search_long_term_memory(text="query")
```

## Advanced Features

### Custom Tool Workflows

```python
class CustomMemoryAgent:
    def __init__(self, memory_client: MemoryAPIClient):
        self.memory = memory_client

    async def intelligent_search(self, query: str, user_id: str):
        # Multi-stage search with refinement
        initial_results = await self.memory.search_long_term_memory(
            text=query,
            user_id=user_id,
            limit=20
        )

        if not initial_results.memories:
            # Try broader search
            return await self.memory.search_long_term_memory(
                text=query,
                limit=10
            )

        # Filter by relevance threshold
        relevant_memories = [
            m for m in initial_results.memories
            if m.relevance_score > 0.7
        ]

        return relevant_memories[:5]

    async def contextual_store(self, text: str, context: dict, user_id: str):
        # Extract topics and entities from context
        topics = context.get("topics", [])
        entities = context.get("entities", [])

        # Search for similar existing memories
        similar = await self.memory.search_long_term_memory(
            text=text,
            user_id=user_id,
            limit=3,
            min_relevance_score=0.8
        )

        if similar.memories:
            # Update existing memory instead of creating duplicate
            await self.memory.edit_memory(
                memory_id=similar.memories[0].id,
                updates={
                    "text": f"{similar.memories[0].text}. {text}",
                    "topics": list(set(similar.memories[0].topics + topics)),
                    "entities": list(set(similar.memories[0].entities + entities))
                }
            )
        else:
            # Create new memory
            await self.memory.create_long_term_memories([{
                "text": text,
                "memory_type": "semantic",
                "topics": topics,
                "entities": entities,
                "user_id": user_id
            }])
```

### Performance Optimization

```python
from functools import lru_cache
import asyncio

class OptimizedMemoryClient:
    def __init__(self, client: MemoryAPIClient):
        self.client = client
        self._search_cache = {}

    @lru_cache(maxsize=100)
    def _cache_key(self, text: str, user_id: str, limit: int) -> str:
        return f"{text}:{user_id}:{limit}"

    async def cached_search(self, text: str, user_id: str, limit: int = 5):
        cache_key = self._cache_key(text, user_id, limit)

        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = await self.client.search_long_term_memory(
            text=text,
            user_id=user_id,
            limit=limit
        )

        # Cache results for 5 minutes
        self._search_cache[cache_key] = results
        asyncio.create_task(self._expire_cache(cache_key, 300))

        return results

    async def _expire_cache(self, key: str, delay: int):
        await asyncio.sleep(delay)
        self._search_cache.pop(key, None)
```

## Best Practices

### 1. Client Management

```python
# Use a single client instance per application
class MemoryService:
    def __init__(self):
        self.client = MemoryAPIClient(
            base_url=os.getenv("MEMORY_SERVER_URL"),
            api_key=os.getenv("MEMORY_API_KEY")
        )

    async def close(self):
        await self.client.close()

# Singleton pattern
memory_service = MemoryService()
```

### 2. Memory Organization

```python
# Use consistent naming patterns
async def create_user_memory(text: str, user_id: str, category: str):
    return await client.create_long_term_memories([{
        "text": text,
        "memory_type": "semantic",
        "topics": [category, "user-preference"],
        "user_id": user_id,
        "namespace": f"user:{user_id}:preferences"
    }])
```

### 3. Context Management

```python
# Implement context-aware memory storage
async def store_conversation_memory(conversation: dict, session_id: str):
    # Extract key information
    important_facts = extract_facts(conversation)

    if important_facts:
        await client.create_long_term_memories([{
            "text": fact,
            "memory_type": "semantic",
            "session_id": session_id,
            "metadata": {"conversation_turn": i}
        } for i, fact in enumerate(important_facts)])
```

## Configuration Reference

### Environment Variables

```bash
# Client configuration
MEMORY_SERVER_URL=http://localhost:8000
MEMORY_API_KEY=your-api-token

# Connection settings
MEMORY_TIMEOUT=30
MEMORY_MAX_RETRIES=3

# Default user settings
DEFAULT_USER_ID=default-user
DEFAULT_NAMESPACE=production
```

### Client Options

```python
client = MemoryAPIClient(
    base_url="http://localhost:8000",
    api_key="optional-token",
    bearer_token="optional-jwt",
    timeout=30.0,
    max_retries=3,
    session_id="default-session",
    user_id="default-user",
    namespace="default",
    http_client=custom_httpx_client
)
```

The Python SDK makes it easy to add sophisticated memory capabilities to any AI application, with minimal setup and maximum flexibility. Use the tool integrations for LLM-driven memory, direct API calls for code-driven approaches, or combine both patterns for hybrid solutions.
