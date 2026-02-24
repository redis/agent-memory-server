# Python SDK

The Python SDK (`agent-memory-client`) provides the easiest way to integrate memory into your AI applications. It includes high-level abstractions, tool integration for OpenAI and Anthropic, and automatic function call resolution.

**Version**: 0.14.0+

## Installation

**Requirements**: Python 3.10 or higher

```bash
pip install agent-memory-client
```

## Quick Start

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum

# Configure and create client
config = MemoryClientConfig(
    base_url="http://localhost:8000",
    default_namespace="my-app"
)

async with MemoryAPIClient(config) as client:
    # Store a memory
    await client.create_long_term_memory([
        ClientMemoryRecord(
            text="User prefers morning meetings",
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=["scheduling", "preferences"],
            user_id="alice"
        )
    ])

    # Search memories
    results = await client.search_long_term_memory(
        text="when does user prefer meetings",
        limit=5
    )

    for memory in results.memories:
        print(f"{memory.text} (score: {1 - memory.dist:.2f})")
```

## Client Configuration

### Using MemoryClientConfig

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

# Minimal configuration (development)
config = MemoryClientConfig(base_url="http://localhost:8000")
client = MemoryAPIClient(config)

# Production configuration with defaults
config = MemoryClientConfig(
    base_url="https://your-memory-server.com",
    timeout=30.0,
    default_namespace="production",
    default_model_name="gpt-4o",  # For token counting
    default_context_window_max=128000  # Override context window
)
client = MemoryAPIClient(config)
```

### Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `base_url` | `str` | Memory server URL (required) |
| `timeout` | `float` | HTTP timeout in seconds (default: 30.0) |
| `default_namespace` | `str` | Default namespace for all operations |
| `default_model_name` | `str` | Model name for context window sizing |
| `default_context_window_max` | `int` | Override max context window tokens |

### Async Context Manager

The client supports async context manager for proper resource cleanup:

```python
async with MemoryAPIClient(config) as client:
    # Client automatically closes when exiting the context
    results = await client.search_long_term_memory(text="query")

# Or manually manage lifecycle
client = MemoryAPIClient(config)
try:
    results = await client.search_long_term_memory(text="query")
finally:
    await client.close()
```

## Tool Integration

### OpenAI Integration

The SDK provides automatic tool schemas and function call resolution for OpenAI:

```python
import openai
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

# Setup clients
config = MemoryClientConfig(base_url="http://localhost:8000")
memory_client = MemoryAPIClient(config)
openai_client = openai.AsyncClient()

# Get tool schemas for OpenAI (returns ToolSchemaCollection)
memory_tools = MemoryAPIClient.get_all_memory_tool_schemas()

async def chat_with_memory(message: str, session_id: str):
    # Make request with memory tools (convert to list for API)
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}],
        tools=memory_tools.to_list(),
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
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

# Setup clients
config = MemoryClientConfig(base_url="http://localhost:8000")
memory_client = MemoryAPIClient(config)
anthropic_client = anthropic.AsyncClient()

# Get tool schemas for Anthropic (returns ToolSchemaCollection)
memory_tools = MemoryAPIClient.get_all_memory_tool_schemas_anthropic()

async def chat_with_memory(message: str, session_id: str):
    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": message}],
        tools=memory_tools.to_list(),
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

1. **`eagerly_create_long_term_memory`** - Create long-term memories directly for immediate storage and retrieval
2. **`lazily_create_long_term_memory`** - Store memories that will be automatically promoted to long-term storage
3. **`search_memory`** - Search with semantic similarity across long-term memories
4. **`edit_long_term_memory`** - Update existing long-term memories
5. **`delete_long_term_memories`** - Remove long-term memories
6. **`get_or_create_working_memory`** - Retrieve or create a working memory session
7. **`update_working_memory_data`** - Update session-specific data in working memory
8. **`get_long_term_memory`** - Retrieve a specific long-term memory by ID
9. **`get_current_datetime`** - Get current UTC datetime for grounding relative time expressions

**Note:** The following tool names have been deprecated and will continue to work as aliases:
- `create_long_term_memory` → use `eagerly_create_long_term_memory`
- `add_memory_to_working_memory` → use `lazily_create_long_term_memory`

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
    search_tool.set_parameter_description("max_results", "Max results to return")

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

### Creating Long-Term Memories

```python
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum

# Create multiple memories
memories = [
    ClientMemoryRecord(
        text="User works as a software engineer at TechCorp",
        memory_type=MemoryTypeEnum.SEMANTIC,
        topics=["career", "work", "company"],
        entities=["TechCorp", "software engineer"],
        user_id="alice"
    ),
    ClientMemoryRecord(
        text="User prefers Python and TypeScript for development",
        memory_type=MemoryTypeEnum.SEMANTIC,
        topics=["programming", "preferences", "languages"],
        entities=["Python", "TypeScript"],
        user_id="alice"
    )
]

result = await client.create_long_term_memory(memories)
print(f"Created memories: {result.status}")
```

### Searching Memories with Filters

The SDK provides powerful filter classes for precise memory retrieval:

```python
from agent_memory_client.filters import (
    Topics, Entities, CreatedAt, UserId, Namespace, MemoryType
)
from datetime import datetime, timedelta, timezone

# Basic semantic search
results = await client.search_long_term_memory(
    text="user programming experience",
    limit=10
)

# Filter using filter objects (recommended)
results = await client.search_long_term_memory(
    text="user preferences",
    user_id=UserId(eq="alice"),
    topics=Topics(any=["programming", "food"]),  # Match any of these topics
    distance_threshold=0.3,  # Lower = more relevant (0-1 scale)
    limit=5
)

# Time-based filtering with CreatedAt
week_ago = datetime.now(timezone.utc) - timedelta(days=7)
results = await client.search_long_term_memory(
    text="recent updates",
    created_at=CreatedAt(gte=week_ago),  # Greater than or equal
    limit=10
)

# Filter by memory type
results = await client.search_long_term_memory(
    text="events that happened",
    memory_type=MemoryType(eq="episodic"),
    limit=10
)

# Process results
for memory in results.memories:
    relevance = 1 - memory.dist if memory.dist else None
    print(f"Relevance: {relevance:.2f}" if relevance else "No score")
    print(f"Text: {memory.text}")
    print(f"Topics: {', '.join(memory.topics or [])}")
```

#### Filter Reference

| Filter | Options | Description |
|--------|---------|-------------|
| `SessionId` | `eq`, `in_`, `not_eq`, `not_in`, `startswith` | Filter by session ID |
| `Namespace` | `eq`, `in_`, `not_eq`, `not_in`, `startswith` | Filter by namespace |
| `UserId` | `eq`, `in_`, `not_eq`, `not_in`, `startswith` | Filter by user ID |
| `Topics` | `any`, `all`, `none` | Filter by topics |
| `Entities` | `any`, `all`, `none` | Filter by entities |
| `CreatedAt` | `gte`, `lte`, `eq` | Filter by creation date |
| `LastAccessed` | `gte`, `lte`, `eq` | Filter by last access date |
| `MemoryType` | `eq`, `in_`, `not_eq`, `not_in` | Filter by memory type |

### Memory Editing

```python
# Update a memory by ID (get ID from search results)
updated = await client.edit_long_term_memory(
    memory_id="01HXYZ...",  # ULID from search results
    updates={
        "text": "User works as a senior software engineer at TechCorp",
        "topics": ["career", "work", "company", "senior"],
        "entities": ["TechCorp", "senior software engineer"]
    }
)
print(f"Updated: {updated.text}")

# Get a specific memory by ID
memory = await client.get_long_term_memory(memory_id="01HXYZ...")
print(f"Memory: {memory.text}")

# Delete memories
await client.delete_long_term_memories(["memory-id-1", "memory-id-2"])
```

### Working Memory

```python
from agent_memory_client.models import (
    WorkingMemory, MemoryMessage, ClientMemoryRecord, MemoryTypeEnum
)

# Get or create working memory (returns tuple of created, memory)
created, memory = await client.get_or_create_working_memory(
    session_id="session-123",
    user_id="alice",
    namespace="my-app"
)
if created:
    print("Created new session")
else:
    print(f"Found existing session with {len(memory.messages)} messages")

# Store/update working memory with messages
working_memory = WorkingMemory(
    session_id="session-123",
    namespace="my-app",
    messages=[
        MemoryMessage(role="user", content="I'm planning a trip to Italy"),
        MemoryMessage(role="assistant", content="That sounds exciting!"),
    ],
    memories=[
        ClientMemoryRecord(
            text="User is planning a trip to Italy",
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=["travel", "italy"]
        )
    ],
    data={"destination": "Italy", "budget": 2000}
)

response = await client.put_working_memory("session-123", working_memory)
print(f"Stored {len(response.messages)} messages")

# Convenience: Set only the data portion
await client.set_working_memory_data(
    session_id="session-123",
    data={"trip_destination": "Rome", "travel_dates": ["2024-06-01", "2024-06-07"]}
)

# Convenience: Add memories to working memory
await client.add_memories_to_working_memory(
    session_id="session-123",
    memories=[
        ClientMemoryRecord(
            text="User prefers boutique hotels",
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=["travel", "preferences"]
        )
    ]
)

# Delete working memory when session ends
await client.delete_working_memory(session_id="session-123")
```

### Forgetting Memories

Use `ForgetPolicy` to clean up old or inactive memories:

```python
from agent_memory_client.models import ForgetPolicy

# Define a forget policy
policy = ForgetPolicy(
    max_age_days=90,           # Forget memories older than 90 days
    max_inactive_days=30,      # Or inactive for 30+ days
    budget=100,                # Process up to 100 memories per run
    memory_type_allowlist=["episodic"]  # Only forget episodic memories
)

# Dry run to preview what would be deleted
preview = await client.forget_long_term_memories(
    policy=policy,
    namespace="my-app",
    user_id="alice",
    dry_run=True  # Preview only, do not delete
)
print(f"Would delete {preview.deleted} of {preview.scanned} memories")

# Execute forget operation
result = await client.forget_long_term_memories(
    policy=policy,
    namespace="my-app",
    user_id="alice",
    pinned_ids=["memory-to-keep-1", "memory-to-keep-2"],  # Exclude these
    dry_run=False
)
print(f"Deleted {result.deleted} memories: {result.deleted_ids}")
```

### Summary Views

Summary Views create aggregated summaries of memories, grouped by fields you specify:

```python
from agent_memory_client.models import CreateSummaryViewRequest, SummaryViewSource

# Create a summary view that groups by user and topic
request = CreateSummaryViewRequest(
    name="User Topic Summaries",
    source=SummaryViewSource.LONG_TERM,
    group_by=["user_id", "topics"],
    time_window_days=30,  # Only last 30 days
    continuous=True,      # Auto-refresh in background
    prompt="Summarize these memories concisely:",  # Custom prompt
    model_name="gpt-4o-mini"  # Override model
)

view = await client.create_summary_view(request)
print(f"Created view: {view.id}")

# List all views
views = await client.list_summary_views()
for v in views:
    print(f"View: {v.name} (groups by: {v.group_by})")

# Run a specific partition (sync)
partition_result = await client.run_summary_view_partition(
    view_id=view.id,
    group={"user_id": "alice", "topics": "travel"}
)
print(f"Summary: {partition_result.summary}")
print(f"Based on {partition_result.memory_count} memories")

# Run full view as background task
task = await client.run_summary_view(view_id=view.id, force=True)
print(f"Task ID: {task.id}, Status: {task.status}")

# Poll for completion
import asyncio
while True:
    task = await client.get_task(task.id)
    if task.status in ["completed", "failed"]:
        break
    await asyncio.sleep(1)

# List computed partitions
partitions = await client.list_summary_view_partitions(
    view_id=view.id,
    user_id="alice"
)
for p in partitions:
    print(f"Group: {p.group}, Summary: {p.summary[:100]}...")

# Delete a view
await client.delete_summary_view(view.id)
```

### Recency Boosting

Use `RecencyConfig` to boost recent memories in search results:

```python
from agent_memory_client.models import RecencyConfig

# Boost recently accessed memories
results = await client.search_long_term_memory(
    text="user preferences",
    recency=RecencyConfig(
        decay_factor=0.9,  # How fast relevance decays (0-1)
        reference_timestamp=None  # Use current time
    ),
    limit=10
)
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
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum

# Process large datasets efficiently
async def import_user_data(user_data: list, user_id: str):
    batch_size = 50

    for i in range(0, len(user_data), batch_size):
        batch = user_data[i:i + batch_size]

        memories = [
            ClientMemoryRecord(
                text=item["description"],
                memory_type=MemoryTypeEnum.SEMANTIC,
                topics=item.get("categories", []),
                entities=item.get("entities", []),
                user_id=user_id,
            )
            for item in batch
        ]

        result = await client.create_long_term_memory(memories)
        print(f"Imported batch {i//batch_size + 1}: {result.status}")
```

### Bulk Search Operations

```python
import asyncio
from agent_memory_client.filters import UserId

# Search multiple queries efficiently
async def multi_search(queries: list[str], user_id: str):
    results = {}

    # Use asyncio.gather for concurrent searches
    search_tasks = [
        client.search_long_term_memory(
            text=query,
            user_id=UserId(eq=user_id),
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

### Exception Classes

```python
from agent_memory_client import (
    MemoryClientError,      # Base exception for all client errors
    MemoryNotFoundError,    # Memory not found (404)
    MemoryServerError,      # Server error (5xx)
    MemoryValidationError,  # Invalid input (400)
)
```

### Robust Client Usage

```python
from agent_memory_client import (
    MemoryAPIClient, MemoryClientConfig,
    MemoryClientError, MemoryNotFoundError, MemoryServerError
)
import asyncio
import logging

config = MemoryClientConfig(base_url="http://localhost:8000")

async def robust_memory_operation(client: MemoryAPIClient):
    try:
        results = await client.search_long_term_memory(
            text="user preferences",
            limit=5
        )
        return results.memories

    except MemoryNotFoundError:
        logging.warning("No matching memories found")
        return []

    except MemoryServerError as e:
        logging.error(f"Server error: {e}")
        await asyncio.sleep(5)
        return await robust_memory_operation(client)  # Retry

    except MemoryClientError as e:
        logging.error(f"Client error: {e}")
        return []

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []
```

### Using Async Context Manager

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

config = MemoryClientConfig(base_url="http://localhost:8000", timeout=30.0)

# Recommended: Use context manager for automatic cleanup
async with MemoryAPIClient(config) as client:
    results = await client.search_long_term_memory(text="query")
    # Client automatically closes when exiting
```

## Advanced Features

### Custom Tool Workflows

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum
from agent_memory_client.filters import UserId

class CustomMemoryAgent:
    def __init__(self, memory_client: MemoryAPIClient):
        self.memory = memory_client

    async def intelligent_search(self, query: str, user_id: str):
        # Multi-stage search with refinement
        initial_results = await self.memory.search_long_term_memory(
            text=query,
            user_id=UserId(eq=user_id),
            limit=20
        )

        if not initial_results.memories:
            # Try broader search
            return await self.memory.search_long_term_memory(
                text=query,
                limit=10
            )

        # Filter by distance (lower is more relevant)
        relevant_memories = [
            m for m in initial_results.memories
            if m.dist and m.dist < 0.3  # Close matches
        ]

        return relevant_memories[:5]

    async def contextual_store(self, text: str, context: dict, user_id: str):
        # Extract topics and entities from context
        topics = context.get("topics", [])
        entities = context.get("entities", [])

        # Search for similar existing memories
        similar = await self.memory.search_long_term_memory(
            text=text,
            user_id=UserId(eq=user_id),
            limit=3,
            distance_threshold=0.2  # Close matches only
        )

        if similar.memories:
            # Update existing memory instead of creating duplicate
            existing = similar.memories[0]
            await self.memory.edit_long_term_memory(
                memory_id=existing.id,
                updates={
                    "text": f"{existing.text}. {text}",
                    "topics": list(set((existing.topics or []) + topics)),
                    "entities": list(set((existing.entities or []) + entities))
                }
            )
        else:
            # Create new memory
            await self.memory.create_long_term_memory([
                ClientMemoryRecord(
                    text=text,
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    topics=topics,
                    entities=entities,
                    user_id=user_id
                )
            ])
```

### Performance Optimization

```python
import asyncio
from agent_memory_client.filters import UserId

class OptimizedMemoryClient:
    def __init__(self, client: MemoryAPIClient):
        self.client = client
        self._search_cache = {}

    def _cache_key(self, text: str, user_id: str, limit: int) -> str:
        return f"{text}:{user_id}:{limit}"

    async def cached_search(self, text: str, user_id: str, limit: int = 5):
        cache_key = self._cache_key(text, user_id, limit)

        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = await self.client.search_long_term_memory(
            text=text,
            user_id=UserId(eq=user_id),
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
import os
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

# Use a single client instance per application
class MemoryService:
    def __init__(self):
        config = MemoryClientConfig(
            base_url=os.getenv("MEMORY_SERVER_URL", "http://localhost:8000"),
            default_namespace=os.getenv("DEFAULT_NAMESPACE", "production"),
            timeout=float(os.getenv("MEMORY_TIMEOUT", "30"))
        )
        self.client = MemoryAPIClient(config)

    async def close(self):
        await self.client.close()

# Usage with context manager
async def main():
    service = MemoryService()
    try:
        # Use service.client
        pass
    finally:
        await service.close()
```

### 2. Memory Organization

```python
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum

# Use consistent naming patterns
async def create_user_memory(text: str, user_id: str, category: str):
    return await client.create_long_term_memory([
        ClientMemoryRecord(
            text=text,
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=[category, "user-preference"],
            user_id=user_id,
            namespace=f"user:{user_id}:preferences"
        )
    ])
```

### 3. Context Management

```python
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum

# Implement context-aware memory storage
async def store_conversation_memory(
    facts: list[str],
    session_id: str,
    user_id: str
):
    if facts:
        memories = [
            ClientMemoryRecord(
                text=fact,
                memory_type=MemoryTypeEnum.EPISODIC,
                session_id=session_id,
                user_id=user_id
            )
            for fact in facts
        ]
        await client.create_long_term_memory(memories)
```

## Configuration Reference

### Environment Variables

```bash
# Client configuration
MEMORY_SERVER_URL=http://localhost:8000

# Connection settings
MEMORY_TIMEOUT=30

# Default settings
DEFAULT_NAMESPACE=production
```

### MemoryClientConfig Options

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

config = MemoryClientConfig(
    base_url="http://localhost:8000",       # Required: Server URL
    timeout=30.0,                           # HTTP timeout (seconds)
    default_namespace="production",          # Default namespace for all ops
    default_model_name="gpt-4o",            # Model for token counting
    default_context_window_max=128000,       # Override context window
)

async with MemoryAPIClient(config) as client:
    # Use client...
    pass
```

The Python SDK makes it easy to add sophisticated memory capabilities to any AI application, with minimal setup and maximum flexibility. Use the tool integrations for LLM-driven memory, direct API calls for code-driven approaches, or combine both patterns for hybrid solutions.
