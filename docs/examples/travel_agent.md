# 🧳 Travel Agent

**File**: [`examples/travel_agent.py`](https://github.com/redis/agent-memory-server/blob/main/examples/travel_agent.py)

A comprehensive travel assistant that demonstrates the most complete integration patterns.

## Key Features

- **Automatic Tool Discovery**: Uses `MemoryAPIClient.get_all_memory_tool_schemas()` to automatically discover and integrate all available memory tools
- **Unified Tool Resolution**: Leverages `client.resolve_tool_call()` to handle all memory tool calls uniformly across different LLM providers
- **Working Memory Management**: Session-based conversation state and structured memory storage
- **Long-term Memory**: Persistent memory storage with semantic, keyword, and hybrid search capabilities
- **Optional Web Search**: Cached web search using Tavily API with Redis caching

## Available Tools

The travel agent automatically discovers and uses all memory tools:

1. **search_memory** — Search through previous conversations and stored information (supports `semantic`, `keyword`, and `hybrid` search modes)
2. **get_or_create_working_memory** — Check current working memory session
3. **lazily_create_long_term_memory** — Store important information as structured memories (promoted to long-term storage later)
4. **update_working_memory_data** — Store/update session-specific data like trip plans
5. **get_long_term_memory** — Retrieve a specific long-term memory by ID
6. **eagerly_create_long_term_memory** — Create long-term memories directly for immediate storage
7. **edit_long_term_memory** — Update existing long-term memories
8. **delete_long_term_memories** — Remove long-term memories
9. **get_current_datetime** — Get current UTC datetime for grounding relative time expressions
10. **web_search** (optional) — Search the internet for current travel information

## Usage Examples

```bash
# Basic interactive usage
cd examples
python travel_agent.py

# Automated demo showing capabilities
python travel_agent.py --demo

# With custom configuration
python travel_agent.py --session-id my_trip --user-id john_doe --memory-server-url http://localhost:8001
```

## Environment Setup

```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional (for web search)
export TAVILY_API_KEY="your-tavily-key"
export REDIS_URL="redis://localhost:6379"
```

## Key Implementation Patterns

```python
# Tool auto-discovery
memory_tools = MemoryAPIClient.get_all_memory_tool_schemas()

# Unified tool resolution for any provider
result = await client.resolve_tool_call(
    tool_call=provider_tool_call,
    session_id=session_id
)

if result["success"]:
    print(result["formatted_response"])
```
