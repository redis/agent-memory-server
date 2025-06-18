# Examples

This directory contains example implementations showing how to use the Agent Memory Server.

## Travel Agent (`travel_agent.py`)

A comprehensive travel assistant that demonstrates:

### Core Features
- **Automatic Tool Discovery**: Uses `MemoryAPIClient.get_all_memory_tool_schemas()` to automatically discover and integrate all available memory tools
- **Unified Tool Resolution**: Leverages `client.resolve_tool_call()` to handle all memory tool calls uniformly across different LLM providers
- **Working Memory Management**: Session-based conversation state and structured memory storage
- **Long-term Memory**: Persistent memory storage and semantic search capabilities
- **Optional Web Search**: Cached web search using Tavily API with Redis caching

### Available Tools
The travel agent automatically discovers and uses all memory tools available from the client:

1. **search_memory** - Search through previous conversations and stored information
2. **get_working_memory** - Check current session state, stored memories, and data
3. **add_memory_to_working_memory** - Store important information as structured memories
4. **update_working_memory_data** - Store/update session-specific data like trip plans

Plus optional:
- **web_search** - Search the internet for current travel information (requires TAVILY_API_KEY)

### Usage

```bash
# Basic usage
python travel_agent.py

# With custom session
python travel_agent.py --session-id my_trip --user-id john_doe

# With custom memory server
python travel_agent.py --memory-server-url http://localhost:8001
```

### Environment Variables
- `OPENAI_API_KEY` - Required for OpenAI ChatGPT
- `TAVILY_API_KEY` - Optional for web search functionality
- `MEMORY_SERVER_URL` - Memory server URL (default: http://localhost:8000)
- `REDIS_URL` - Redis URL for caching (default: redis://localhost:6379)

### Key Implementation Details
- **Tool Auto-Discovery**: Uses the client's built-in tool management for maximum compatibility
- **Provider Agnostic**: Tool resolution works with OpenAI, Anthropic, and other LLM providers
- **Error Handling**: Robust error handling for tool calls and network issues
- **Logging**: Comprehensive logging shows which tools are available and being used
