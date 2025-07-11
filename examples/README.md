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

## Memory Prompt Agent (`memory_prompt_agent.py`)

A conversational assistant that demonstrates the memory prompt feature:

### Core Features
- **Memory Prompt Integration**: Uses `client.memory_prompt()` to automatically retrieve relevant memories
- **Context-Aware Responses**: Combines system prompt with memory-enriched context
- **Simplified Memory Management**: No manual history management - memories are automatically retrieved
- **Personalized Interactions**: Provides contextual responses based on conversation history

### How It Works
1. **Store Messages**: All user and assistant messages are stored in working memory
2. **Memory Prompt**: For each turn, `memory_prompt()` retrieves relevant context from both working memory and long-term memories
3. **Enriched Context**: The memory prompt results are combined with the system prompt
4. **LLM Generation**: The enriched context is sent to the LLM for response generation

### Usage

```bash
# Basic usage
python memory_prompt_agent.py

# With custom session
python memory_prompt_agent.py --session-id my_session --user-id jane_doe

# With custom memory server
python memory_prompt_agent.py --memory-server-url http://localhost:8001
```

### Environment Variables
- `OPENAI_API_KEY` - Required for OpenAI ChatGPT
- `MEMORY_SERVER_URL` - Memory server URL (default: http://localhost:8000)

### Key Implementation Details
- **Automatic Memory Retrieval**: Uses `memory_prompt()` to get relevant memories without manual management
- **Context Enrichment**: Combines system prompt with formatted memory context
- **Simplified Flow**: No function calling - just enriched prompts for more contextual responses
- **Personalization**: Naturally incorporates user preferences and past conversations
