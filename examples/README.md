# Examples

This directory contains example implementations showing how to use the Agent Memory Server.

## Prerequisites

### Install Dependencies

The examples require additional dependencies not included in the base installation. Install them with:

```bash
uv sync --group examples
```

Or if you're using pip:

```bash
pip install openai langchain langchain-core langchain-openai langchain-community python-dotenv tavily-python redis
```

### Environment Variables

All examples require:
- `OPENAI_API_KEY` - Required for OpenAI ChatGPT

Some examples use additional environment variables documented in their sections below.

### Running the Memory Server

Examples expect the Agent Memory Server to be running. Start it with:

```bash
uv run agent-memory api
```

Or using Docker:

```bash
docker-compose up
```

---

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
2. **get_or_create_working_memory** - Check current working memory session
3. **lazily_create_long_term_memory** - Store memories that will be promoted to long-term storage later
4. **update_working_memory_data** - Store/update session-specific data like trip plans
5. **eagerly_create_long_term_memory** - Create long-term memories directly for immediate storage
6. **get_long_term_memory** - Retrieve specific memories by ID
7. **edit_long_term_memory** - Update existing memories
8. **delete_long_term_memories** - Remove memories
9. **get_current_datetime** - Get current date/time for temporal context

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

## Memory Editing Agent (`memory_editing_agent.py`)

A conversational assistant that demonstrates comprehensive memory editing capabilities:

### Core Features
- **Memory Editing Workflow**: Complete lifecycle of creating, searching, editing, and deleting memories through natural conversation
- **All Memory Tools**: Utilizes all available memory management tools including the new editing capabilities
- **Realistic Scenarios**: Shows common patterns like correcting information, updating preferences, and managing outdated data
- **Interactive Demo**: Both automated demo and interactive modes for exploring memory editing

### Available Tools
The memory editing agent uses all memory tools to demonstrate comprehensive memory management:

1. **search_memory** - Find existing memories using natural language queries
2. **get_long_term_memory** - Retrieve specific memories by ID for detailed review
3. **lazily_create_long_term_memory** - Store memories that will be promoted to long-term storage later
4. **eagerly_create_long_term_memory** - Create long-term memories directly for immediate storage
5. **edit_long_term_memory** - Update existing memories with corrections or new information
6. **delete_long_term_memories** - Remove memories that are no longer relevant or accurate
7. **get_or_create_working_memory** - Check current working memory session
8. **update_working_memory_data** - Store session-specific data
9. **get_current_datetime** - Get current date/time for temporal context

### Common Memory Editing Scenarios
- **Corrections**: "Actually, I work at Microsoft, not Google" → Search for job memory, edit company name
- **Updates**: "I got promoted to Senior Engineer" → Find job memory, update title and add promotion date
- **Preference Changes**: "I prefer tea over coffee now" → Search beverage preferences, update from coffee to tea
- **Life Changes**: "I moved to Seattle" → Find location memories, update address/city information
- **Information Cleanup**: "Delete that old job information" → Search and remove outdated employment data

### Usage

```bash
# Interactive mode (default)
python memory_editing_agent.py

# Automated demo showing memory editing scenarios
python memory_editing_agent.py --demo

# With custom session
python memory_editing_agent.py --session-id my_session --user-id alice

# With custom memory server
python memory_editing_agent.py --memory-server-url http://localhost:8001
```

### Environment Variables
- `OPENAI_API_KEY` - Required for OpenAI ChatGPT
- `MEMORY_SERVER_URL` - Memory server URL (default: http://localhost:8000)

### Key Implementation Details
- **Memory-First Approach**: Always searches for existing memories before creating new ones to avoid duplicates
- **Intelligent Updates**: Provides context-aware suggestions for editing vs creating new memories
- **Error Handling**: Robust handling of memory operations with clear user feedback
- **Natural Conversation**: Explains memory actions as part of natural dialogue flow
- **Comprehensive Coverage**: Demonstrates all memory CRUD operations through realistic conversation patterns

### Demo Conversation Flow
The automated demo shows a realistic conversation where the agent:
1. **Initial Information**: User shares basic profile information (name, job, preferences)
2. **Corrections**: User corrects previously shared information (job company change)
3. **Updates**: User provides updates to existing information (promotion, new title)
4. **Multiple Changes**: User updates multiple pieces of information at once (location, preferences)
5. **Information Retrieval**: User asks what the agent remembers to verify updates
6. **Ongoing Updates**: User continues to update information (new job level)
7. **Memory Management**: User requests specific memory operations (show/delete specific memories)

This example provides a complete reference for implementing memory editing in conversational AI applications.

## AI Tutor (`ai_tutor.py`)

A functional tutor: runs quizzes, stores results as episodic memories, tracks weak concepts as semantic memories, suggests next practice, and summarizes recent activity.

### Usage

```bash
python ai_tutor.py --demo
python ai_tutor.py --user-id student --session-id s1
```

### Highlights
- **Episodic**: Per-question results with `event_date` and `topics=["quiz", topic, concept]`
- **Semantic**: Weak concepts tracked with `topics=["weak_concept", topic, concept]`
- **Guidance**: `practice-next` and `summary` commands
