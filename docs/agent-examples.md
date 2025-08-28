# Agent Examples

This section provides comprehensive working examples that demonstrate real-world usage patterns of the Redis Agent Memory Server. Each example showcases different aspects of memory management, from basic conversation storage to advanced memory editing workflows.

## 🧳 Travel Agent

**File**: [`examples/travel_agent.py`](https://github.com/redis/agent-memory-server/blob/main/examples/travel_agent.py)

A comprehensive travel assistant that demonstrates the most complete integration patterns.

### Key Features

- **Automatic Tool Discovery**: Uses `MemoryAPIClient.get_all_memory_tool_schemas()` to automatically discover and integrate all available memory tools
- **Unified Tool Resolution**: Leverages `client.resolve_tool_call()` to handle all memory tool calls uniformly across different LLM providers
- **Working Memory Management**: Session-based conversation state and structured memory storage
- **Long-term Memory**: Persistent memory storage and semantic search capabilities
- **Optional Web Search**: Cached web search using Tavily API with Redis caching

### Available Tools

The travel agent automatically discovers and uses all memory tools:

1. **search_memory** - Search through previous conversations and stored information
2. **get_or_create_working_memory** - Check current session state, stored memories, and data
3. **add_memory_to_working_memory** - Store important information as structured memories
4. **update_working_memory_data** - Store/update session-specific data like trip plans
5. **web_search** (optional) - Search the internet for current travel information

### Usage Examples

```bash
# Basic interactive usage
cd examples
python travel_agent.py

# Automated demo showing capabilities
python travel_agent.py --demo

# With custom configuration
python travel_agent.py --session-id my_trip --user-id john_doe --memory-server-url http://localhost:8001
```

### Environment Setup

```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional (for web search)
export TAVILY_API_KEY="your-tavily-key"
export REDIS_URL="redis://localhost:6379"
```

### Key Implementation Patterns

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

## 🧠 Memory Prompt Agent

**File**: [`examples/memory_prompt_agent.py`](https://github.com/redis/agent-memory-server/blob/main/examples/memory_prompt_agent.py)

Demonstrates the simplified memory prompt feature for context-aware conversations without manual tool management.

### Core Concept

Uses `client.memory_prompt()` to automatically retrieve relevant memories and enrich prompts with contextual information.

### How It Works

1. **Store Messages**: All conversation messages stored in working memory
2. **Memory Prompt**: `memory_prompt()` retrieves relevant context automatically
3. **Enriched Context**: Memory context combined with system prompt
4. **LLM Generation**: Enhanced context sent to LLM for personalized responses

### Usage Examples

```bash
cd examples
python memory_prompt_agent.py

# With custom session
python memory_prompt_agent.py --session-id my_session --user-id jane_doe
```

### Key Implementation Pattern

```python
# Automatic memory retrieval and context enrichment
context = await client.memory_prompt(
    query=user_message,
    session_id=session_id,
    long_term_search={
        "text": user_message,
        "limit": 5,
        "user_id": user_id
    }
)

# Enhanced prompt with memory context
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=context.messages
)
```

## ✏️ Memory Editing Agent

**File**: [`examples/memory_editing_agent.py`](https://github.com/redis/agent-memory-server/blob/main/examples/memory_editing_agent.py)

Demonstrates comprehensive memory editing capabilities through natural conversation patterns.

### Core Features

- **Memory Editing Workflow**: Complete lifecycle of creating, searching, editing, and deleting memories
- **All Memory Tools**: Uses all available memory management tools including editing capabilities
- **Realistic Scenarios**: Common patterns like corrections, updates, and information cleanup
- **Interactive Demo**: Both automated demo and interactive modes

### Memory Operations Demonstrated

1. **search_memory** - Find existing memories using natural language
2. **get_long_term_memory** - Retrieve specific memories by ID
3. **add_memory_to_working_memory** - Store new information
4. **edit_long_term_memory** - Update existing memories
5. **delete_long_term_memories** - Remove outdated information
6. **get_or_create_working_memory** - Check current session context

### Common Editing Scenarios

```python
# Correction scenario
"Actually, I work at Microsoft, not Google"
# → Search for job memory, edit company name

# Update scenario
"I got promoted to Senior Engineer"
# → Find job memory, update title and add promotion date

# Preference change
"I prefer tea over coffee now"
# → Search beverage preferences, update from coffee to tea

# Information cleanup
"Delete that old job information"
# → Search and remove outdated employment data
```

### Usage Examples

```bash
cd examples

# Interactive mode (explore memory editing)
python memory_editing_agent.py

# Automated demo (see complete workflow)
python memory_editing_agent.py --demo

# Custom configuration
python memory_editing_agent.py --session-id alice_session --user-id alice
```

### Demo Conversation Flow

The automated demo shows a realistic conversation:

1. **Initial Information**: User shares profile (name, job, preferences)
2. **Corrections**: User corrects information (job company change)
3. **Updates**: User provides updates (promotion, new title)
4. **Multiple Changes**: User updates location and preferences
5. **Information Retrieval**: User asks what agent remembers
6. **Ongoing Updates**: Continued information updates
7. **Memory Management**: Specific memory operations (show/delete)

## 🏫 AI Tutor

**File**: [`examples/ai_tutor.py`](https://github.com/redis/agent-memory-server/blob/main/examples/ai_tutor.py)

A functional tutoring system that demonstrates episodic memory for learning tracking and semantic memory for concept management.

### Core Features

- **Quiz Management**: Runs interactive quizzes and stores results
- **Learning Tracking**: Stores quiz results as episodic memories with timestamps
- **Concept Tracking**: Tracks weak concepts as semantic memories
- **Progress Analysis**: Provides summaries and personalized practice suggestions

### Memory Patterns Used

```python
# Episodic: Per-question results with event dates
{
    "text": "User answered 'photosynthesis' question incorrectly",
    "memory_type": "episodic",
    "event_date": "2024-01-15T10:30:00Z",
    "topics": ["quiz", "biology", "photosynthesis"]
}

# Semantic: Weak concepts for targeted practice
{
    "text": "User struggles with photosynthesis concepts",
    "memory_type": "semantic",
    "topics": ["weak_concept", "biology", "photosynthesis"]
}
```

### Usage Examples

```bash
cd examples

# Interactive tutoring session
python ai_tutor.py

# Demo with sample quiz flow
python ai_tutor.py --demo

# Custom student session
python ai_tutor.py --user-id student123 --session-id bio_course
```

### Key Commands

- **Practice**: Start a quiz on specific topics
- **Summary**: Get learning progress summary
- **Practice-next**: Get personalized practice recommendations based on weak areas

## Getting Started with Examples

### 1. Prerequisites

```bash
# Install dependencies
cd /path/to/agent-memory-server
uv install --all-extras

# Start memory server
uv run agent-memory server

# Set required API keys
export OPENAI_API_KEY="your-openai-key"
```

### 2. Run Examples

```bash
cd examples

# Start with the travel agent (most comprehensive)
python travel_agent.py --demo

# Try memory editing workflows
python memory_editing_agent.py --demo

# Explore simplified memory prompts
python memory_prompt_agent.py

# Experience learning tracking
python ai_tutor.py --demo
```

### 3. Customize and Extend

Each example is designed to be:

- **Self-contained**: Runs independently with minimal setup
- **Configurable**: Supports custom sessions, users, and server URLs
- **Educational**: Well-commented code showing best practices
- **Production-ready**: Robust error handling and logging

### 4. Implementation Patterns

Key patterns demonstrated across examples:

```python
# Memory client setup
client = MemoryAPIClient(
    base_url="http://localhost:8000",
    default_namespace=namespace,
    user_id=user_id
)

# Tool integration
tools = MemoryAPIClient.get_all_memory_tool_schemas()
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

# Tool resolution
for tool_call in response.choices[0].message.tool_calls:
    result = await client.resolve_tool_call(
        tool_call=tool_call,
        session_id=session_id
    )
```

## Next Steps

- **Start with Travel Agent**: Most comprehensive example showing all features
- **Explore Memory Editing**: Learn advanced memory management patterns
- **Study Code Patterns**: Each example demonstrates different architectural approaches
- **Build Your Own**: Use examples as templates for your specific use case

All examples include detailed inline documentation and can serve as starting points for building production memory-enhanced AI applications.
