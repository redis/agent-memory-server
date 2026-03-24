# Agent Examples

This section provides comprehensive working examples that demonstrate real-world usage patterns of the Redis Agent Memory Server. Each example showcases different aspects of memory management, from basic conversation storage to advanced memory editing workflows.

## Interactive Technical Guide (Notebook)

**File:** `examples/agent_memory_server_interactive_guide.ipynb`

The Interactive Technical Guide is a comprehensive, cell-by-cell walkthrough of the
Agent Memory Server. Unlike the standalone scripts below, this notebook is designed
for hands-on exploration in an interactive environment — each section builds on the
previous one, with executable code cells and inline explanations.

### Format

The notebook uses the [Jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html),
where cells are delimited by `# %%` markers. You can open it directly in:

- **VS Code** (with the Jupyter extension)
- **JetBrains IDEs** (PyCharm, DataSpell)
- **JupyterLab** (via `jupytext --to notebook` conversion)

### Prerequisites

Before running the notebook, ensure the following services are available:

1. **Redis 8** running locally (via `docker-compose up redis -d`)
2. **Agent Memory Server** running in development mode:
   ```bash
   DISABLE_AUTH=true uv run agent-memory api --task-backend=asyncio
   ```
3. **Environment variables** configured:
   ```bash
   export OPENAI_API_KEY=<your-key>
   export DISABLE_AUTH=true
   ```
4. **Python dependencies** installed:
   ```bash
   pip install agent-memory-client httpx openai
   ```

### Sections

The guide is organized into twelve sections, each covering a distinct aspect of the
memory system:

| Section | Topic | Description |
|---------|-------|-------------|
| 1 | Problem & Solution | Introduces the statelessness problem and the two-tier memory architecture (working memory and long-term memory). |
| 2 | Quick Start | Configures the SDK client, verifies server connectivity, and explains the Redis key structure used internally. |
| 3 | Integration Patterns Overview | Describes the three integration patterns — Code-Driven, LLM-Driven, and Background Extraction — and when to use each. |
| 4 | Pattern 1: Code-Driven (SDK) | Demonstrates deterministic memory operations using the `agent_memory_client` SDK: creating sessions, seeding long-term memories, retrieving context with `memory_prompt()`, and storing conversations. |
| 5 | Pattern 2: LLM-Driven (Tools) | Shows how to expose memory operations as OpenAI-compatible tool schemas, let the model decide when to store or retrieve memories, and resolve tool calls with `resolve_tool_call()`. |
| 6 | Pattern 3: Background Extraction | Covers automatic memory extraction from conversations using discrete and custom extraction strategies, including debounce configuration. |
| 7 | Combining Patterns | Discusses how production applications typically combine multiple patterns (e.g., code-driven context hydration with LLM-driven storage). |
| 8 | Working Memory Deep Dive | Explores session management, working memory summarization (trigger conditions, token allocation, progressive summarization), and structured data storage. |
| 9 | Long-Term Memory Deep Dive | Covers semantic, keyword, and hybrid search modes; `hybrid_alpha` tuning; recency boost configuration; and filtered search by topics, entities, or timestamps. |
| 10 | Memory Types & Contextual Grounding | Explains semantic vs. episodic memory types, event dating, and how contextual grounding enriches stored memories. |
| 11 | Extraction Strategy Comparison | Provides a side-by-side comparison of discrete (default) and custom extraction strategies using a vehicle rental scenario. |
| 12 | Production Considerations | Covers authentication, background task workers, and LLM provider configuration for deployment. |

### Key concepts demonstrated

- **Two-tier memory architecture**: Working memory (session-scoped, ephemeral) automatically promotes structured memories to long-term storage (persistent, searchable).
- **Search mode comparison**: Executable cells that run the same query across `semantic`, `keyword`, and `hybrid` search modes, illustrating how each mode surfaces different results.
- **Recency boost tuning**: Shows how to adjust or disable the recency weight to control whether newer memories are prioritized over older, semantically relevant ones.
- **Tool schema integration**: Generates OpenAI-compatible function schemas from the SDK and demonstrates the full tool-call lifecycle (schema → LLM decision → execution → response).
- **Extraction strategy configuration**: Compares what discrete vs. custom extraction strategies produce from the same input conversation.

### Running the guide

```bash
cd examples

# VS Code: open the file directly — the Jupyter extension recognizes percent-format cells
code agent_memory_server_interactive_guide.ipynb

# JupyterLab: convert to standard notebook format first
jupytext --to notebook agent_memory_server_interactive_guide.ipynb
jupyter lab agent_memory_server_interactive_guide.ipynb
```

---

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

1. **search_memory** - Search through previous conversations and stored information (supports `semantic`, `keyword`, and `hybrid` search modes)
2. **get_or_create_working_memory** - Check current working memory session
3. **lazily_create_long_term_memory** - Store important information as structured memories (promoted to long-term storage later)
4. **update_working_memory_data** - Store/update session-specific data like trip plans
5. **get_long_term_memory** - Retrieve a specific long-term memory by ID
6. **eagerly_create_long_term_memory** - Create long-term memories directly for immediate storage
7. **edit_long_term_memory** - Update existing long-term memories
8. **delete_long_term_memories** - Remove long-term memories
9. **get_current_datetime** - Get current UTC datetime for grounding relative time expressions
10. **web_search** (optional) - Search the internet for current travel information

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
    messages=context["messages"]
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

1. **search_memory** - Find existing memories using natural language (supports `semantic`, `keyword`, and `hybrid` search modes)
2. **get_long_term_memory** - Retrieve specific memories by ID
3. **lazily_create_long_term_memory** - Store new information (promoted to long-term storage later)
4. **eagerly_create_long_term_memory** - Create long-term memories directly for immediate storage
5. **edit_long_term_memory** - Update existing memories
6. **delete_long_term_memories** - Remove outdated information
7. **get_or_create_working_memory** - Check current working memory session
8. **update_working_memory_data** - Store/update session-specific data
9. **get_current_datetime** - Get current UTC datetime for grounding relative time expressions

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

## 🔗 LangChain Integration Example

**File**: [`examples/langchain_integration_example.py`](https://github.com/redis/agent-memory-server/blob/main/examples/langchain_integration_example.py)

Demonstrates how to use the `agent_memory_client` LangChain integration to create memory-enabled agents **without manual tool wrapping**.

### Core Concept

Uses `get_memory_tools()` from `agent_memory_client.integrations.langchain` to automatically generate LangChain-compatible tools, then creates an agent with `create_agent` (LangGraph-based).

### Key Features

- **Automatic Tool Generation**: No manual `@tool` wrappers needed — `get_memory_tools()` handles it
- **Modern LangGraph Agent**: Uses `create_agent` from `langchain.agents` (not the deprecated `AgentExecutor`)
- **State Persistence**: Demonstrates `MemorySaver` checkpointer for multi-turn conversations
- **Search Modes**: Supports `semantic`, `keyword`, and `hybrid` search via `search_memory`

### Usage Examples

```bash
cd examples
python langchain_integration_example.py
```

### Key Implementation Pattern

```python
from agent_memory_client import create_memory_client
from agent_memory_client.integrations.langchain import get_memory_tools
from langchain.agents import create_agent

memory_client = await create_memory_client("http://localhost:8000")
tools = get_memory_tools(memory_client=memory_client, session_id="session", user_id="user")

agent = create_agent(
    ChatOpenAI(model="gpt-4o"), tools,
    system_prompt="You are a helpful assistant with persistent memory."
)

result = await agent.ainvoke({"messages": [("human", "Remember I love pizza")]})
print(result["messages"][-1].content)
```

---

## 📊 Recent Messages Limit Demo

**File**: [`examples/recent_messages_limit_demo.py`](https://github.com/redis/agent-memory-server/blob/main/examples/recent_messages_limit_demo.py)

Demonstrates the `recent_messages_limit` parameter for efficiently retrieving only the most recent N messages from working memory.

### Core Concept

When working memory grows large, retrieving all messages is expensive. The `recent_messages_limit` parameter lets you fetch only the N most recent messages, useful for context windows and UI displays.

### Key Features

- **Efficient Retrieval**: Fetch only the messages you need instead of the full history
- **SDK & HTTP Integration**: Uses `agent_memory_client` to store working memory and raw HTTP requests to retrieve it
- **Multiple Scenarios**: Tests various limits (3, 5, 20, 2) to show how `recent_messages_limit` changes the results
- **Direct API Verification**: Uses the raw HTTP API for retrieval so you can inspect the exact responses from the server

### Usage Examples

```bash
cd examples
python recent_messages_limit_demo.py
```

### Key Implementation Pattern

```python
from agent_memory_client import create_memory_client

client = await create_memory_client(base_url="http://localhost:8000")

# Get only the 3 most recent messages
memory = await client.get_working_memory(
    session_id="my-session",
    namespace="demo",
    context_window_max=3
)
```

---

## Getting Started with Examples

### 1. Prerequisites

```bash
# Install dependencies
cd /path/to/agent-memory-server
uv sync --all-extras

# Start memory server (disable auth for local development)
DISABLE_AUTH=true uv run agent-memory api --task-backend=asyncio

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

# LangChain integration (requires langchain, langchain_openai)
python langchain_integration_example.py

# Recent messages limit feature demo
python recent_messages_limit_demo.py
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

- **Start with the Interactive Guide**: Best for learning the full system end-to-end in an exploratory environment
- **Start with Travel Agent**: Most comprehensive standalone example showing all features
- **Explore Memory Editing**: Learn advanced memory management patterns
- **Study Code Patterns**: Each example demonstrates different architectural approaches
- **Build Your Own**: Use examples as templates for your specific use case

All examples include detailed inline documentation and can serve as starting points for building production memory-enhanced AI applications.
