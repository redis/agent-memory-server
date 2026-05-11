# Agent Examples

Working examples that demonstrate real-world usage patterns of the Agent
Memory Server. Each example focuses on a different aspect of memory
management — from basic conversation storage to advanced editing
workflows — and runs as a self-contained script (or notebook).

If you're just trying things out, **start with the
[Interactive Technical Guide](agent_memory_server_interactive_guide.ipynb)**:
it walks through the whole system end-to-end in a notebook. If you'd
rather see one complete agent first, start with the
[**Travel Agent**](travel_agent.md) — it uses every memory tool the
server exposes.

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **[Interactive Guide](agent_memory_server_interactive_guide.ipynb)**

    ---

    Twelve-section, cell-by-cell walkthrough of the full memory system.

-   :material-airplane:{ .lg .middle } **[Travel Agent](travel_agent.md)**

    ---

    Most comprehensive example. Auto-discovers and uses every memory tool.

-   :material-message-text:{ .lg .middle } **[Memory Prompt Agent](memory_prompt_agent.md)**

    ---

    Simplified pattern using `memory_prompt()` for context hydration.

-   :material-pencil:{ .lg .middle } **[Memory Editing Agent](memory_editing_agent.md)**

    ---

    Full editing lifecycle: create, search, correct, update, delete.

-   :material-school:{ .lg .middle } **[AI Tutor](ai_tutor.md)**

    ---

    Episodic + semantic memory for learning tracking and weak-concept practice.

-   :material-link-variant:{ .lg .middle } **[LangChain Integration](langchain.md)**

    ---

    Memory-enabled LangChain agents via `get_memory_tools()`.

-   :material-counter:{ .lg .middle } **[Recent Messages Limit Demo](recent_messages_limit_demo.md)**

    ---

    Efficient retrieval of the most recent N messages from working memory.

</div>

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

### 2. Run an Example

```bash
cd examples

# Most comprehensive standalone agent
python travel_agent.py --demo

# Memory editing workflows
python memory_editing_agent.py --demo

# Simplified memory prompts
python memory_prompt_agent.py

# Learning tracking
python ai_tutor.py --demo

# LangChain integration
python langchain_integration_example.py

# Recent messages limit feature
python recent_messages_limit_demo.py
```

### 3. Customize and Extend

Each example is designed to be:

- **Self-contained**: Runs independently with minimal setup
- **Configurable**: Supports custom sessions, users, and server URLs
- **Educational**: Well-commented code showing best practices
- **Production-ready**: Robust error handling and logging

### 4. Shared Implementation Patterns

Most examples follow the same shape:

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

## Where to Next

- **Learning the system end-to-end?** → [Interactive Guide](agent_memory_server_interactive_guide.ipynb)
- **Most complete production-style agent?** → [Travel Agent](travel_agent.md)
- **Editing patterns?** → [Memory Editing Agent](memory_editing_agent.md) and the [Memory editing how-to](../user_guide/how_to_guides/memory_editing.md)
- **Building your own?** Each example is a template — clone, adapt, ship.
