# 🧠 Memory Prompt Agent

**File**: [`examples/memory_prompt_agent.py`](https://github.com/redis/agent-memory-server/blob/main/examples/memory_prompt_agent.py)

Demonstrates the simplified memory prompt feature for context-aware conversations without manual tool management.

## Core Concept

Uses `client.memory_prompt()` to automatically retrieve relevant memories and enrich prompts with contextual information.

## How It Works

1. **Store Messages**: All conversation messages stored in working memory
2. **Memory Prompt**: `memory_prompt()` retrieves relevant context automatically
3. **Enriched Context**: Memory context combined with system prompt
4. **LLM Generation**: Enhanced context sent to LLM for personalized responses

## Usage Examples

```bash
cd examples
python memory_prompt_agent.py

# With custom session
python memory_prompt_agent.py --session-id my_session --user-id jane_doe
```

## Key Implementation Pattern

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
