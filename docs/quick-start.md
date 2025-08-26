# Quick Start Guide

Get up and running with Redis Agent Memory Server in 5 minutes. This guide shows you how to build memory-enabled AI applications using the Python SDK, with REST API examples as backup.

## What You'll Learn

By the end of this guide, you'll:
- Have a running memory server with authentication disabled for development
- Use the Python SDK to store and search memories seamlessly
- Build memory-enhanced conversations with OpenAI or Anthropic
- Understand the difference between working and long-term memory

## Prerequisites

- Python 3.12 (for the memory server)
- Docker (for Redis)
- 5 minutes

## Step 1: Install Dependencies

Install the Python SDK and memory server:

```bash
# Install the Python SDK
pip install agent-memory-client

# Install uv for running the server
pip install uv

# Clone the repository to run the server locally
git clone https://github.com/redis/redis-memory-server.git
cd redis-memory-server

# Install server dependencies
uv sync
```

## Step 2: Start Redis

Start Redis using Docker:

```bash
# Start Redis with RediSearch module
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

# Or use the provided docker-compose
docker-compose up redis -d
```

## Step 3: Configure for Development

Set up environment variables for development (no authentication):

```bash
# Create a .env file
cat > .env << EOF
# Disable authentication for development
DISABLE_AUTH=true

# Redis connection
REDIS_URL=redis://localhost:6379

# Enable all memory features
LONG_TERM_MEMORY=true
ENABLE_DISCRETE_MEMORY_EXTRACTION=true

# AI API keys (add your own)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
EOF
```

**Note**: You'll need API keys for OpenAI or Anthropic to use AI features like memory extraction and search optimization.

## Step 4: Start the Server

Start the REST API server:

```bash
# Start the API server (runs on port 8000)
uv run agent-memory api

# In another terminal, start the task worker for background processing
uv run agent-memory task-worker
```

Your server is now running at `http://localhost:8000`!

Check the API docs at: `http://localhost:8000/docs`

## Step 5: Your First Memory-Enhanced App

Now let's build a memory-enhanced chat application using the Python SDK:

```python
import asyncio
import openai
from agent_memory_client import MemoryAPIClient

# Setup clients
memory_client = MemoryAPIClient(base_url="http://localhost:8000")
openai_client = openai.AsyncClient(api_key="your-openai-key")

async def chat_with_memory(message: str, session_id: str):
    # Get memory-enriched context
    context = await memory_client.memory_prompt(
        query=message,
        session={
            "session_id": session_id,
            "model_name": "gpt-4o"
        },
        long_term_search={
            "text": message,
            "limit": 5
        }
    )

    # Send to OpenAI with context
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
    await memory_client.set_working_memory(session_id, conversation)

    return response.choices[0].message.content

# Try it out!
async def main():
    # First conversation
    response1 = await chat_with_memory(
        "Hi! I love Italian food, especially pasta like carbonara",
        "my-session-123"
    )
    print(f"AI: {response1}")

    # Later conversation - AI will remember your food preferences!
    response2 = await chat_with_memory(
        "Can you recommend a good recipe for dinner?",
        "my-session-123"
    )
    print(f"AI: {response2}")

asyncio.run(main())
```

The AI will automatically remember your food preferences and give personalized recipe recommendations!

## Step 6: Create Persistent Memories

Store long-term facts that persist across all sessions:

```python
# Store user preferences that persist across sessions
await memory_client.create_long_term_memories([
    {
        "text": "User works as a software engineer specializing in Python and web development",
        "memory_type": "semantic",
        "topics": ["career", "programming", "python"],
        "entities": ["software engineer", "Python", "web development"],
        "user_id": "alice"
    },
    {
        "text": "User prefers morning meetings and hates scheduling calls after 4 PM",
        "memory_type": "semantic",
        "topics": ["scheduling", "preferences", "work"],
        "entities": ["morning meetings", "4 PM"],
        "user_id": "alice"
    }
])
```

## Step 7: Search Your Memories

Search across all stored memories with semantic similarity:

```python
# Search for work-related information
results = await memory_client.search_long_term_memory(
    text="user work preferences and schedule",
    user_id="alice",
    limit=5
)

for memory in results.memories:
    print(f"Relevance: {memory.relevance_score:.2f}")
    print(f"Memory: {memory.text}")
    print(f"Topics: {', '.join(memory.topics or [])}")
```

## Step 8: Tool Integration (Advanced)

For more advanced use cases, use automatic tool integration with OpenAI:

```python
# Get OpenAI tool schemas
memory_tools = memory_client.get_openai_tool_schemas()

# Chat with automatic memory tools
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Remember that I'm allergic to nuts"}],
    tools=memory_tools,
    tool_choice="auto"
)

# Let the AI decide when to store memories
if response.choices[0].message.tool_calls:
    tool_results = await memory_client.resolve_openai_tool_calls(
        tool_calls=response.choices[0].message.tool_calls,
        session_id="my-session"
    )
    print("AI automatically stored your allergy information!")
```

## Alternative: REST API Usage

If you prefer REST API calls instead of the Python SDK:

<details>
<summary>Click to see REST API examples</summary>

### Store Working Memory

```bash
curl -X PUT "http://localhost:8000/v1/working-memory/my-session" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I love Italian food, especially pasta"}
    ],
    "memories": [{
      "text": "User loves Italian food, especially pasta",
      "memory_type": "semantic",
      "topics": ["food", "preferences"]
    }]
  }'
```

### Search Memories

```bash
curl -X POST "http://localhost:8000/v1/long-term-memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "user food preferences",
    "limit": 5
  }'
```

### Memory-Enriched Prompts

```bash
curl -X POST "http://localhost:8000/v1/memory/prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Recommend a recipe",
    "session": {"session_id": "my-session", "model_name": "gpt-4o"},
    "long_term_search": {"text": "user food preferences", "limit": 3}
  }'
```

</details>

## Using MCP Interface (Optional)

If you want to use the MCP interface with Claude Desktop:

### Configure Claude Desktop

**Note**: You don't need to manually start the MCP server. Claude Desktop will automatically start and manage the server process when needed.

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "redis-memory-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/redis-memory-server",
        "run",
        "agent-memory",
        "mcp",
        "--mode",
        "stdio"
      ]
    }
  }
}
```

Now Claude can use memory tools directly in conversations!

### Alternative: SSE Mode (Advanced)

For web-based MCP clients, you can use SSE mode, but this requires manually starting the server:

```bash
# Only needed for SSE mode
uv run agent-memory mcp --mode sse --port 9000
```

**Recommendation**: Use stdio mode with Claude Desktop as it's much simpler to set up.

## Understanding Memory Types

You've just worked with both types of memory:

### Working Memory
- **Scope**: Session-specific
- **Lifetime**: 1 hour (configurable TTL)
- **Use case**: Active conversation state
- **Auto-promotion**: Structured memories automatically move to long-term storage

### Long-Term Memory
- **Scope**: Cross-session, persistent
- **Lifetime**: Permanent until deleted
- **Use case**: User preferences, facts, knowledge
- **Search**: Semantic vector search with advanced filtering

## Next Steps

Now that you have the basics working, explore these advanced features:

### 🔍 **Advanced Search**
- Try filtering by topics, entities, or time ranges
- Experiment with recency boost and query optimization
- See [Memory Types Guide](memory-types.md) for detailed examples

### ✏️ **Memory Editing**
- Update existing memories with corrections
- Add more context to sparse memories
- See [Memory Editing Guide](memory-editing.md)

### 🔒 **Production Setup**
- Enable authentication (OAuth2/JWT or token-based)
- Configure background tasks and memory compaction
- See [Authentication Guide](authentication.md) and [Configuration Guide](configuration.md)

### 🚀 **Advanced Features**
- **Query Optimization**: Improve search accuracy with configurable models
- **Contextual Grounding**: Resolve pronouns and references in extracted memories
- **Recency Boost**: Time-aware memory ranking
- **Vector Store Backends**: Use different storage backends (Pinecone, Chroma, etc.)

## Common Issues

**"Redis connection failed"**
- Ensure Redis is running: `docker ps | grep redis`
- Check Redis URL: `redis://localhost:6379`

**"API key required"**
- Add your OpenAI or Anthropic API key to `.env`
- Or disable AI features temporarily

**"Module 'redisvl' not found"**
- Run: `uv sync` (redisvl is a required dependency, not optional)
- If still failing, try: `uv add redisvl>=0.6.0`

**"Background tasks not processing"**
- Make sure the task worker is running: `uv run agent-memory task-worker`
- Check logs for worker errors

## Get Help

- **API Documentation**: Visit `http://localhost:8000/docs`
- **Configuration Guide**: [Configuration](configuration.md)
- **Memory Types**: [Memory Types Guide](memory-types.md)
- **GitHub Issues**: Report problems or ask questions

## What's Next?

You now have a working AI agent memory system! Your memories will:
- ✅ Persist across sessions
- ✅ Be searchable with semantic similarity
- ✅ Automatically extract context from conversations
- ✅ Provide relevant context to AI responses

The memory server learns and improves over time as you add more memories and interactions. Start building your AI agent and let it develop a persistent memory that gets smarter with every conversation!

## Complete Example Application

Here's a complete memory-enhanced chatbot that learns about users over time:

```python
import asyncio
import openai
from agent_memory_client import MemoryAPIClient

class MemoryEnhancedChatbot:
    def __init__(self, memory_url: str, openai_api_key: str):
        self.memory = MemoryAPIClient(base_url=memory_url)
        self.openai = openai.AsyncClient(api_key=openai_api_key)

    async def chat(self, message: str, user_id: str, session_id: str):
        # Get relevant context from memory
        context = await self.memory.memory_prompt(
            query=message,
            session={
                "session_id": session_id,
                "user_id": user_id,
                "model_name": "gpt-4o"
            },
            long_term_search={
                "text": message,
                "user_id": user_id,
                "limit": 5
            }
        )

        # Generate AI response with memory context
        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=context.messages + [{"role": "user", "content": message}]
        )

        ai_response = response.choices[0].message.content

        # Store the conversation for future reference
        conversation = {
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ai_response}
            ]
        }
        await self.memory.set_working_memory(session_id, conversation)

        return ai_response

# Usage example
async def main():
    chatbot = MemoryEnhancedChatbot(
        memory_url="http://localhost:8000",
        openai_api_key="your-openai-key"
    )

    # Simulate a conversation that builds memory over time
    user_id = "alice"
    session_id = "session-1"

    # First interaction - establish preferences
    response1 = await chatbot.chat(
        "Hi! I'm Alice. I love Italian cuisine and I'm vegetarian.",
        user_id, session_id
    )
    print(f"AI: {response1}")

    # Later interaction - AI remembers preferences
    response2 = await chatbot.chat(
        "What should I cook for dinner tonight?",
        user_id, session_id
    )
    print(f"AI: {response2}")  # Will suggest vegetarian Italian dishes!

    # Even later - persistent memory across sessions
    new_session = "session-2"
    response3 = await chatbot.chat(
        "I'm having friends over. Any meal suggestions?",
        user_id, new_session
    )
    print(f"AI: {response3}")  # Still remembers Alice is vegetarian!

asyncio.run(main())
```

This chatbot automatically learns and remembers user preferences, making every conversation more personalized!

Happy memory building! 🧠✨
