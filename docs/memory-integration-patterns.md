# Memory Patterns

The most common question developers have is: *"How do I actually get memories into and out of my LLM?"* Redis Agent Memory Server provides three distinct patterns for integrating memory with your AI applications, each optimized for different use cases and levels of control.

## Overview of Using Memory

These integration patterns are **not mutually exclusive** and can be combined based on your application's needs. Each pattern excels in different scenarios, but most production systems benefit from using multiple patterns together.

| Pattern | Control | Best For | Memory Flow |
|---------|---------|----------|-------------|
| **🤖 LLM-Driven** | LLM decides | Conversational agents, chatbots | LLM ← tools → Memory |
| **📝 Code-Driven** | Your code decides | Applications, workflows | Code ← SDK → Memory |
| **🔄 Background** | Automatic extraction | Learning systems | Conversation → Auto Extract → Memory |

**Pro tip**: Start with Code-Driven for predictable behavior, then add Background extraction for continuous learning, and finally consider LLM tools for conversational control when needed.

## Pattern 1: LLM-Driven Memory (Tool-Based)

**When to use**: When you want the LLM to decide what to remember and when to retrieve memories through natural conversation.

**How it works**: The LLM has access to memory tools and chooses when to store or search memories based on conversation context.

### Basic Setup

```python
from agent_memory_client import MemoryAPIClient
import openai

# Initialize clients
memory_client = MemoryAPIClient(base_url="http://localhost:8000")
openai_client = openai.AsyncOpenAI()

# Get memory tools for the LLM
memory_tools = MemoryAPIClient.get_all_memory_tool_schemas()

# Give LLM access to memory tools
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with persistent memory. Use the provided tools to remember important information and retrieve relevant context."},
        {"role": "user", "content": "Hi! I'm Alice and I love Italian food, especially pasta carbonara."}
    ],
    tools=memory_tools
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        result = await memory_client.resolve_function_call(
            function_name=tool_call.function.name,
            args=json.loads(tool_call.function.arguments),
            session_id="chat_alice",
            user_id="alice"
        )
        print(f"LLM stored memory: {result}")
```

### Complete Conversation Loop

```python
class LLMMemoryAgent:
    def __init__(self, memory_url: str, session_id: str, user_id: str):
        self.memory_client = MemoryAPIClient(base_url=memory_url)
        self.openai_client = openai.AsyncOpenAI()
        self.session_id = session_id
        self.user_id = user_id
        self.conversation_history = []

    async def chat(self, user_message: str) -> str:
        # Add user message to conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Get memory tools
        tools = MemoryAPIClient.get_all_memory_tool_schemas()

        # Generate response with memory tools
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with persistent memory. Remember important user information and retrieve relevant context when needed."},
                *self.conversation_history
            ],
            tools=tools
        )

        # Handle any tool calls
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                await self.memory_client.resolve_function_call(
                    function_name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments),
                    session_id=self.session_id,
                    user_id=self.user_id
                )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

# Usage
agent = LLMMemoryAgent(
    memory_url="http://localhost:8000",
    session_id="alice_chat",
    user_id="alice"
)

# First conversation
response1 = await agent.chat("I'm planning a trip to Italy next month")
# LLM might store: "User is planning a trip to Italy next month"

# Later conversation
response2 = await agent.chat("What restaurants should I try?")
# LLM retrieves Italy trip context and suggests Italian restaurants
```

### Advantages
- **Natural conversation flow**: Memory operations happen organically
- **User control**: Users can explicitly ask to remember or forget things
- **Contextual decisions**: LLM understands when memory is relevant
- **Flexible**: Works with any conversational pattern

### Disadvantages
- **Token overhead**: Tool schemas consume input tokens
- **Inconsistent behavior**: LLM might not always use memory optimally
- **Cost implications**: More API calls for tool usage
- **Latency**: Additional round trips for tool execution

### Best Practices

```python
# 1. Provide clear system instructions
system_prompt = """
You are an AI assistant with persistent memory capabilities.

When to remember:
- User preferences (food, communication style, etc.)
- Important personal information
- Project details and context
- Recurring topics or interests

When to search memory:
- User asks about previous conversations
- Context would help provide better responses
- User references something from the past

Always be transparent about what you're remembering or have remembered.
"""

# 2. Handle tool call errors gracefully
try:
    result = await memory_client.resolve_function_call(
        function_name=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        session_id=session_id,
        user_id=user_id
    )
except Exception as e:
    logger.warning(f"Memory operation failed: {e}")
    # Continue conversation without failing

# 3. Limit tool schemas to essential ones
essential_tools = [
    memory_client.get_long_term_memory_tool_schema(),
    memory_client.search_long_term_memory_tool_schema(),
    memory_client.create_long_term_memories_tool_schema()
]
```

## Pattern 2: Code-Driven Memory (Programmatic)

**When to use**: When your application logic should control memory operations, or when you need predictable memory behavior.

**How it works**: Your code explicitly manages when to store memories and when to retrieve context, then provides enriched context to the LLM.

### Basic Memory Operations

```python
from agent_memory_client import MemoryAPIClient
from agent_memory_client.models import MemoryRecord

# Initialize client
client = MemoryAPIClient(base_url="http://localhost:8000")

# Store memories programmatically
user_preferences = [
    MemoryRecord(
        text="User Alice prefers email communication over phone calls",
        memory_type="semantic",
        topics=["communication", "preferences"],
        entities=["email", "phone calls"],
        user_id="alice"
    ),
    MemoryRecord(
        text="User Alice works in marketing at TechCorp",
        memory_type="semantic",
        topics=["work", "job", "company"],
        entities=["marketing", "TechCorp"],
        user_id="alice"
    )
]

await client.create_long_term_memories(user_preferences)

# Retrieve relevant context
search_results = await client.search_long_term_memory(
    text="user work and communication preferences",
    filters={"user_id": {"eq": "alice"}},
    limit=5
)

print(f"Found {len(search_results.memories)} relevant memories")
for memory in search_results.memories:
    print(f"- {memory.text}")
```

### Memory-Enriched Conversations

```python
class CodeDrivenAgent:
    def __init__(self, memory_url: str):
        self.memory_client = MemoryAPIClient(base_url=memory_url)
        self.openai_client = openai.AsyncOpenAI()

    async def get_contextual_response(
        self,
        user_message: str,
        user_id: str,
        session_id: str
    ) -> str:
        # 1. Get working memory session (creates if doesn't exist)
        result = await self.memory_client.get_or_create_working_memory(session_id)
        working_memory = result.memory

        # 2. Search for relevant context using session ID
        context_search = await self.memory_client.memory_prompt(
            query=user_message,
            session_id=session_id,
            long_term_search={
                "text": user_message,
                "filters": {"user_id": {"eq": user_id}},
                "limit": 5,
                "recency_boost": True
            }
        )

        # 3. Generate response with enriched context
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=context_search.messages  # Pre-loaded with relevant memories
        )

        # 4. Optionally store the interaction
        await self.store_interaction(user_message, response.choices[0].message.content, user_id, session_id)

        return response.choices[0].message.content

    async def store_interaction(self, user_msg: str, assistant_msg: str, user_id: str, session_id: str):
        """Store important information from the interaction"""
        # Extract key information (you could use LLM or rules for this)
        if "prefer" in user_msg.lower() or "like" in user_msg.lower():
            # Store user preference
            await self.memory_client.create_long_term_memories([
                MemoryRecord(
                    text=f"User expressed: {user_msg}",
                    memory_type="semantic",
                    topics=["preferences"],
                    user_id=user_id,
                    session_id=session_id
                )
            ])

# Usage
agent = CodeDrivenAgent(memory_url="http://localhost:8000")

response = await agent.get_contextual_response(
    user_message="What's a good project management tool?",
    user_id="alice",
    session_id="work_chat"
)
# Response will include context about Alice working in marketing at TechCorp
```

### Batch Operations

```python
# Efficient batch memory storage
batch_memories = []

# Process user data
user_profile = get_user_profile("alice")
for preference in user_profile.preferences:
    batch_memories.append(MemoryRecord(
        text=f"User prefers {preference.value} for {preference.category}",
        memory_type="semantic",
        topics=[preference.category, "preferences"],
        entities=[preference.value],
        user_id="alice"
    ))

# Store all at once
await client.create_long_term_memories(batch_memories)

# Batch search with different queries
search_queries = [
    "user food preferences",
    "user work schedule",
    "user communication style"
]

search_tasks = [
    client.search_long_term_memory(
        text=query,
        filters={"user_id": {"eq": "alice"}},
        limit=3
    )
    for query in search_queries
]

results = await asyncio.gather(*search_tasks)
```

### Advantages
- **Predictable behavior**: You control exactly when memory operations happen
- **Efficient**: No token overhead for tools, fewer API calls
- **Reliable**: No dependency on LLM decision-making
- **Optimizable**: You can optimize memory storage and retrieval patterns

### Disadvantages
- **More coding required**: You need to implement memory logic
- **Less natural**: Memory operations don't happen organically in conversation
- **Maintenance overhead**: Need to maintain memory extraction/retrieval logic

### Best Practices

```python
# 1. Use memory_prompt for enriched context
async def get_enriched_context(user_query: str, user_id: str, session_id: str):
    """Get context that includes both working memory and relevant long-term memories"""
    # First, get the working memory session (creates if doesn't exist)
    result = await client.get_or_create_working_memory(session_id)
    working_memory = result.memory

    # Then use memory_prompt with session ID
    return await client.memory_prompt(
        query=user_query,
        session_id=session_id,
        long_term_search={
            "text": user_query,
            "filters": {
                "user_id": {"eq": user_id},
                "namespace": {"eq": "personal"}  # Filter by domain
            },
            "limit": 5,
            "recency_boost": True  # Prefer recent relevant memories
        }
    )

# 2. Structure memories for searchability
good_memory = MemoryRecord(
    text="User Alice prefers Italian restaurants, especially ones with outdoor seating and vegetarian options",
    memory_type="semantic",
    topics=["food", "restaurants", "preferences", "dietary"],
    entities=["Italian", "outdoor seating", "vegetarian"],
    user_id="alice",
    namespace="dining"
)

# 3. Handle memory errors gracefully
async def safe_memory_search(query: str, **kwargs):
    try:
        return await client.search_long_term_memory(text=query, **kwargs)
    except Exception as e:
        logger.warning(f"Memory search failed: {e}")
        return MemoryRecordResults(memories=[], total=0)  # Empty results
```

## Pattern 3: Background Extraction (Automatic)

**When to use**: When you want the system to automatically learn from conversations without manual intervention.

**How it works**: Store conversations in working memory, and the system automatically extracts important information to long-term memory in the background.

### Basic Automatic Extraction

```python
from agent_memory_client import MemoryAPIClient
from agent_memory_client.models import WorkingMemory, MemoryMessage

client = MemoryAPIClient(base_url="http://localhost:8000")

async def store_conversation_with_auto_extraction(
    session_id: str,
    user_message: str,
    assistant_message: str,
    user_id: str
):
    """Store conversation - system will automatically extract memories"""

    # Create working memory with the conversation
    working_memory = WorkingMemory(
        session_id=session_id,
        messages=[
            MemoryMessage(role="user", content=user_message),
            MemoryMessage(role="assistant", content=assistant_message)
        ],
        user_id=user_id
    )

    # Store in working memory - background extraction will happen automatically
    await client.set_working_memory(session_id, working_memory)

    # The system will:
    # 1. Analyze the conversation for important information
    # 2. Extract structured memories (preferences, facts, events)
    # 3. Apply contextual grounding (resolve pronouns, references)
    # 4. Store extracted memories in long-term storage
    # 5. Deduplicate similar memories

# Example conversation that triggers extraction
await store_conversation_with_auto_extraction(
    session_id="alice_onboarding",
    user_message="I'm Alice, I work as a Product Manager at StartupCorp. I prefer morning meetings and I'm vegetarian.",
    assistant_message="Nice to meet you Alice! I'll remember your role at StartupCorp and your preferences for meetings and dietary needs.",
    user_id="alice"
)

# System automatically extracts:
# - "User Alice works as Product Manager at StartupCorp" (semantic)
# - "User prefers morning meetings" (semantic)
# - "User is vegetarian" (semantic)
```

### Structured Memory Addition

```python
async def add_structured_memories_for_extraction(
    session_id: str,
    structured_memories: list[dict],
    user_id: str
):
    """Add structured memories that will be promoted to long-term storage"""

    # Convert to MemoryRecord objects
    memory_records = [
        MemoryRecord(**memory_data, user_id=user_id)
        for memory_data in structured_memories
    ]

    # Add to working memory for automatic promotion
    working_memory = WorkingMemory(
        session_id=session_id,
        memories=memory_records,
        user_id=user_id
    )

    await client.set_working_memory(session_id, working_memory)

# Usage
await add_structured_memories_for_extraction(
    session_id="alice_profile_setup",
    structured_memories=[
        {
            "text": "User has 5 years experience in product management",
            "memory_type": "semantic",
            "topics": ["experience", "career", "product_management"],
            "entities": ["5 years", "product management"]
        },
        {
            "text": "User completed MBA at Stanford in 2019",
            "memory_type": "episodic",
            "event_date": "2019-06-15T00:00:00Z",
            "topics": ["education", "mba", "stanford"],
            "entities": ["MBA", "Stanford", "2019"]
        }
    ],
    user_id="alice"
)
```

### Long-Running Learning System

```python
class AutoLearningAgent:
    def __init__(self, memory_url: str):
        self.memory_client = MemoryAPIClient(base_url=memory_url)
        self.openai_client = openai.AsyncOpenAI()

    async def process_conversation(
        self,
        user_message: str,
        session_id: str,
        user_id: str
    ) -> str:
        """Process conversation with automatic learning"""

        # 1. Get working memory session (creates if doesn't exist)
        result = await self.memory_client.get_or_create_working_memory(session_id)
        working_memory = result.memory

        # 2. Get existing context for better responses
        context = await self.memory_client.memory_prompt(
            query=user_message,
            session_id=session_id,
            long_term_search={
                "text": user_message,
                "filters": {"user_id": {"eq": user_id}},
                "limit": 3
            }
        )

        # 3. Generate response with context
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=context.messages + [
                {"role": "user", "content": user_message}
            ]
        )

        assistant_message = response.choices[0].message.content

        # 4. Store conversation for automatic extraction
        await self.memory_client.set_working_memory(
            session_id,
            WorkingMemory(
                session_id=session_id,
                messages=[
                    MemoryMessage(role="user", content=user_message),
                    MemoryMessage(role="assistant", content=assistant_message)
                ],
                user_id=user_id
            )
        )

        return assistant_message

    async def get_learned_information(self, user_id: str, topic: str = None):
        """See what the system has learned about a user"""
        search_query = f"user {topic}" if topic else "user information preferences"

        results = await self.memory_client.search_long_term_memory(
            text=search_query,
            filters={"user_id": {"eq": user_id}},
            limit=10
        )

        return [memory.text for memory in results.memories]

# Usage - system learns over multiple conversations
agent = AutoLearningAgent(memory_url="http://localhost:8000")

# Conversation 1
await agent.process_conversation(
    user_message="I'm working on a React project with TypeScript",
    session_id="coding_help_1",
    user_id="dev_alice"
)

# Conversation 2
await agent.process_conversation(
    user_message="I prefer using functional components over class components",
    session_id="coding_help_2",
    user_id="dev_alice"
)

# Check what system learned
learned_info = await agent.get_learned_information(
    user_id="dev_alice",
    topic="coding preferences"
)
print("System learned:", learned_info)
# Might include: "User prefers functional components over class components"
```

### Advantages
- **Zero overhead**: No manual memory management required
- **Learns continuously**: System improves understanding over time
- **Contextual grounding**: Automatically resolves references and pronouns
- **Deduplication**: Prevents duplicate memories
- **Scales naturally**: Works with any conversation volume

### Disadvantages
- **Less control**: Can't control exactly what gets remembered
- **Delayed availability**: Extraction happens in background, not immediately
- **Potential noise**: Might extract irrelevant information
- **Requires conversation**: Needs conversational context to work well

### Best Practices

```python
# 1. Provide rich conversation context
working_memory = WorkingMemory(
    session_id=session_id,
    messages=[
        MemoryMessage(role="system", content="User is setting up their profile"),
        MemoryMessage(role="user", content="I'm a senior developer at Google"),
        MemoryMessage(role="assistant", content="I'll note your role as senior developer at Google")
    ],
    context="User onboarding conversation",
    user_id=user_id,
    namespace="profile_setup"  # Organize by domain
)

# 2. Monitor extraction quality
async def check_extracted_memories(user_id: str, session_id: str):
    """Review what was extracted from a session"""
    memories = await client.search_long_term_memory(
        text="",  # Get all memories
        filters={
            "user_id": {"eq": user_id},
            "session_id": {"eq": session_id}
        },
        limit=20
    )

    for memory in memories.memories:
        print(f"Extracted: {memory.text}")
        print(f"Topics: {memory.topics}")
        print(f"Created: {memory.created_at}")

# 3. Combine with manual memory editing when needed
if extracted_memory_needs_correction:
    await client.edit_long_term_memory(
        memory_id=memory.id,
        updates={
            "text": "Corrected version of the memory",
            "topics": ["updated", "topics"]
        }
    )
```

## Hybrid Patterns

Most production systems benefit from combining multiple patterns:

### Pattern Combination: Code + Background

```python
class HybridMemoryAgent:
    """Combines code-driven retrieval with background extraction"""

    def __init__(self, memory_url: str):
        self.memory_client = MemoryAPIClient(base_url=memory_url)
        self.openai_client = openai.AsyncOpenAI()

    async def chat(self, user_message: str, user_id: str, session_id: str) -> str:
        # 1. Get working memory session (creates if doesn't exist)
        result = await self.memory_client.get_or_create_working_memory(session_id)
        working_memory = result.memory

        # 2. Code-driven: Get relevant context
        context = await self.memory_client.memory_prompt(
            query=user_message,
            session_id=session_id,
            long_term_search={
                "text": user_message,
                "filters": {"user_id": {"eq": user_id}},
                "limit": 5
            }
        )

        # 3. Generate response
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=context.messages + [
                {"role": "user", "content": user_message}
            ]
        )

        assistant_message = response.choices[0].message.content

        # 4. Background: Store for automatic extraction
        await self.memory_client.set_working_memory(
            session_id,
            WorkingMemory(
                messages=[
                    MemoryMessage(role="user", content=user_message),
                    MemoryMessage(role="assistant", content=assistant_message)
                ],
                user_id=user_id
            )
        )

        return assistant_message
```

### Pattern Combination: LLM Tools + Background

```python
class SmartChatAgent:
    """LLM can use tools, plus automatic background learning"""

    async def chat(self, user_message: str, user_id: str, session_id: str) -> str:
        # Get memory tools
        tools = MemoryAPIClient.get_all_memory_tool_schemas()

        # LLM-driven: Let LLM use memory tools
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You have memory tools. Use them when relevant."},
                {"role": "user", "content": user_message}
            ],
            tools=tools
        )

        # Handle tool calls
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                await self.memory_client.resolve_function_call(
                    function_name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments),
                    session_id=session_id,
                    user_id=user_id
                )

        # Background: Also store conversation for automatic extraction
        # First ensure working memory session exists
        result = await self.memory_client.get_or_create_working_memory(session_id)
        working_memory = result.memory

        await self.memory_client.set_working_memory(
            session_id,
            WorkingMemory(
                messages=[
                    MemoryMessage(role="user", content=user_message),
                    MemoryMessage(role="assistant", content=response.choices[0].message.content)
                ],
                user_id=user_id
            )
        )

        return response.choices[0].message.content
```

## Decision Framework

Choose your integration pattern based on your requirements:

### 🤖 Use LLM-Driven When:
- Building conversational agents or chatbots
- Users should control what gets remembered
- Natural conversation flow is important
- You can handle token overhead and variable costs

### 📝 Use Code-Driven When:
- Building applications with specific workflows
- You need predictable memory behavior
- Memory operations should be optimized for performance
- You want full control over what gets stored and retrieved

### 🔄 Use Background Extraction When:
- Building learning systems that improve over time
- You want zero-overhead memory management
- Conversations provide rich context for extraction
- Long-term learning is more important than immediate control

### 🔗 Use Hybrid Patterns When:
- You want benefits of multiple approaches
- Different parts of your system have different needs
- You're building sophisticated AI applications
- You can handle the additional complexity

## Getting Started

1. **Start Simple**: Begin with Code-Driven pattern for predictable results
2. **Add Background**: Enable automatic extraction for continuous learning
3. **Consider LLM Tools**: Add when conversational control becomes important
4. **Optimize**: Monitor performance and adjust patterns based on usage

Each pattern can be implemented incrementally, allowing you to start simple and add complexity as your application grows.
