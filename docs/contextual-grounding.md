# Contextual Grounding

Contextual grounding is an advanced feature that ensures extracted memories contain complete, unambiguous information by resolving pronouns, temporal references, and other contextual elements within the full conversation history. This eliminates confusion from vague references like "he," "yesterday," or "that place" in stored memories.

## Overview

When AI agents extract memories from conversations, they often contain ambiguous references that lose meaning when viewed outside the original context. Contextual grounding solves this by automatically resolving these references using the complete conversation history.

**Problem Example:**

```
Original conversation:
User: "I met John at the coffee shop yesterday"
Assistant: "That sounds nice! How did it go?"
User: "He was really helpful with the project"

Without grounding: "He was really helpful with the project"
With grounding: "John was really helpful with the project"
```

**Key Benefits:**

- **Clear memories**: No ambiguous pronouns or references
- **Standalone context**: Memories make sense without conversation history
- **Better search**: More precise matching with complete information
- **Reduced confusion**: Eliminates "who/what/when/where" ambiguity

## Types of Contextual Grounding

### 1. Pronoun Resolution

Replaces pronouns with their actual referents from conversation context.

**Examples:**

- "He likes coffee" → "John likes coffee"
- "She recommended the book" → "Sarah recommended the book"
- "They are meeting tomorrow" → "Alice and Bob are meeting tomorrow"
- "It was expensive" → "The restaurant was expensive"

### 2. Temporal Grounding

Converts relative time references to specific dates and times.

**Examples:**

- "Yesterday" → "January 15, 2024"
- "Last week" → "The week of January 8-14, 2024"
- "Tomorrow" → "January 17, 2024"
- "This morning" → "January 16, 2024 morning"

### 3. Spatial Grounding

Resolves location references to specific places mentioned in context.

**Examples:**

- "That place" → "Starbucks on Main Street"
- "There" → "The office conference room"
- "Here" → "The user's home office"

### 4. Entity Grounding

Links vague references to specific entities from the conversation.

**Examples:**

- "The project" → "The website redesign project"
- "The meeting" → "The quarterly review meeting"
- "The document" → "The project proposal document"

## How Contextual Grounding Works

### Memory Extraction Process

1. **Conversation Analysis**: System analyzes the full conversation thread
2. **Memory Identification**: Identifies important information to store
3. **Context Resolution**: Uses conversation history to resolve ambiguous references
4. **Memory Creation**: Stores resolved, context-complete memories

### Technical Implementation

Contextual grounding uses advanced language models to understand conversation context and resolve references:

```python
# Example of contextual grounding in action
conversation_messages = [
    "User: I had lunch with Sarah at the new Italian place downtown",
    "Assistant: How was the food?",
    "User: It was amazing! She loved the pasta too",
    "Assistant: That's great to hear!"
]

# Without grounding:
extracted_memory = "She loved the pasta too"

# With contextual grounding:
grounded_memory = "Sarah loved the pasta at the new Italian place downtown"
```

## Configuration

Contextual grounding is automatically enabled when memory extraction is active and works with the configured language model.

### Environment Variables

```bash
# Enable memory extraction (includes contextual grounding)
ENABLE_DISCRETE_MEMORY_EXTRACTION=true

# Model used for extraction and grounding
GENERATION_MODEL=gpt-4o-mini

# Enable long-term memory features
LONG_TERM_MEMORY=true
```

### Model Requirements

Contextual grounding works with any supported language model, but performance varies:

**Recommended Models:**
- **gpt-4o**: Best accuracy for complex grounding
- **gpt-4o-mini**: Good balance of speed and accuracy
- **claude-3-5-sonnet**: Excellent at contextual understanding
- **claude-3-haiku**: Fast, good for simple grounding

See [LLM Providers](llm-providers.md) for complete model configuration options.

## Usage Examples

### Automatic Memory Extraction

Contextual grounding works automatically when memories are extracted from conversations:

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Add conversation messages to working memory
working_memory = WorkingMemory(
    session_id="conversation_123",
    messages=[
        MemoryMessage(role="user", content="I met Dr. Smith yesterday"),
        MemoryMessage(role="assistant", content="How did the appointment go?"),
        MemoryMessage(role="user", content="He said I need to exercise more"),
    ]
)

# Save working memory - system automatically extracts and grounds memories
await client.set_working_memory("conversation_123", working_memory)

# Extracted memory will be: "Dr. Smith said the user needs to exercise more"
# Instead of: "He said I need to exercise more"
```

### Manual Memory Creation

Even manually created memories benefit from contextual grounding when context is available:

```python
# Create memory with context
memory_record = MemoryRecord(
    text="She really enjoyed the presentation",
    session_id="meeting_456",
    memory_type="episodic"
)

# If conversation context exists, grounding will resolve "She" to the specific person
await client.create_long_term_memories([memory_record])
```

## Real-World Examples

### Customer Support Context

**Conversation:**
```
Customer: "I ordered a laptop last week, order #12345"
Agent: "I can help with that. What's the issue?"
Customer: "It arrived damaged. The screen has cracks"
Agent: "I'm sorry to hear that. We'll replace it right away"
Customer: "Thank you! When will the replacement arrive?"
```

**Without Grounding:**
- "It arrived damaged"
- "The screen has cracks"
- "We'll replace it right away"

**With Contextual Grounding:**
- "The laptop from order #12345 arrived damaged"
- "The laptop screen from order #12345 has cracks"
- "The company will replace the damaged laptop from order #12345 right away"

### Personal Assistant Context

**Conversation:**
```
User: "I have a meeting with Jennifer at 2 PM about the marketing campaign"
Assistant: "I've noted that. Anything else to prepare?"
User: "Yes, she wants to see the budget numbers"
Assistant: "I'll remind you to bring those"
User: "Also, the meeting is in her office on the 5th floor"
```

**Without Grounding:**
- "She wants to see the budget numbers"
- "The meeting is in her office on the 5th floor"

**With Contextual Grounding:**
- "Jennifer wants to see the budget numbers for the marketing campaign"
- "The meeting with Jennifer about the marketing campaign is in her office on the 5th floor"

## Quality Evaluation

The system includes LLM-as-a-Judge evaluation to assess contextual grounding quality:

### Evaluation Categories

1. **Pronoun Grounding**: How well pronouns are resolved
2. **Temporal Grounding**: Accuracy of time reference resolution
3. **Spatial Grounding**: Precision of location reference resolution
4. **Entity Grounding**: Completeness of entity reference resolution

### Quality Metrics

```python
# Example evaluation results
grounding_quality = {
    "pronoun_accuracy": 0.85,      # 85% of pronouns correctly resolved
    "temporal_accuracy": 0.92,     # 92% of time references resolved
    "spatial_accuracy": 0.78,      # 78% of location references resolved
    "entity_accuracy": 0.89,       # 89% of entity references resolved
    "overall_score": 0.86          # Overall grounding quality
}
```

## Troubleshooting

### Common Issues

**Pronouns not resolved:**
- Verify the conversation includes clear entity introductions
- Check that the conversation history is available during extraction
- Ensure the language model has sufficient context window

**Time references incorrect:**
- Confirm conversation timestamps are accurate
- Check timezone settings in your application
- Verify temporal context is clear in the conversation

**Entity references ambiguous:**
- Use specific names and identifiers in conversations
- Avoid overloading conversations with too many similar entities
- Provide clear context when introducing new entities

### Debug Information

Enable detailed logging to troubleshoot grounding issues:

```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Review extraction and grounding logs
tail -f logs/agent_memory_server.log | grep "grounding"
```

## Advanced Features

### Multi-Turn Context

Contextual grounding works across multiple conversation turns:

```python
# Turn 1
"User mentioned the project deadline is next Friday"

# Turn 5
"He's concerned about finishing on time"
# Grounds to: "User is concerned about finishing the project by next Friday"

# Turn 10
"The team should prioritize it"
# Grounds to: "The team should prioritize the project with the Friday deadline"
```

### Cross-Session References

When memories span multiple sessions, grounding can reference previous context:

```python
# Session 1: Project discussion
# Session 2: "Update on that project we discussed"
# Grounds to: "Update on [specific project name] we discussed"
```

### Complex Entity Resolution

Handles complex entity relationships and hierarchies:

```python
# Original: "The CEO's assistant called about the board meeting"
# Context: CEO is "John Smith", assistant is "Mary Johnson"
# Grounded: "Mary Johnson (John Smith's assistant) called about the board meeting"
```

## Integration with Other Features

### Memory Search

Contextual grounding improves search quality by providing complete context:

```python
# Search for: "John project discussion"
# Finds grounded memory: "John was concerned about finishing the project by Friday"
# Instead of vague: "He was concerned about finishing on time"
```

### Recency Boost

Grounded memories work better with recency boost since they contain complete temporal information:

```python
# Grounded memory: "User met with Dr. Smith on January 15, 2024"
# Recency boost can accurately weight by specific date
# Instead of ambiguous: "User met with him yesterday"
```

This contextual grounding feature ensures that stored memories are clear, complete, and meaningful when retrieved, significantly improving the overall quality of AI agent memory systems.
