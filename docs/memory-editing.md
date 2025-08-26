# Memory Editing

The Redis Agent Memory Server provides comprehensive memory editing capabilities, allowing you to update, correct, and refine stored memories through both REST API endpoints and MCP tools. This feature enables AI agents and applications to maintain accurate, up-to-date memory records over time.

## Overview

Memory editing allows you to modify existing long-term memories without losing their search indexing or metadata. This is essential for:

- **Correcting mistakes**: Fix inaccurate information in stored memories
- **Updating information**: Reflect changes in user preferences or circumstances
- **Adding details**: Enrich memories with additional context or information
- **Maintaining accuracy**: Keep memory store current and reliable

**Key Features:**
- **Partial updates**: Modify only the fields you want to change
- **Automatic re-indexing**: Updated memories are re-indexed for search
- **Vector consistency**: Embeddings are regenerated when text changes
- **Metadata preservation**: IDs, timestamps, and other metadata remain stable
- **Atomic operations**: Updates succeed or fail completely

## Memory Editing Workflow

### 1. Find the Memory

First, locate the memory you want to edit using search:

```python
# Search for memories to edit
results = await client.search_long_term_memory(
    text="user food preferences",
    limit=5
)

# Find the specific memory
memory_to_edit = results.memories[0]
memory_id = memory_to_edit.id
```

### 2. Prepare Updates

Specify only the fields you want to change:

```python
# Update only the text content
updates = {
    "text": "User prefers Mediterranean cuisine and is vegetarian"
}

# Or update multiple fields
updates = {
    "text": "User was promoted to Senior Engineer on January 15, 2024",
    "memory_type": "episodic",
    "event_date": "2024-01-15T14:30:00Z",
    "topics": ["career", "promotion", "engineering"],
    "entities": ["Senior Engineer", "promotion"]
}
```

### 3. Apply the Update

Use the appropriate interface to apply your changes:

```python
# Update the memory
updated_memory = await client.edit_long_term_memory(
    memory_id=memory_id,
    updates=updates
)
```

## REST API Interface

### Endpoint

**PATCH /v1/long-term-memory/{memory_id}**

Updates specific fields of an existing memory record.

### Request Format

```http
PATCH /v1/long-term-memory/01HXE2B1234567890ABCDEF
Content-Type: application/json
Authorization: Bearer your_token_here

{
  "text": "Updated memory text",
  "topics": ["new", "topics"],
  "entities": ["updated", "entities"],
  "memory_type": "semantic",
  "event_date": "2024-01-15T14:30:00Z",
  "namespace": "updated_namespace",
  "user_id": "updated_user"
}
```

### Response Format

```json
{
  "id": "01HXE2B1234567890ABCDEF",
  "text": "Updated memory text",
  "memory_type": "semantic",
  "topics": ["new", "topics"],
  "entities": ["updated", "entities"],
  "created_at": "2024-01-10T12:00:00Z",
  "persisted_at": "2024-01-10T12:00:00Z",
  "updated_at": "2024-01-16T10:30:00Z",
  "last_accessed": "2024-01-16T10:30:00Z",
  "user_id": "user_123",
  "session_id": "session_456",
  "namespace": "updated_namespace",
  "memory_hash": "new_hash_after_update"
}
```

### cURL Examples

**Update memory text:**
```bash
curl -X PATCH "http://localhost:8000/v1/long-term-memory/01HXE2B1234567890ABCDEF" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "text": "User prefers dark mode interfaces and uses vim for coding"
  }'
```

**Update multiple fields:**
```bash
curl -X PATCH "http://localhost:8000/v1/long-term-memory/01HXE2B1234567890ABCDEF" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "text": "User completed Python certification on January 15, 2024",
    "memory_type": "episodic",
    "event_date": "2024-01-15T14:30:00Z",
    "topics": ["education", "certification", "python"],
    "entities": ["Python", "certification"]
  }'
```

## MCP Tool Interface

### Tool: edit_long_term_memory

The MCP server provides an `edit_long_term_memory` tool for AI agents to modify memories through natural conversation.

### Tool Schema

```python
{
    "name": "edit_long_term_memory",
    "description": "Update an existing long-term memory with new or corrected information",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The ID of the memory to edit (get this from search results)"
            },
            "text": {
                "type": "string",
                "description": "Updated memory text content"
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Updated list of topics"
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Updated list of entities"
            },
            "memory_type": {
                "type": "string",
                "enum": ["semantic", "episodic", "message"],
                "description": "Type of memory"
            },
            "event_date": {
                "type": "string",
                "description": "Event date for episodic memories (ISO 8601 format)"
            },
            "namespace": {
                "type": "string",
                "description": "Memory namespace"
            },
            "user_id": {
                "type": "string",
                "description": "User ID associated with the memory"
            }
        },
        "required": ["memory_id"]
    }
}
```

### MCP Usage Examples

**Simple text update:**
```python
await client.call_tool("edit_long_term_memory", {
    "memory_id": "01HXE2B1234567890ABCDEF",
    "text": "User prefers tea over coffee (updated preference)"
})
```

**Update memory type and event date:**
```python
await client.call_tool("edit_long_term_memory", {
    "memory_id": "01HXE2B1234567890ABCDEF",
    "memory_type": "episodic",
    "event_date": "2024-01-15T14:30:00Z"
})
```

**Comprehensive update:**
```python
await client.call_tool("edit_long_term_memory", {
    "memory_id": "01HXE2B1234567890ABCDEF",
    "text": "User was promoted to Principal Engineer on January 15, 2024",
    "memory_type": "episodic",
    "event_date": "2024-01-15T14:30:00Z",
    "topics": ["career", "promotion", "engineering", "principal"],
    "entities": ["Principal Engineer", "promotion", "January 15, 2024"]
})
```

## Python Client Interface

### Method: edit_long_term_memory

```python
async def edit_long_term_memory(
    self,
    memory_id: str,
    updates: dict[str, Any]
) -> MemoryRecord:
    """
    Edit an existing long-term memory record.

    Args:
        memory_id: The ID of the memory to edit
        updates: Dictionary of fields to update

    Returns:
        The updated memory record

    Raises:
        HTTPException: If memory not found or update fails
    """
```

### Client Usage Examples

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Simple text correction
updated_memory = await client.edit_long_term_memory(
    memory_id="01HXE2B1234567890ABCDEF",
    updates={"text": "User actually prefers coffee, not tea"}
)

# Add more context
updated_memory = await client.edit_long_term_memory(
    memory_id="01HXE2B1234567890ABCDEF",
    updates={
        "text": "User prefers Italian cuisine, especially pasta and pizza",
        "topics": ["food", "preferences", "italian", "cuisine"],
        "entities": ["Italian cuisine", "pasta", "pizza"]
    }
)

# Update namespace and user
updated_memory = await client.edit_long_term_memory(
    memory_id="01HXE2B1234567890ABCDEF",
    updates={
        "namespace": "work_preferences",
        "user_id": "user_456"
    }
)
```

## Editable Fields

### Core Content Fields

- **text**: The main memory content (triggers embedding regeneration)
- **topics**: List of topic tags for categorization
- **entities**: List of named entities mentioned in the memory
- **memory_type**: Type classification (semantic, episodic, message)

### Temporal Fields

- **event_date**: Specific date/time for episodic memories (ISO 8601 format)

### Organization Fields

- **namespace**: Memory namespace for organization
- **user_id**: User associated with the memory

### Read-Only Fields

These fields cannot be edited and are managed automatically:

- **id**: Unique memory identifier
- **created_at**: Original creation timestamp
- **persisted_at**: When memory was first saved to long-term storage
- **updated_at**: Last modification timestamp (updated automatically)
- **last_accessed**: Last time memory was retrieved (managed by recency system)
- **memory_hash**: Content hash (regenerated when text changes)

## Update Behavior

### Automatic Updates

When you edit a memory, the system automatically:

1. **Updates timestamps**: Sets `updated_at` to current time
2. **Regenerates embeddings**: If text content changes, new embeddings are created
3. **Recalculates hash**: Content hash is updated for deduplication
4. **Re-indexes memory**: Search index is updated with new content
5. **Updates access time**: Sets `last_accessed` to current time

### Partial Updates

Only specify fields you want to change - other fields remain unchanged:

```python
# Only update topics - text, entities, etc. stay the same
updates = {"topics": ["programming", "python", "web-development"]}

# Only update text - topics, entities, etc. stay the same
updates = {"text": "Updated description of the user's preferences"}
```

### Vector Re-indexing

When memory text changes, the system automatically:
- Generates new embeddings using the configured embedding model
- Updates the vector index for accurate semantic search
- Maintains search performance and accuracy

## Error Handling

### Common Errors

**Memory Not Found (404):**
```json
{
  "detail": "Memory not found: 01HXE2B1234567890ABCDEF",
  "status_code": 404
}
```

**Invalid Memory ID (400):**
```json
{
  "detail": "Invalid memory ID format",
  "status_code": 400
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "event_date"],
      "msg": "invalid datetime format",
      "type": "value_error"
    }
  ],
  "status_code": 422
}
```

### Error Handling in Code

```python
try:
    updated_memory = await client.edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        updates={"text": "Updated text"}
    )
except HTTPException as e:
    if e.status_code == 404:
        print("Memory not found")
    elif e.status_code == 422:
        print("Invalid update data")
    else:
        print(f"Update failed: {e.detail}")
```

## Use Cases and Examples

### Correcting User Information

**Scenario**: User corrects their job title

```python
# 1. Search for the memory
results = await client.search_long_term_memory(
    text="user job title engineer",
    limit=1
)

# 2. Update with correction
if results.memories:
    await client.edit_long_term_memory(
        memory_id=results.memories[0].id,
        updates={
            "text": "User works as a Senior Software Engineer at TechCorp",
            "entities": ["Senior Software Engineer", "TechCorp"]
        }
    )
```

### Adding Context to Sparse Memories

**Scenario**: Enrich a basic memory with additional details

```python
# Original: "User likes pizza"
# Enhanced with context:
await client.edit_long_term_memory(
    memory_id="01HXE2B1234567890ABCDEF",
    updates={
        "text": "User likes pizza, especially thin crust with pepperoni and mushrooms from Mario's Pizzeria",
        "topics": ["food", "preferences", "pizza", "italian"],
        "entities": ["pizza", "thin crust", "pepperoni", "mushrooms", "Mario's Pizzeria"]
    }
)
```

### Converting Memory Types

**Scenario**: Convert a general memory to an episodic memory with event date

```python
# Change from semantic to episodic with specific date
await client.edit_long_term_memory(
    memory_id="01HXE2B1234567890ABCDEF",
    updates={
        "text": "User got promoted to Team Lead on March 15, 2024",
        "memory_type": "episodic",
        "event_date": "2024-03-15T09:00:00Z",
        "topics": ["career", "promotion", "team-lead"],
        "entities": ["Team Lead", "promotion", "March 15, 2024"]
    }
)
```

### Batch Memory Updates

**Scenario**: Update multiple related memories

```python
# Find all memories about a specific topic
results = await client.search_long_term_memory(
    text="old project name",
    limit=10
)

# Update each memory with the new project name
for memory in results.memories:
    updated_text = memory.text.replace("old project", "new project name")
    await client.edit_long_term_memory(
        memory_id=memory.id,
        updates={
            "text": updated_text,
            "entities": [entity.replace("old project", "new project name")
                        for entity in memory.entities or []]
        }
    )
```

## Best Practices

### Memory Identification

1. **Use search first**: Always search to find the correct memory ID
2. **Verify before editing**: Check memory content matches your expectations
3. **Handle duplicates**: Consider if multiple memories need the same update

### Update Strategy

1. **Minimal changes**: Only update fields that actually need to change
2. **Preserve context**: Don't remove important information when updating
3. **Consistent formatting**: Maintain consistent data formats across memories
4. **Validate inputs**: Check data formats before making updates

### Error Prevention

1. **Check memory exists**: Handle 404 errors gracefully
2. **Validate data**: Ensure update data matches expected formats
3. **Test updates**: Verify changes work as expected in development
4. **Monitor performance**: Watch for degradation with frequent updates

### Performance Considerations

1. **Batch operations**: Group related updates when possible
2. **Avoid unnecessary updates**: Don't update if content hasn't actually changed
3. **Monitor embedding costs**: Text updates trigger new embedding generation
4. **Consider timing**: Updates during low-traffic periods for better performance

## Integration with Other Features

### Memory Search

Updated memories are immediately searchable with their new content:

```python
# After updating memory with new content
await client.edit_long_term_memory(
    memory_id="01HXE2B1234567890ABCDEF",
    updates={"text": "User loves Mediterranean cuisine"}
)

# Can immediately search for the updated content
results = await client.search_long_term_memory(
    text="Mediterranean cuisine",
    limit=5
)
# Updated memory will appear in results
```

### Recency Boost

Memory editing updates the `last_accessed` timestamp, which affects recency scoring:

```python
# Editing a memory makes it "recently accessed"
# This can boost its ranking in recency-weighted searches
```

### Working Memory

Memories can be updated based on new information from working memory:

```python
# Extract new information from current conversation
# Update existing memories with corrections or additions
# Maintain consistency between working and long-term memory
```

This comprehensive memory editing system ensures that your AI agent's memory remains accurate, current, and useful over time, adapting to new information and corrections as they become available.
