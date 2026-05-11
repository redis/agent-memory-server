# ✏️ Memory Editing Agent

**File**: [`examples/memory_editing_agent.py`](https://github.com/redis/agent-memory-server/blob/main/examples/memory_editing_agent.py)

Demonstrates comprehensive memory editing capabilities through natural conversation patterns.

## Core Features

- **Memory Editing Workflow**: Complete lifecycle of creating, searching, editing, and deleting memories
- **All Memory Tools**: Uses all available memory management tools including editing capabilities
- **Realistic Scenarios**: Common patterns like corrections, updates, and information cleanup
- **Interactive Demo**: Both automated demo and interactive modes

## Memory Operations Demonstrated

1. **search_memory** — Find existing memories using natural language (supports `semantic`, `keyword`, and `hybrid` search modes)
2. **get_long_term_memory** — Retrieve specific memories by ID
3. **lazily_create_long_term_memory** — Store new information (promoted to long-term storage later)
4. **eagerly_create_long_term_memory** — Create long-term memories directly for immediate storage
5. **edit_long_term_memory** — Update existing memories
6. **delete_long_term_memories** — Remove outdated information
7. **get_or_create_working_memory** — Check current working memory session
8. **update_working_memory_data** — Store/update session-specific data
9. **get_current_datetime** — Get current UTC datetime for grounding relative time expressions

## Common Editing Scenarios

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

## Usage Examples

```bash
cd examples

# Interactive mode (explore memory editing)
python memory_editing_agent.py

# Automated demo (see complete workflow)
python memory_editing_agent.py --demo

# Custom configuration
python memory_editing_agent.py --session-id alice_session --user-id alice
```

## Demo Conversation Flow

The automated demo shows a realistic conversation:

1. **Initial Information**: User shares profile (name, job, preferences)
2. **Corrections**: User corrects information (job company change)
3. **Updates**: User provides updates (promotion, new title)
4. **Multiple Changes**: User updates location and preferences
5. **Information Retrieval**: User asks what agent remembers
6. **Ongoing Updates**: Continued information updates
7. **Memory Management**: Specific memory operations (show/delete)

## See Also

- [Memory editing how-to guide](../user_guide/how_to_guides/memory_editing.md) — endpoint reference, MCP tool, and editable field details
