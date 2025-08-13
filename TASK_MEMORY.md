# Task Memory

**Created:** 2025-08-12 17:40:24
**Branch:** feature/add-an-api

## Requirements

Add an API endpoint, MCP endpoint, and tool for editing existing memories. And whenever we add memories to a prompt, always include the memory ID, so the LLM can use the edit memory tool to edit that memory by ID.

## Development Notes

*Update this section as you work on the task. Include:*
- *Progress updates*
- *Key decisions made*
- *Challenges encountered*
- *Solutions implemented*
- *Files modified*
- *Testing notes*

### Work Log

- [2025-08-12 17:40:24] Task setup completed, TASK_MEMORY.md created
- [2025-08-13 11:47:00] Set up development environment with uv
- [2025-08-13 11:48:00] Analyzed existing API and MCP structures:
  * REST API: endpoints in api.py use HTTP methods with background tasks
  * MCP: tools in mcp.py that call core API functions
  * Models: MemoryRecord with id field already exists
  * Long-term memory: stored in RedisVL vectorstore with search capabilities
- [2025-08-13 11:49:00] Design decisions:
  * Add PATCH endpoint for editing: `/v1/long-term-memory/{memory_id}`
  * Add MCP tool: `edit_long_term_memory`
  * Update memory prompt functions to include memory IDs in responses
  * Support partial updates (text, topics, entities, memory_type, namespace)
  * Maintain audit trail with updated_at timestamp
- [2025-08-13 11:50:00] Implementation completed:
  * Added `get_long_term_memory_by_id` and `update_long_term_memory` functions in long_term_memory.py
  * Added `EditMemoryRecordRequest` model in models.py
  * Added REST API endpoints: GET and PATCH `/v1/long-term-memory/{memory_id}`
  * Added MCP tools: `get_long_term_memory` and `edit_long_term_memory`
  * Updated memory prompt to include memory IDs: `- {memory.text} (ID: {memory.id})`
  * Fixed linting issues and ran code formatting
  * Tested model creation and import functionality

### Summary of Changes

1. **New Functions in long_term_memory.py:**
   - `get_long_term_memory_by_id(memory_id)` - Retrieve memory by ID
   - `update_long_term_memory(memory_id, updates)` - Update memory with validation

2. **New Model in models.py:**
   - `EditMemoryRecordRequest` - Pydantic model for partial memory updates

3. **New REST API Endpoints in api.py:**
   - `GET /v1/long-term-memory/{memory_id}` - Get memory by ID
   - `PATCH /v1/long-term-memory/{memory_id}` - Update memory by ID

4. **New MCP Tools in mcp.py:**
   - `get_long_term_memory(memory_id)` - Retrieve memory by ID
   - `edit_long_term_memory(memory_id, **updates)` - Update memory fields

5. **Enhanced Memory Prompt:**
   - Memory prompt now includes IDs: `- {memory.text} (ID: {memory.id})`
   - LLMs can use the IDs to call edit_long_term_memory tool

### Testing Status
- ✅ Code imports successfully
- ✅ New model works correctly
- ✅ Linting and formatting passes
- ⚠️ Full integration tests require Redis and OpenAI API key (not run)

---

*This file serves as your working memory for this task. Keep it updated as you progress through the implementation.*
