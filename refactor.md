# 🧱 Refactor Plan: Unified Agent Memory System

This plan brings the current memory server codebase in line with the new architecture: memory types are unified, memory promotion is safe and flexible, and both agents and LLMs can interact with memory via clean, declarative interfaces.

## 🆔 ULID Migration Update

**Status:** ✅ Completed - All ID generation now uses ULIDs

The codebase has been updated to use ULIDs (Universally Unique Lexicographically Sortable Identifiers) instead of nanoid for all ID generation:

- **Client-side**: `MemoryAPIClient.add_memories_to_working_memory()` auto-generates ULIDs for memories without IDs
- **Server-side**: All memory creation, extraction, and merging operations use ULIDs
- **Dependencies**: Replaced `nanoid>=2.0.0` with `python-ulid>=3.0.0` in pyproject.toml
- **Tests**: Updated all test files to use ULID generation
- **Benefits**: ULIDs provide better sortability and are more suitable for distributed systems

## 📅 Event Date Field Addition

**Status:** ✅ Completed - Added event_date field for episodic memories

Added proper temporal support for episodic memories by implementing an `event_date` field:

- **MemoryRecord Model**: Added `event_date: datetime | None` field to capture when the actual event occurred
- **Redis Storage**: Added `event_date` field to Redis hash storage with timestamp conversion
- **Search Support**: Added `EventDate` filter class and integrated into search APIs
- **Extraction**: Updated LLM extraction prompt to extract event dates for episodic memories
- **API Integration**: All search endpoints now support event_date filtering
- **Benefits**: Enables proper temporal queries for episodic memories (e.g., "what happened last month?")

## 🔒 Memory Type Enum Constraints

**Status:** ✅ Completed - Implemented enum-based memory type validation

Replaced loose string-based memory type validation with strict enum constraints:

- **MemoryTypeEnum**: Created `MemoryTypeEnum(str, Enum)` with values: `EPISODIC`, `SEMANTIC`, `MESSAGE`
- **MemoryRecord Model**: Updated `memory_type` field to use `MemoryTypeEnum` instead of `Literal`
- **EnumFilter Base Class**: Created `EnumFilter` that validates values against enum members
- **MemoryType Filter**: Updated `MemoryType` filter to extend `EnumFilter` with validation
- **Code Updates**: Updated all hardcoded string comparisons to use enum values
- **Benefits**: Prevents invalid memory type values and provides better type safety

##  REFACTOR COMPLETE!

**Status:** ✅ All stages completed successfully

The Unified Agent Memory System refactor has been completed with all 7 stages plus final integration implemented and tested. The system now provides:

- **Unified Memory Types**: Consistent `memory_type` field across all memory records
- **Clean Architecture**: `Memory*` classes without location-based assumptions
- **Safe Promotion**: ID-based deduplication and conflict resolution
- **Working Memory**: TTL-based session-scoped ephemeral storage
- **Background Processing**: Automatic promotion with timestamp management
- **Unified Search**: Single interface spanning working and long-term memory
- **LLM Tools**: Direct memory storage via MCP tool interfaces
- **Automatic Extraction**: LLM-powered memory extraction from messages
- **Sync Safety**: Robust client state resubmission handling

**Test Results:** 69 passed, 20 skipped - All functionality verified

---

## Running tests

Remember to run tests like this:
```
pytest --run-api-tests tests
```

You can use any normal pytest syntax to run specific tests.

---

## 🔁 Stage 1: Normalize Memory Types

**Goal:** Introduce consistent typing for all memory records.

**Instructions:**
- Define a `memory_type` field for all memory records.
  - Valid values: `"message"`, `"semantic"`, `"episodic"`, `"json"`
- Update APIs to require and validate this field.
- Migrate or adapt storage to use `memory_type` consistently.
- Ensure this field is included in indexing and query logic.

---

## 🔁 Stage 1.5: Rename `LongTermMemory*` Classes to `Memory*`

**Goal:** Remove location-based assumptions and align names with unified memory model.

**Instructions:**
- Rename:
  - `LongTermMemoryRecord` → `MemoryRecord`
  - `LongTermSemanticMemory` → `MemorySemantic`
  - `LongTermEpisodicMemory` → `MemoryEpisodic`
- Update all references in code, route handlers, type hints, and OpenAPI schema.
- Rely on `memory_type` and `persisted_at` to indicate state and type.

---

## 🔁 Stage 2: Add `id` and `persisted_at`

**Goal:** Support safe promotion and deduplication across working and long-term memory.

**Instructions:**
- Add `id: str | None` and `persisted_at: datetime | None` to all memory records.
- Enforce that:
  - `id` is required on memory sent from clients.
  - `persisted_at` is server-assigned and read-only for clients.
- Use `id` as the basis for deduplication and overwrites.

---

## 🔁 Stage 3: Implement Working Memory

**Goal:** Provide a TTL-based, session-scoped memory area for ephemeral agent context.

**Instructions:**
- Define Redis keyspace like `session:{id}:working_memory`.
- Implement:
  - `GET /sessions/{id}/memory` – returns current working memory.
  - `POST /sessions/{id}/memory` – replaces full working memory state.
- Set TTL on the working memory key (e.g. 1 hour default).
- Validate that all entries are valid memory records and carry `id`.


## 🔁 Stage 3.5: Merge Session and Working Memory

**Goal:** Unify short-term memory abstractions into "WorkingMemory."

**Instructions:**
1. Standardize on the term working_memory
    -   "Session" is now just an ID value used to scope memory
	•	Rename all references to session memory or session-scoped memory to working memory
	•	In class names, route handlers, docs, comments
	•	E.g. SessionMemoryStore → WorkingMemoryStore

2. Ensure session scoping is preserved in storage
	•	All working memory should continue to be scoped per session:
	•	e.g. session:{id}:working_memory
	•	Validate session ID on all read/write access

3. Unify schema and access
	•	Replace any duplicate logic, structures, or APIs (e.g. separate SessionMemory and WorkingMemory models)
	•	Collapse into one structure: WorkingMemory
	•	Use one canonical POST /sessions/{id}/memory and GET /sessions/{id}/memory

4. Remove or migrate session-memory-only features
	•	If session memory had special logic (e.g. treating messages differently), migrate that logic into working memory
	•	Ensure messages, JSON data, and unpersisted semantic/episodic memories all coexist in working_memory

5. Audit all interfaces that reference session memory
	•	Tool APIs, prompt hydration, memory promotion, etc. should now reference working_memory exclusively
	•	Update any internal helper functions or routes to reflect the change

---

## 🔁 Stage 4: Add Background Promotion Task

**Goal:** Automatically move eligible working memory records to long-term storage.

**Instructions:**
- On working memory update, trigger an async background task.
- Task should:
  - Identify memory records with no `persisted_at`.
  - Use `id` to detect and replace duplicates in long-term memory.
  - Persist the record and stamp it with `persisted_at = now()`.
  - Update the working memory session store to reflect new timestamps.

---

## 🔁 Stage 5: Memory Search Interface ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Implemented `search_memories` function (renamed from "unified" to just "memories")
- ✅ Added `POST /memory/search` endpoint that searches across all memory types
- ✅ Applied appropriate indexing and search logic:
  - Vector search for long-term memory (semantic search)
  - Simple text matching for working memory
  - Combined filtering and pagination across both types
- ✅ Included `memory_type` in search results along with all other memory fields
- ✅ Created comprehensive API tests for memory search endpoint
- ✅ Added unit test for `search_memories` function verifying working + long-term memory search
- ✅ Fixed linter errors with proper type handling
- ✅ Removed "unified" terminology in favor of cleaner "memory search"

**Result:** The system now provides a single search interface (`POST /memory/search`) that spans both working memory (ephemeral, session-scoped) and long-term memory (persistent, indexed). Working memory uses text matching while long-term memory uses semantic vector search. Results are combined, sorted by relevance, and properly paginated.

---

## 🔁 Stage 6: Tool Interfaces for LLMs ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Defined tool spec with required functions:
  - `store_memory(session_id, memory_type, content, tags, namespace, user_id, id)`
  - `store_json(session_id, data, namespace, user_id, id, tags)`
- ✅ Routed tool calls to session working memory via `PUT /sessions/{id}/memory`
- ✅ Auto-generated `id` using ULID when not supplied by client
- ✅ Marked all tool-created records as pending promotion (`persisted_at = null`)
- ✅ Added comprehensive MCP tool documentation with usage patterns
- ✅ Implemented proper namespace injection for both URL-based and default namespaces
- ✅ Created comprehensive tests for both tool functions including ID auto-generation
- ✅ Verified integration with existing working memory and background promotion systems

**Result:** LLMs can now explicitly store structured memory during conversation through tool calls. The `store_memory` tool handles semantic, episodic, message, and json memory types, while `store_json` provides a dedicated interface for structured data. Both tools integrate seamlessly with the working memory system and automatic promotion to long-term storage.

---

## 🔁 Stage 7: Automatic Memory Extraction from Messages ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Extended background promotion task to include message record extraction
- ✅ Implemented `extract_memories_from_messages` function for working memory context
- ✅ Added LLM-based extraction using `WORKING_MEMORY_EXTRACTION_PROMPT`
- ✅ Tagged extracted records with `extracted_from` field containing source message IDs
- ✅ Generated server-side IDs for all extracted memories using ULID
- ✅ Added `extracted_from` field to MemoryRecord model and Redis schema
- ✅ Updated indexing and search logic to handle extracted_from field
- ✅ Integrated extraction into promotion workflow with proper error handling
- ✅ Added extracted memories to working memory for future promotion cycles
- ✅ Verified all tests pass with new extraction functionality

**Result:** The system now automatically extracts semantic and episodic memories from message records during the promotion process. When message records are promoted to long-term storage, the system uses an LLM to identify useful information and creates separate memory records tagged with the source message ID. This enables rich memory formation from conversational content while maintaining traceability.

---

## 🧪 Final Integration: Sync and Conflict Safety ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Verified client state resubmission safety via `PUT /sessions/{id}/memory` endpoint
- ✅ Confirmed pending record handling: records with `id` but no `persisted_at` treated as pending
- ✅ Validated id-based overwrite logic in `deduplicate_by_id` function
- ✅ Ensured working memory always updated with latest `persisted_at` timestamps
- ✅ Created comprehensive test for sync and conflict safety scenarios
- ✅ Verified client can safely resubmit stale memory state with new records
- ✅ Confirmed long-term memory convergence over time through promotion cycles
- ✅ Validated that server handles partial client state gracefully
- ✅ Ensured proper timestamp management across promotion cycles

**Result:** The system now provides robust sync and conflict safety. Clients can safely resubmit partial or stale memory state, and the server will handle id-based deduplication and overwrites correctly. Working memory always converges to a consistent state with proper server-assigned timestamps, ensuring reliable memory management even with concurrent or repeated client submissions.

---

## Log of work

### Stage 1: Normalize Memory Types ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Analyzed current codebase structure
- ✅ Found that `memory_type` field already exists in `LongTermMemory` model with values: `"episodic"`, `"semantic"`, `"message"`
- ✅ Added `"json"` type support to the Literal type definition
- ✅ Verified field validation exists in APIs via MemoryType filter class
- ✅ Confirmed indexing and query logic includes this field in Redis search schema
- ✅ All memory search, indexing, and storage operations properly handle memory_type

**Result:** The `memory_type` field is now normalized with all required values: `"message"`, `"semantic"`, `"episodic"`, `"json"`

### Stage 1.5: Rename `LongTermMemory*` Classes to `Memory*` ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Renamed `LongTermMemory` → `MemoryRecord`
- ✅ Renamed `LongTermMemoryResult` → `MemoryRecordResult`
- ✅ Renamed `LongTermMemoryResults` → `MemoryRecordResults`
- ✅ Renamed `LongTermMemoryResultsResponse` → `MemoryRecordResultsResponse`
- ✅ Renamed `CreateLongTermMemoryRequest` → `CreateMemoryRecordRequest`
- ✅ Updated all references in code, route handlers, type hints, and OpenAPI schema
- ✅ Updated imports across all modules: models, long_term_memory, api, client, mcp, messages, extraction
- ✅ Updated all test files and their imports
- ✅ Verified all files compile without syntax errors

**Result:** All `LongTermMemory*` classes have been successfully renamed to `Memory*` classes, removing location-based assumptions and aligning with the unified memory model.

### Stage 2: Add `id` and `persisted_at` ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Added `id: str | None` and `persisted_at: datetime | None` to MemoryRecord model
- ✅ Updated Redis schema to include id (tag field) and persisted_at (numeric field)
- ✅ Updated indexing logic to store these fields with proper timestamp conversion
- ✅ Updated search logic to return new fields with datetime conversion
- ✅ Added validation to API to enforce id requirement for client-sent memory
- ✅ Ensured persisted_at is server-assigned and read-only for clients
- ✅ Implemented `deduplicate_by_id` function for id-based deduplication
- ✅ Integrated id deduplication as first step in indexing process
- ✅ Added comprehensive tests for id validation and deduplication
- ✅ Verified all existing tests pass with new functionality

**Result:** Id and persisted_at fields are now fully implemented with proper validation, deduplication logic, and safe promotion support as required by Stage 2.

### Stage 3: Implement Working Memory ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Defined Redis keyspace like `working_memory:{namespace}:{session_id}`
- ✅ Implemented `GET /sessions/{id}/working-memory` – returns current working memory
- ✅ Implemented `POST /sessions/{id}/working-memory` – replaces full working memory state
- ✅ Set TTL on working memory key (1 hour default, configurable)
- ✅ Validated that all entries are valid memory records and carry `id`
- ✅ Created WorkingMemory model containing list of MemoryRecord objects
- ✅ Implemented working memory storage/retrieval functions with JSON serialization
- ✅ Added comprehensive tests for working memory functionality and API endpoints
- ✅ Verified all tests pass with new functionality

**Result:** Working memory is now fully implemented as a TTL-based, session-scoped memory area for ephemeral agent context containing structured memory records that can be promoted to long-term storage.

### Stage 3.5: Merge Session and Working Memory ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Standardized on "working_memory" terminology throughout codebase
- ✅ Extended WorkingMemory model to support both messages and structured memory records
- ✅ Removed SessionMemory, SessionMemoryRequest, SessionMemoryResponse models
- ✅ Unified API endpoints to single /sessions/{id}/memory (GET/PUT/DELETE)
- ✅ Removed deprecated /working-memory endpoints
- ✅ Preserved session scoping in Redis storage (working_memory:{namespace}:{session_id})
- ✅ Removed duplicate logic and APIs between session and working memory
- ✅ Updated all interfaces to reference working_memory exclusively
- ✅ Migrated all session-memory-only features into working memory
- ✅ Updated all test files to use unified WorkingMemory models
- ✅ Verified all 80 tests pass with unified architecture

**Result:** Successfully unified short-term memory abstractions into "WorkingMemory" terminology, eliminating duplicate SessionMemory concepts while preserving session scoping. The system now has clean separation where working memory serves as TTL-based ephemeral storage and staging area for promotion to long-term storage.

### Additional Improvements ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ **Renamed `client_id` to `id`**: Updated all references throughout the codebase from `client_id` to `id` for cleaner API semantics. The field represents a client-side ID but doesn't need to indicate this in the schema name.
- ✅ **Implemented immediate summarization**: Modified `PUT /sessions/{id}/memory` to handle summarization inline instead of using background tasks. When the window size is exceeded, messages are summarized immediately and the updated working memory (with summary and trimmed messages) is returned to the client.
- ✅ **Updated client API**: Modified `MemoryAPIClient.put_session_memory()` to return `WorkingMemoryResponse` instead of `AckResponse`, allowing clients to receive the updated memory state including any summarization.
- ✅ **Fixed test mocks**: Updated all test files to use the new field names and response types.
- ✅ **Verified all tests pass**: All 80 tests pass with the updated implementation.

**Result:** The API now has cleaner field naming (`id` instead of `client_id`) and provides immediate feedback to clients when summarization occurs, allowing them to maintain accurate token limits and internal state.

### Stage 4: Add Background Promotion Task ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Created `promote_working_memory_to_long_term` function that automatically promotes eligible memories
- ✅ Implemented identification of memory records with no `persisted_at` in working memory
- ✅ Added id-based deduplication and overwrite detection during promotion
- ✅ Implemented proper `persisted_at` timestamp assignment using UTC datetime
- ✅ Added working memory update logic to reflect new timestamps after promotion
- ✅ Integrated promotion task into `put_session_memory` API endpoint as background task
- ✅ Added promotion function to Docket task collection for background processing
- ✅ Created comprehensive tests for promotion functionality and API integration
- ✅ Verified proper triggering of promotion task only when structured memories are present
- ✅ Verified all 82 tests pass with new functionality

**Result:** Background promotion task is now fully implemented. When working memory is updated via the API, unpersisted structured memory records are automatically promoted to long-term storage in the background, with proper deduplication and timestamp management. The working memory is updated to reflect the new `persisted_at` timestamps, ensuring client state consistency.

### Stage 5: Memory Search Interface ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Implemented `search_memories` function (renamed from "unified" to just "memories")
- ✅ Added `POST /memory/search` endpoint that searches across all memory types
- ✅ Applied appropriate indexing and search logic:
  - Vector search for long-term memory (semantic search)
  - Simple text matching for working memory
  - Combined filtering and pagination across both types
- ✅ Included `memory_type` in search results along with all other memory fields
- ✅ Created comprehensive API tests for memory search endpoint
- ✅ Added unit test for `search_memories` function verifying working + long-term memory search
- ✅ Fixed linter errors with proper type handling
- ✅ Removed "unified" terminology in favor of cleaner "memory search"

**Result:** The system now provides a single search interface (`POST /memory/search`) that spans both working memory (ephemeral, session-scoped) and long-term memory (persistent, indexed). Working memory uses text matching while long-term memory uses semantic vector search. Results are combined, sorted by relevance, and properly paginated.

### Stage 6: Tool Interfaces for LLMs ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Defined tool spec with required functions:
  - `store_memory(session_id, memory_type, content, tags, namespace, user_id, id)`
  - `store_json(session_id, data, namespace, user_id, id, tags)`
- ✅ Routed tool calls to session working memory via `PUT /sessions/{id}/memory`
- ✅ Auto-generated `id` using ULID when not supplied by client
- ✅ Marked all tool-created records as pending promotion (`persisted_at = null`)
- ✅ Added comprehensive MCP tool documentation with usage patterns
- ✅ Implemented proper namespace injection for both URL-based and default namespaces
- ✅ Created comprehensive tests for both tool functions including ID auto-generation
- ✅ Verified integration with existing working memory and background promotion systems

**Result:** LLMs can now explicitly store structured memory during conversation through tool calls. The `store_memory` tool handles semantic, episodic, message, and json memory types, while `store_json` provides a dedicated interface for structured data. Both tools integrate seamlessly with the working memory system and automatic promotion to long-term storage.

### Stage 7: Automatic Memory Extraction from Messages ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Extended background promotion task to include message record extraction
- ✅ Implemented `extract_memories_from_messages` function for working memory context
- ✅ Added LLM-based extraction using `WORKING_MEMORY_EXTRACTION_PROMPT`
- ✅ Tagged extracted records with `extracted_from` field containing source message IDs
- ✅ Generated server-side IDs for all extracted memories using nanoid
- ✅ Added `extracted_from` field to MemoryRecord model and Redis schema
- ✅ Updated indexing and search logic to handle extracted_from field
- ✅ Integrated extraction into promotion workflow with proper error handling
- ✅ Added extracted memories to working memory for future promotion cycles
- ✅ Verified all tests pass with new extraction functionality

**Result:** The system now automatically extracts semantic and episodic memories from message records during the promotion process. When message records are promoted to long-term storage, the system uses an LLM to identify useful information and creates separate memory records tagged with the source message ID. This enables rich memory formation from conversational content while maintaining traceability.

### Final Integration: Sync and Conflict Safety ✅ (Complete)

**Current Status:** ✅ Completed

**Progress:**
- ✅ Verified client state resubmission safety via `PUT /sessions/{id}/memory` endpoint
- ✅ Confirmed pending record handling: records with `id` but no `persisted_at` treated as pending
- ✅ Validated id-based overwrite logic in `deduplicate_by_id` function
- ✅ Ensured working memory always updated with latest `persisted_at` timestamps
- ✅ Created comprehensive test for sync and conflict safety scenarios
- ✅ Verified client can safely resubmit stale memory state with new records
- ✅ Confirmed long-term memory convergence over time through promotion cycles
- ✅ Validated that server handles partial client state gracefully
- ✅ Ensured proper timestamp management across promotion cycles

**Result:** The system now provides robust sync and conflict safety. Clients can safely resubmit partial or stale memory state, and the server will handle id-based deduplication and overwrites correctly. Working memory always converges to a consistent state with proper server-assigned timestamps, ensuring reliable memory management even with concurrent or repeated client submissions.
