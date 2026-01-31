# Agent Memory Client (JavaScript/TypeScript)

A TypeScript/JavaScript client for the [Agent Memory Server](https://redis.github.io/agent-memory-server/) REST API.

## Installation

```bash
npm install agent-memory-client
```

## Quick Start

```typescript
import { MemoryAPIClient } from "agent-memory-client";

const client = new MemoryAPIClient({
  baseUrl: "http://localhost:8000",
});

// Store a memory
await client.createLongTermMemory([
  {
    text: "User prefers dark mode and morning meetings",
    memory_type: "semantic",
    topics: ["preferences", "ui"],
    user_id: "alice",
  },
]);

// Search memories
const results = await client.searchLongTermMemory({
  text: "user interface preferences",
  limit: 10,
});

console.log(`Found ${results.memories?.length} relevant memories`);
```

## Configuration

```typescript
import { MemoryAPIClient } from "agent-memory-client";

const client = new MemoryAPIClient({
  baseUrl: "http://localhost:8000",
  timeout: 30000, // Request timeout in ms (default: 30000)
  apiKey: "your-api-key", // Optional API key
  bearerToken: "your-token", // Optional Bearer token
  defaultNamespace: "my-app", // Optional default namespace
});
```

## Working Memory

```typescript
// Create/update working memory
await client.putWorkingMemory("session-123", {
  messages: [
    { role: "user", content: "Hello!" },
    { role: "assistant", content: "Hi there!" },
  ],
  data: { preferences: { theme: "dark" } },
});

// Get working memory
const memory = await client.getWorkingMemory("session-123");

// Get or create (creates if not exists)
const result = await client.getOrCreateWorkingMemory("session-123");

// Delete working memory
await client.deleteWorkingMemory("session-123");

// List all sessions
const sessions = await client.listSessions({ limit: 100 });
```

## Long-Term Memory

```typescript
import { SessionId, Topics, UserId, CreatedAt } from "agent-memory-client";

// Create memories
await client.createLongTermMemory([
  {
    text: "User enjoys science fiction books",
    memory_type: "semantic",
    topics: ["books", "preferences"],
    user_id: "user-123",
  },
]);

// Search with filters
const results = await client.searchLongTermMemory({
  text: "science fiction",
  topics: new Topics({ any: ["books", "entertainment"] }),
  userId: new UserId({ eq: "user-123" }),
  limit: 20,
});

// Get by ID
const memory = await client.getLongTermMemory("memory-id");

// Edit memory
await client.editLongTermMemory("memory-id", { text: "Updated text" });

// Delete memories
await client.deleteLongTermMemories(["memory-1", "memory-2"]);
```

## Filters

All filter types support flexible matching:

```typescript
import {
  SessionId,
  Namespace,
  UserId,
  Topics,
  Entities,
  CreatedAt,
  MemoryType,
} from "agent-memory-client";

// Equality
new SessionId({ eq: "session-1" });

// Multiple values
new SessionId({ in_: ["session-1", "session-2"] });

// Negation
new SessionId({ not_eq: "session-1", not_in: ["session-2"] });

// Topics/Entities matching
new Topics({ any: ["topic1", "topic2"] }); // Match any
new Topics({ all: ["topic1", "topic2"] }); // Match all
new Topics({ none: ["topic3"] }); // Exclude

// Date ranges
new CreatedAt({ gte: new Date("2024-01-01"), lte: new Date("2024-12-31") });
```

## Batch Operations

```typescript
// Bulk create with rate limiting
const batches = [batch1, batch2, batch3];
await client.bulkCreateLongTermMemories(batches, {
  batchSize: 50,
  delayBetweenBatches: 100, // ms
});

// Auto-paginating search
for await (const memory of client.searchAllLongTermMemories({
  text: "user preferences",
  batchSize: 100,
})) {
  console.log(memory.text);
}
```

## Validation

```typescript
import { MemoryValidationError } from "agent-memory-client";

try {
  client.validateMemoryRecord({ text: "", memory_type: "semantic" });
} catch (error) {
  if (error instanceof MemoryValidationError) {
    console.error("Validation failed:", error.message);
  }
}

client.validateSearchFilters({ limit: 10, offset: 0 });
```

## Memory Prompt (Context Hydration)

Get memory-enhanced prompts for AI agents:

```typescript
const prompt = await client.memoryPrompt({
  query: "What does the user like?",
  session: { session_id: "session-123" },
  long_term_search: {
    text: "user preferences",
    limit: 5,
  },
});
```

## Summary Views

Create and manage dynamic summaries:

```typescript
// Create a summary view
const view = await client.createSummaryView({
  name: "User Summaries",
  source: "long_term",
  group_by: ["user_id"],
});

// Run a partition
const result = await client.runSummaryViewPartition("view-id", {
  user_id: "alice",
});

// List partitions
const partitions = await client.listSummaryViewPartitions("view-id");
```

## Forgetting Memories

Apply forgetting policies:

```typescript
const result = await client.forgetLongTermMemories({
  policy: { max_age_days: 30 },
  dryRun: true, // Preview what would be deleted
});
```

## Error Handling

```typescript
import {
  MemoryClientError,
  MemoryValidationError,
  MemoryNotFoundError,
  MemoryServerError,
} from "agent-memory-client";

try {
  await client.getWorkingMemory("nonexistent-session");
} catch (error) {
  if (error instanceof MemoryNotFoundError) {
    console.log("Session not found");
  } else if (error instanceof MemoryServerError) {
    console.log(`Server error ${error.statusCode}: ${error.message}`);
  } else if (error instanceof MemoryClientError) {
    console.log(`Client error: ${error.message}`);
  }
}
```

## TypeScript Support

Full TypeScript support with exported types:

```typescript
import type {
  MemoryRecord,
  WorkingMemory,
  SearchOptions,
  MemoryRecordResults,
} from "agent-memory-client";
```

## License

MIT
