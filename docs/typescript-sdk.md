# TypeScript SDK

The TypeScript SDK (`agent-memory-client`) provides a type-safe client for integrating memory capabilities into Node.js and browser applications.

**Version**: 0.3.2+
**Requirements**: Node.js 20.0.0 or higher

## Installation

```bash
npm install agent-memory-client
# or
yarn add agent-memory-client
# or
pnpm add agent-memory-client
```

## Quick Start

```typescript
import { MemoryAPIClient, UserId, Topics } from "agent-memory-client";

// Create client
const client = new MemoryAPIClient({
  baseUrl: "http://localhost:8000",
  defaultNamespace: "my-app",
});

// Store a memory
await client.createLongTermMemory([
  {
    text: "User prefers morning meetings",
    memory_type: "semantic",
    topics: ["scheduling", "preferences"],
    user_id: "alice",
  },
]);

// Search memories with filters
const results = await client.searchLongTermMemory({
  text: "when does user prefer meetings",
  userId: new UserId({ eq: "alice" }),
  topics: new Topics({ any: ["scheduling"] }),
  limit: 5,
});

for (const memory of results.memories) {
  console.log(`${memory.text} (distance: ${memory.dist})`);
}

// Clean up
client.close();
```

## Client Configuration

```typescript
import { MemoryAPIClient, type MemoryClientConfig } from "agent-memory-client";

const config: MemoryClientConfig = {
  baseUrl: "http://localhost:8000",  // Required
  timeout: 30000,                     // Request timeout (ms)
  defaultNamespace: "production",     // Default namespace
  defaultModelName: "gpt-4o",        // For auto-summarization
  defaultContextWindowMax: 128000,    // Context window limit
  apiKey: "your-api-key",            // Optional API key auth
  bearerToken: "your-jwt",           // Optional JWT auth
};

const client = new MemoryAPIClient(config);
```

## Memory Operations

### Creating Memories

```typescript
import type { MemoryRecord } from "agent-memory-client";

const memories: MemoryRecord[] = [
  {
    text: "User works as a software engineer at TechCorp",
    memory_type: "semantic",
    topics: ["career", "work"],
    entities: ["TechCorp"],
    user_id: "alice",
  },
];

await client.createLongTermMemory(memories);
```

### Searching with Filters

The SDK provides type-safe filter classes:

```typescript
import {
  SessionId,
  Namespace,
  UserId,
  Topics,
  Entities,
  CreatedAt,
  LastAccessed,
  MemoryType,
} from "agent-memory-client";

// Basic semantic search (default)
const results = await client.searchLongTermMemory({
  text: "user preferences",
  limit: 10,
});

// Keyword search - full-text matching
const keywordResults = await client.searchLongTermMemory({
  text: "TechCorp engineer",
  searchMode: "keyword",
  limit: 10,
});

// Hybrid search - combines semantic and keyword matching
const hybridResults = await client.searchLongTermMemory({
  text: "user preferences",
  searchMode: "hybrid",
  hybridAlpha: 0.7, // 0.0=keyword, 1.0=semantic
  limit: 10,
});

// With filters
const filtered = await client.searchLongTermMemory({
  text: "programming languages",
  userId: new UserId({ eq: "alice" }),
  topics: new Topics({ any: ["programming", "languages"] }),
  memoryType: new MemoryType({ eq: "semantic" }),
  createdAt: new CreatedAt({ gte: new Date("2024-01-01") }),
  distanceThreshold: 0.3,
  limit: 5,
});

// Process results
for (const memory of filtered.memories) {
  const relevance = memory.dist ? 1 - memory.dist : null;
  console.log(`[${relevance?.toFixed(2)}] ${memory.text}`);
}
```

### Filter Reference

| Filter | Options | Description |
|--------|---------|-------------|
| `SessionId` | `eq`, `in_`, `not_eq`, `not_in` | Filter by session ID |
| `Namespace` | `eq`, `in_`, `not_eq`, `not_in` | Filter by namespace |
| `UserId` | `eq`, `in_`, `not_eq`, `not_in` | Filter by user ID |
| `Topics` | `any`, `all`, `none` | Filter by topics |
| `Entities` | `any`, `all`, `none` | Filter by entities |
| `CreatedAt` | `gte`, `lte`, `eq` | Filter by creation date |
| `LastAccessed` | `gte`, `lte`, `eq` | Filter by last access |
| `MemoryType` | `eq`, `in_`, `not_eq`, `not_in` | Filter by type |

### Editing and Deleting

```typescript
// Edit a memory
const updated = await client.editLongTermMemory("memory-id", {
  text: "Updated text content",
  topics: ["updated", "topics"],
});

// Get a specific memory
const memory = await client.getLongTermMemory("memory-id");

// Delete memories
await client.deleteLongTermMemories(["memory-id-1", "memory-id-2"]);
```

## Working Memory

```typescript
import type { WorkingMemory } from "agent-memory-client";

// Get or create working memory
const response = await client.getOrCreateWorkingMemory("session-123", {
  userId: "alice",
  namespace: "my-app",
});

// Update working memory
const workingMemory: Partial<WorkingMemory> = {
  messages: [
    { role: "user", content: "I'm planning a trip to Italy" },
    { role: "assistant", content: "That sounds exciting!" },
  ],
  memories: [
    {
      text: "User is planning a trip to Italy",
      memory_type: "semantic",
      topics: ["travel"],
    },
  ],
  data: { destination: "Italy" },
};

await client.putWorkingMemory("session-123", workingMemory);

// Delete working memory
await client.deleteWorkingMemory("session-123");
```

## Forgetting Memories

```typescript
import type { ForgetPolicy } from "agent-memory-client";

const policy: ForgetPolicy = {
  max_age_days: 90,
  max_inactive_days: 30,
  budget: 100,
  memory_type_allowlist: ["episodic"],
};

// Preview what would be deleted
const preview = await client.forgetLongTermMemories({
  policy,
  namespace: "my-app",
  dryRun: true,
});
console.log(`Would delete ${preview.deleted} of ${preview.scanned}`);

// Execute forget
const result = await client.forgetLongTermMemories({
  policy,
  namespace: "my-app",
  pinnedIds: ["keep-this-memory"],
});
```

## Summary Views

```typescript
import type { CreateSummaryViewRequest } from "agent-memory-client";

// Create a summary view
const request: CreateSummaryViewRequest = {
  name: "User Topic Summaries",
  source: "long_term",
  group_by: ["user_id", "topics"],
  time_window_days: 30,
  continuous: true,
};

const view = await client.createSummaryView(request);

// Run a partition
const partition = await client.runSummaryViewPartition(view.id, {
  user_id: "alice",
  topics: "travel",
});
console.log(`Summary: ${partition.summary}`);

// Run full view as background task
const task = await client.runSummaryView(view.id, { force: true });

// Poll for completion
let taskStatus = await client.getTask(task.id);
while (taskStatus && !["completed", "failed"].includes(taskStatus.status)) {
  await new Promise((r) => setTimeout(r, 1000));
  taskStatus = await client.getTask(task.id);
}

// List and delete views
const views = await client.listSummaryViews();
await client.deleteSummaryView(view.id);
```

## Bulk Operations

```typescript
// Bulk create with rate limiting
const batches = [memories1, memories2, memories3];
const results = await client.bulkCreateLongTermMemories(batches, {
  batchSize: 50,
  delayBetweenBatches: 100,
});

// Auto-paginating search
for await (const memory of client.searchAllLongTermMemories({
  text: "user preferences",
  userId: new UserId({ eq: "alice" }),
  batchSize: 50,
})) {
  console.log(memory.text);
}
```

## Error Handling

```typescript
import {
  MemoryClientError,
  MemoryNotFoundError,
  MemoryServerError,
  MemoryValidationError,
} from "agent-memory-client";

try {
  const memory = await client.getLongTermMemory("invalid-id");
  if (memory === null) {
    console.log("Memory not found");
  }
} catch (error) {
  if (error instanceof MemoryNotFoundError) {
    console.log("Memory does not exist");
  } else if (error instanceof MemoryServerError) {
    console.log(`Server error: ${error.message}`);
  } else if (error instanceof MemoryValidationError) {
    console.log(`Invalid input: ${error.message}`);
  } else if (error instanceof MemoryClientError) {
    console.log(`Client error: ${error.message}`);
  }
}
```

## Memory Prompt

```typescript
import type { MemoryPromptRequest } from "agent-memory-client";

const request: MemoryPromptRequest = {
  query: "What are the user's preferences?",
  session: {
    session_id: "session-123",
    user_id: "alice",
    model_name: "gpt-4o",
  },
  long_term_search: {
    text: "user preferences",
    limit: 5,
  },
};

const context = await client.memoryPrompt(request);
// Use context.messages with your LLM
```

## Type Exports

The SDK exports all types for TypeScript usage:

```typescript
import type {
  // Client config
  MemoryClientConfig,
  SearchOptions,
  // Models
  WorkingMemory,
  WorkingMemoryResponse,
  MemoryMessage,
  MemoryRecord,
  MemoryRecordResults,
  // Forget
  ForgetPolicy,
  ForgetResponse,
  // Summary Views
  SummaryView,
  CreateSummaryViewRequest,
  SummaryViewPartitionResult,
  // Tasks
  Task,
  TaskStatus,
} from "agent-memory-client";
```
