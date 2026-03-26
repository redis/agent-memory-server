# Java SDK

The Java SDK (`agent-memory-client-java`) provides a type-safe client for integrating memory capabilities into JVM-based applications.

**Version**: 0.1.0+
**Requirements**: Java 21 or higher

## Installation

### Gradle (Kotlin DSL)

```kotlin
dependencies {
    implementation("com.redis:agent-memory-client-java:0.1.0")
}
```

### Gradle (Groovy)

```groovy
dependencies {
    implementation 'com.redis:agent-memory-client-java:0.1.0'
}
```

### Maven

```xml
<dependency>
    <groupId>com.redis</groupId>
    <artifactId>agent-memory-client-java</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

```java
import com.redis.agentmemory.MemoryAPIClient;
import com.redis.agentmemory.models.longtermemory.*;
import java.util.*;

// Create client
MemoryAPIClient client = MemoryAPIClient.builder("http://localhost:8000")
    .defaultNamespace("my-app")
    .timeout(30.0)
    .build();

// Store a memory
MemoryRecord memory = MemoryRecord.builder()
    .text("User prefers morning meetings")
    .memoryType(MemoryType.SEMANTIC)
    .topics(List.of("scheduling", "preferences"))
    .userId("alice")
    .build();

client.longTermMemory().createLongTermMemories(List.of(memory));

// Search memories
SearchRequest request = SearchRequest.builder()
    .text("when does user prefer meetings")
    .userId("alice")
    .topics(List.of("scheduling"))
    .limit(5)
    .build();

MemoryRecordResults results = client.longTermMemory().searchLongTermMemories(request);

for (MemoryRecordResult result : results.getMemories()) {
    System.out.printf("%s (distance: %.3f)%n", result.getText(), result.getDist());
}

// Clean up
client.close();
```

## Client Configuration

The client uses the Builder pattern:

```java
MemoryAPIClient client = MemoryAPIClient.builder("http://localhost:8000")
    .timeout(30.0)                     // Request timeout (seconds)
    .defaultNamespace("production")     // Default namespace
    .defaultModelName("gpt-4o")        // For auto-summarization
    .defaultContextWindowMax(128000)    // Context window limit
    .build();
```

The client implements `AutoCloseable` for try-with-resources:

```java
try (MemoryAPIClient client = MemoryAPIClient.builder("http://localhost:8000").build()) {
    // Use client
}
```

## Service Architecture

The Java SDK uses a service-based architecture. Access services through the client:

```java
client.health()           // HealthService - health checks
client.workingMemory()    // WorkingMemoryService - session management
client.longTermMemory()   // LongTermMemoryService - persistent memories
client.hydration()        // MemoryHydrationService - prompt hydration
client.summaryViews()     // SummaryViewService - summary views
client.tasks()            // TaskService - background tasks
```

## Memory Operations

### Creating Memories

```java
List<MemoryRecord> memories = List.of(
    MemoryRecord.builder()
        .text("User works at TechCorp")
        .memoryType(MemoryType.SEMANTIC)
        .topics(List.of("career", "work"))
        .entities(List.of("TechCorp"))
        .userId("alice")
        .build()
);

client.longTermMemory().createLongTermMemories(memories);
```

### Searching Memories

```java
// Using builder pattern
SearchRequest request = SearchRequest.builder()
    .text("user preferences")
    .namespace("my-app")
    .userId("alice")
    .topics(List.of("preferences"))
    .limit(10)
    .offset(0)
    .build();

MemoryRecordResults results = client.longTermMemory().searchLongTermMemories(request);

// Simple text search
MemoryRecordResults simpleResults = client.longTermMemory()
    .searchLongTermMemories("user preferences");

// Keyword search - exact term matching
SearchRequest keywordRequest = SearchRequest.builder()
    .text("TechCorp engineer")
    .searchMode("keyword")
    .limit(10)
    .build();

// Hybrid search - combines semantic and keyword matching
SearchRequest hybridRequest = SearchRequest.builder()
    .text("user preferences")
    .searchMode("hybrid")
    .hybridAlpha(0.7)  // 0.0=keyword, 1.0=semantic
    .limit(10)
    .build();
```

### Get, Edit, and Delete

```java
// Get a specific memory
MemoryRecord memory = client.longTermMemory().getLongTermMemory("memory-id");

// Edit a memory
Map<String, Object> updates = Map.of(
    "text", "Updated text content",
    "topics", List.of("updated", "topics")
);
client.longTermMemory().editLongTermMemory("memory-id", updates);

// Delete memories
client.longTermMemory().deleteLongTermMemories(List.of("id1", "id2"));
```

## Working Memory

```java
import com.redis.agentmemory.models.workingmemory.*;

// Get or create working memory
WorkingMemoryResult result = client.workingMemory()
    .getOrCreateWorkingMemory("session-123");

boolean wasCreated = result.isCreated();
WorkingMemoryResponse memory = result.getMemory();

// Update with messages
WorkingMemory update = WorkingMemory.builder()
    .sessionId("session-123")
    .messages(List.of(
        new MemoryMessage("user", "I'm planning a trip to Italy"),
        new MemoryMessage("assistant", "That sounds exciting!")
    ))
    .data(Map.of("destination", "Italy"))
    .build();

client.workingMemory().putWorkingMemory("session-123", update);

// Append messages (more efficient than full update)
List<MemoryMessage> newMessages = List.of(
    new MemoryMessage("user", "What are the best places?")
);
client.workingMemory().appendMessagesToWorkingMemory("session-123", newMessages);

// Delete session
client.workingMemory().deleteWorkingMemory("session-123");
```

## Forgetting Memories

```java
Map<String, Object> policy = Map.of(
    "max_age_days", 90,
    "max_inactive_days", 30,
    "budget", 100,
    "memory_type_allowlist", List.of("episodic")
);

// Preview (dry run)
ForgetResponse preview = client.longTermMemory().forgetLongTermMemories(
    policy,
    "my-app",  // namespace
    null,      // userId
    null,      // sessionId
    1000,      // limit
    true,      // dryRun
    List.of("keep-this-id")  // pinnedIds
);
System.out.printf("Would delete %d of %d%n", preview.getDeleted(), preview.getScanned());

// Execute
ForgetResponse result = client.longTermMemory().forgetLongTermMemories(
    policy, "my-app", null, null, 1000, false, null
);
```

## Bulk Operations

```java
// Bulk create with rate limiting
List<List<MemoryRecord>> batches = List.of(memories1, memories2, memories3);
List<AckResponse> results = client.longTermMemory().bulkCreateLongTermMemories(
    batches,
    50,   // batchSize
    100   // delayBetweenBatchesMs
);

// Auto-paginating search with Iterator
Iterator<MemoryRecord> iterator = client.longTermMemory().searchAllLongTermMemories(
    "user preferences",  // text
    null,               // sessionId
    "my-app",           // namespace
    null,               // topics
    null,               // entities
    "alice",            // userId
    50                  // batchSize
);

while (iterator.hasNext()) {
    MemoryRecord memory = iterator.next();
    System.out.println(memory.getText());
}

// Or use Stream API
Stream<MemoryRecord> stream = client.longTermMemory().searchAllLongTermMemoriesStream(
    "user preferences", null, "my-app", null, null, "alice", 50
);
stream.forEach(m -> System.out.println(m.getText()));
```

## Summary Views

```java
import com.redis.agentmemory.models.summaryview.*;

// Create a summary view
CreateSummaryViewRequest request = CreateSummaryViewRequest.builder()
    .name("User Topic Summaries")
    .source("long_term")
    .groupBy(List.of("user_id", "topics"))
    .timeWindowDays(30)
    .continuous(true)
    .build();

SummaryView view = client.summaryViews().createSummaryView(request);

// Run a partition
Map<String, String> group = Map.of("user_id", "alice", "topics", "travel");
SummaryViewPartitionResult partition = client.summaryViews()
    .runSummaryViewPartition(view.getId(), group);

// List views
List<SummaryView> views = client.summaryViews().listSummaryViews();
```

## Error Handling

```java
import com.redis.agentmemory.exceptions.*;

try {
    MemoryRecord memory = client.longTermMemory().getLongTermMemory("invalid-id");
} catch (MemoryNotFoundException e) {
    System.out.println("Memory not found: " + e.getMessage());
} catch (MemoryServerException e) {
    System.out.println("Server error: " + e.getMessage());
} catch (MemoryValidationException e) {
    System.out.println("Validation error: " + e.getMessage());
} catch (MemoryClientException e) {
    System.out.println("Client error: " + e.getMessage());
}
```

## Validation

The client provides validation utilities:

```java
// Validate a memory record
client.validateMemoryRecord(memory);

// Validate search filters
Map<String, Object> filters = Map.of(
    "limit", 10,
    "offset", 0,
    "distance_threshold", 0.5
);
client.validateSearchFilters(filters);
```
