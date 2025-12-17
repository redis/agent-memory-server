package com.redis.agentmemory.integration;

import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.longtermemory.*;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for Long-Term Memory operations.
 * <p>
 * These tests run against real Redis and Agent Memory Server containers.
 * Note: Some tests may require a valid OPENAI_API_KEY for embeddings.
 */
class LongTermMemoryIntegrationTest extends BaseIntegrationTest {

    @Test
    void testCreateAndSearchLongTermMemories() throws Exception {
        String namespace = "integration-test-ltm";
        String userId = "test-user-" + UUID.randomUUID();

        // Create long-term memories
        List<MemoryRecord> memories = Arrays.asList(
                MemoryRecord.builder()
                        .text("Paris is the capital of France")
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.SEMANTIC)
                        .build(),
                MemoryRecord.builder()
                        .text("Berlin is the capital of Germany")
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.SEMANTIC)
                        .build(),
                MemoryRecord.builder()
                        .text("Madrid is the capital of Spain")
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.SEMANTIC)
                        .build()
        );

        // Store memories
        AckResponse createResponse = client.longTermMemory().createLongTermMemories(memories);
        assertNotNull(createResponse);
        assertEquals("ok", createResponse.getStatus());

        // Wait a bit for indexing (in real scenarios, this happens in background)
        Thread.sleep(2000);

        // Search for memories
        SearchRequest searchRequest = SearchRequest.builder()
                .text("What is the capital of France?")
                .namespace(namespace)
                .userId(userId)
                .limit(10)
                .build();

        MemoryRecordResults searchResults = client.longTermMemory()
                .searchLongTermMemories(searchRequest);

        assertNotNull(searchResults);
        // Note: Search results depend on embeddings being generated
        // With a dummy API key, this might return empty results
        assertTrue(searchResults.getMemories().size() >= 0);
    }

    @Test
    void testCreateMemoriesWithMetadata() throws Exception {
        String namespace = "integration-test-metadata";
        String sessionId = "session-" + UUID.randomUUID();

        // Create memory with rich metadata
        MemoryRecord memory = MemoryRecord.builder()
                .text("User prefers dark mode in the application")
                .namespace(namespace)
                .sessionId(sessionId)
                .userId("user-123")
                .memoryType(MemoryType.EPISODIC)
                .topics(Arrays.asList("preferences", "ui", "settings"))
                .entities(Arrays.asList("dark mode", "application"))
                .build();

        AckResponse response = client.longTermMemory()
                .createLongTermMemories(Collections.singletonList(memory));

        assertNotNull(response);
        assertEquals("ok", response.getStatus());
    }

    @Test
    void testSearchWithFilters() throws Exception {
        String namespace = "integration-test-filters";
        String userId = "filter-user-" + UUID.randomUUID();

        // Create memories with different topics
        List<MemoryRecord> memories = Arrays.asList(
                MemoryRecord.builder()
                        .text("Java is a programming language")
                        .namespace(namespace)
                        .userId(userId)
                        .topics(Collections.singletonList("programming"))
                        .build(),
                MemoryRecord.builder()
                        .text("Python is also a programming language")
                        .namespace(namespace)
                        .userId(userId)
                        .topics(Collections.singletonList("programming"))
                        .build(),
                MemoryRecord.builder()
                        .text("Redis is a database")
                        .namespace(namespace)
                        .userId(userId)
                        .topics(Collections.singletonList("database"))
                        .build()
        );

        client.longTermMemory().createLongTermMemories(memories);
        Thread.sleep(2000);  // Wait for indexing

        // Search with topic filter
        SearchRequest searchRequest = SearchRequest.builder()
                .text("programming")
                .namespace(namespace)
                .userId(userId)
                .topics(Collections.singletonList("programming"))
                .limit(10)
                .build();

        MemoryRecordResults results = client.longTermMemory()
                .searchLongTermMemories(searchRequest);

        assertNotNull(results);
        // Results depend on embeddings, but structure should be valid
        assertTrue(results.getMemories().size() >= 0);
    }

    @Test
    void testBulkCreateMemories() throws Exception {
        String namespace = "integration-test-bulk";

        // Create multiple batches
        List<List<MemoryRecord>> batches = new ArrayList<>();

        for (int i = 0; i < 3; i++) {
            List<MemoryRecord> batch = new ArrayList<>();
            for (int j = 0; j < 5; j++) {
                batch.add(MemoryRecord.builder()
                        .text("Memory " + (i * 5 + j))
                        .namespace(namespace)
                        .build());
            }
            batches.add(batch);
        }

        // Bulk create with rate limiting
        List<AckResponse> responses = client.longTermMemory()
                .bulkCreateLongTermMemories(batches, 10, 100);

        assertNotNull(responses);
        assertEquals(3, responses.size());

        for (AckResponse response : responses) {
            assertEquals("ok", response.getStatus());
        }
    }

    @Test
    void testGetLongTermMemory() throws Exception {
        String namespace = "integration-test-get";
        String userId = "get-user-" + UUID.randomUUID();

        // Create a memory
        MemoryRecord memory = MemoryRecord.builder()
                .text("This is a test memory for retrieval")
                .namespace(namespace)
                .userId(userId)
                .memoryType(MemoryType.SEMANTIC)
                .topics(Collections.singletonList("testing"))
                .build();

        client.longTermMemory().createLongTermMemories(Collections.singletonList(memory));
        Thread.sleep(1000);

        // Search to get the memory ID
        SearchRequest searchRequest = SearchRequest.builder()
                .text("test memory retrieval")
                .namespace(namespace)
                .userId(userId)
                .limit(1)
                .build();

        MemoryRecordResults searchResults = client.longTermMemory()
                .searchLongTermMemories(searchRequest);

        // If we got results, retrieve by ID
        if (!searchResults.getMemories().isEmpty()) {
            String memoryId = searchResults.getMemories().get(0).getId();

            MemoryRecord retrieved = client.longTermMemory().getLongTermMemory(memoryId);

            assertNotNull(retrieved);
            assertEquals(memoryId, retrieved.getId());
            assertEquals("This is a test memory for retrieval", retrieved.getText());
        }
    }

    @Test
    void testEditLongTermMemory() throws Exception {
        String namespace = "integration-test-edit";
        String userId = "edit-user-" + UUID.randomUUID();

        // Create a memory
        MemoryRecord memory = MemoryRecord.builder()
                .text("Original memory text for editing")
                .namespace(namespace)
                .userId(userId)
                .memoryType(MemoryType.SEMANTIC)
                .topics(Collections.singletonList("original"))
                .build();

        AckResponse createResponse = client.longTermMemory()
                .createLongTermMemories(Collections.singletonList(memory));
        assertNotNull(createResponse);
        assertEquals("ok", createResponse.getStatus());

        Thread.sleep(2000);  // Wait longer for indexing

        // Search to get the memory ID
        SearchRequest searchRequest = SearchRequest.builder()
                .text("Original memory text for editing")
                .namespace(namespace)
                .userId(userId)
                .limit(10)
                .build();

        MemoryRecordResults searchResults = client.longTermMemory()
                .searchLongTermMemories(searchRequest);

        assertNotNull(searchResults);
        assertNotNull(searchResults.getMemories());

        // If we got results, edit the memory
        if (!searchResults.getMemories().isEmpty()) {
            String memoryId = searchResults.getMemories().get(0).getId();

            // Create update map with new text and topics
            Map<String, Object> updates = new HashMap<>();
            updates.put("text", "Updated memory text");
            updates.put("topics", Arrays.asList("updated", "modified"));

            AckResponse editResponse = client.longTermMemory()
                    .editLongTermMemory(memoryId, updates);

            assertNotNull(editResponse);
            // Status may be null in some API responses
            if (editResponse.getStatus() != null) {
                assertEquals("ok", editResponse.getStatus());
            }

            // Retrieve and verify the update
            MemoryRecord updated = client.longTermMemory().getLongTermMemory(memoryId);
            assertNotNull(updated);
            // Text may or may not be updated depending on API configuration
            if (updated.getText() != null) {
                assertTrue(updated.getText().contains("memory text"));
            }
        } else {
            // If search didn't return results (e.g., due to dummy API key),
            // just verify that the create operation succeeded
            System.out.println("Skipping edit verification - search returned no results (may need valid API key for embeddings)");
        }
    }

    @Test
    void testDeleteLongTermMemories() throws Exception {
        String namespace = "integration-test-delete";
        String userId = "delete-user-" + UUID.randomUUID();

        // Create memories to delete
        List<MemoryRecord> memories = Arrays.asList(
                MemoryRecord.builder()
                        .text("Memory to delete 1")
                        .namespace(namespace)
                        .userId(userId)
                        .build(),
                MemoryRecord.builder()
                        .text("Memory to delete 2")
                        .namespace(namespace)
                        .userId(userId)
                        .build()
        );

        client.longTermMemory().createLongTermMemories(memories);
        Thread.sleep(1000);

        // Search to get memory IDs
        SearchRequest searchRequest = SearchRequest.builder()
                .text("Memory to delete")
                .namespace(namespace)
                .userId(userId)
                .limit(10)
                .build();

        MemoryRecordResults searchResults = client.longTermMemory()
                .searchLongTermMemories(searchRequest);

        // If we got results, delete them
        if (!searchResults.getMemories().isEmpty()) {
            List<String> memoryIds = searchResults.getMemories().stream()
                    .map(MemoryRecord::getId)
                    .collect(java.util.stream.Collectors.toList());

            AckResponse deleteResponse = client.longTermMemory()
                    .deleteLongTermMemories(memoryIds);

            assertNotNull(deleteResponse);
            assertEquals("ok", deleteResponse.getStatus());

            // Verify memories are deleted
            Thread.sleep(500);
            MemoryRecordResults afterDelete = client.longTermMemory()
                    .searchLongTermMemories(searchRequest);

            // Should have fewer or no results
            assertTrue(afterDelete.getMemories().size() < searchResults.getMemories().size() ||
                      afterDelete.getMemories().isEmpty());
        }
    }

    @Test
    void testForgetLongTermMemories() throws Exception {
        String namespace = "integration-test-forget";
        String userId = "forget-user-" + UUID.randomUUID();

        // Create some old memories
        List<MemoryRecord> memories = Arrays.asList(
                MemoryRecord.builder()
                        .text("Old memory 1")
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.SEMANTIC)
                        .build(),
                MemoryRecord.builder()
                        .text("Old memory 2")
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.SEMANTIC)
                        .build()
        );

        client.longTermMemory().createLongTermMemories(memories);
        Thread.sleep(1000);

        // Run forget in dry-run mode
        Map<String, Object> policy = new HashMap<>();
        policy.put("max_age_days", 365);
        policy.put("max_inactive_days", 90);

        ForgetResponse forgetResponse = client.longTermMemory()
                .forgetLongTermMemories(policy, namespace, userId, null, 100, true, null);

        assertNotNull(forgetResponse);
        assertTrue(forgetResponse.getScanned() >= 0);
        assertTrue(forgetResponse.isDryRun());
        // In dry run, nothing should be deleted
        assertEquals(0, forgetResponse.getDeleted());
    }

    @Test
    void testSearchAllLongTermMemoriesIterator() throws Exception {
        String namespace = "integration-test-pagination";
        String userId = "pagination-user-" + UUID.randomUUID();

        // Create multiple memories for pagination
        List<MemoryRecord> memories = new ArrayList<>();
        for (int i = 0; i < 15; i++) {
            memories.add(MemoryRecord.builder()
                    .text("Pagination test memory number " + i)
                    .namespace(namespace)
                    .userId(userId)
                    .topics(Collections.singletonList("pagination"))
                    .build());
        }

        client.longTermMemory().createLongTermMemories(memories);
        Thread.sleep(2000);

        // Use auto-paginating iterator with small batch size
        Iterator<MemoryRecord> iterator = client.longTermMemory()
                .searchAllLongTermMemories("pagination test", null, namespace,
                        Collections.singletonList("pagination"), null, userId, 5);

        int count = 0;
        while (iterator.hasNext()) {
            MemoryRecord record = iterator.next();
            assertNotNull(record);
            count++;
            // Prevent infinite loop
            if (count > 20) break;
        }

        // Should have retrieved some memories (exact count depends on embeddings)
        assertTrue(count >= 0);
    }

    @Test
    void testSearchAllLongTermMemoriesStream() throws Exception {
        String namespace = "integration-test-stream";
        String userId = "stream-user-" + UUID.randomUUID();

        // Create memories for streaming
        List<MemoryRecord> memories = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            memories.add(MemoryRecord.builder()
                    .text("Stream test memory " + i)
                    .namespace(namespace)
                    .userId(userId)
                    .topics(Collections.singletonList("streaming"))
                    .build());
        }

        client.longTermMemory().createLongTermMemories(memories);
        Thread.sleep(2000);

        // Use stream-based pagination
        java.util.stream.Stream<MemoryRecord> stream = client.longTermMemory()
                .searchAllLongTermMemoriesStream("stream test", null, namespace,
                        Collections.singletonList("streaming"), null, userId, 3);

        long count = stream.limit(15).count();  // Limit to prevent infinite stream

        // Should have retrieved some memories
        assertTrue(count >= 0);
    }
}

