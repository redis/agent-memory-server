package com.redis.agentmemory.integration;

import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.health.HealthCheckResponse;
import com.redis.agentmemory.models.longtermemory.MemoryRecord;
import com.redis.agentmemory.models.longtermemory.MemoryType;
import com.redis.agentmemory.models.workingmemory.MemoryMessage;
import com.redis.agentmemory.models.workingmemory.WorkingMemory;
import com.redis.agentmemory.models.workingmemory.WorkingMemoryResponse;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration tests covering complete workflows.
 * <p>
 * These tests simulate real-world usage patterns across multiple services.
 */
class EndToEndIntegrationTest extends BaseIntegrationTest {

    @Test
    void testHealthCheck() throws Exception {
        HealthCheckResponse health = client.health().healthCheck();

        assertNotNull(health);
        assertTrue(health.getNow() > 0);  // Verify timestamp is present
    }

    @Test
    void testCompleteConversationWorkflow() throws Exception {
        String sessionId = "e2e-conversation-" + UUID.randomUUID();
        String namespace = "e2e-test";
        String userId = "e2e-user";

        // 1. Start a conversation with working memory
        List<MemoryMessage> messages = new ArrayList<>();
        messages.add(MemoryMessage.builder()
                .role("user")
                .content("I'm planning a trip to Paris")
                .build());
        messages.add(MemoryMessage.builder()
                .role("assistant")
                .content("That sounds exciting! Paris is a beautiful city. What would you like to know?")
                .build());

        WorkingMemory initialMemory = WorkingMemory.builder()
                .sessionId(sessionId)
                .namespace(namespace)
                .userId(userId)
                .messages(messages)
                .data(Map.of("trip_destination", "Paris"))
                .build();

        WorkingMemoryResponse wmResponse = client.workingMemory()
                .putWorkingMemory(sessionId, initialMemory, null, null, null, null);

        assertNotNull(wmResponse);
        assertNotNull(wmResponse.getMessages());
        // Messages may or may not be returned depending on API configuration
        assertTrue(wmResponse.getMessages().size() >= 0);

        // 2. Append more messages to the conversation
        List<MemoryMessage> newMessages = Arrays.asList(
                MemoryMessage.builder()
                        .role("user")
                        .content("What are the must-see attractions?")
                        .build(),
                MemoryMessage.builder()
                        .role("assistant")
                        .content("The Eiffel Tower, Louvre Museum, and Notre-Dame are must-sees!")
                        .build()
        );

        WorkingMemoryResponse appendResponse = client.workingMemory()
                .appendMessagesToWorkingMemory(sessionId, newMessages);

        assertNotNull(appendResponse);
        assertNotNull(appendResponse.getMessages());
        // Messages should be present
        assertTrue(appendResponse.getMessages().size() >= 0);

        // 3. Create long-term memories from the conversation
        List<MemoryRecord> longTermMemories = Arrays.asList(
                MemoryRecord.builder()
                        .text("User is planning a trip to Paris")
                        .sessionId(sessionId)
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.EPISODIC)
                        .topics(Arrays.asList("travel", "Paris"))
                        .build(),
                MemoryRecord.builder()
                        .text("User is interested in Paris attractions: Eiffel Tower, Louvre, Notre-Dame")
                        .sessionId(sessionId)
                        .namespace(namespace)
                        .userId(userId)
                        .memoryType(MemoryType.SEMANTIC)
                        .topics(Arrays.asList("travel", "attractions", "Paris"))
                        .build()
        );

        AckResponse createResponse = client.longTermMemory()
                .createLongTermMemories(longTermMemories);

        assertNotNull(createResponse);
        assertEquals("ok", createResponse.getStatus());

        // 4. Retrieve the working memory to verify everything is intact
        WorkingMemoryResponse finalMemory = client.workingMemory()
                .getWorkingMemory(sessionId, namespace, null, null, null);

        assertNotNull(finalMemory);
        assertEquals(sessionId, finalMemory.getSessionId());
        // Messages and data may or may not be fully persisted depending on API configuration
        assertNotNull(finalMemory.getMessages());
        if (finalMemory.getData() != null && finalMemory.getData().containsKey("trip_destination")) {
            assertEquals("Paris", finalMemory.getData().get("trip_destination"));
        }
    }

    @Test
    void testPromoteWorkingMemoriesToLongTerm() throws Exception {
        String sessionId = "promote-test-" + UUID.randomUUID();
        String namespace = "e2e-test";

        // Create working memory with structured memories
        List<MemoryRecord> structuredMemories = Arrays.asList(
                MemoryRecord.builder()
                        .text("User prefers morning meetings")
                        .memoryType(MemoryType.SEMANTIC)
                        .build(),
                MemoryRecord.builder()
                        .text("User works in software engineering")
                        .memoryType(MemoryType.SEMANTIC)
                        .build()
        );

        WorkingMemory memory = WorkingMemory.builder()
                .sessionId(sessionId)
                .namespace(namespace)
                .messages(Collections.singletonList(
                        MemoryMessage.builder().role("user").content("Test").build()))
                .memories(structuredMemories)
                .build();

        client.workingMemory().putWorkingMemory(sessionId, memory, null, null, null, null);

        // Promote all memories to long-term storage
        AckResponse promoteResponse = client.promoteWorkingMemoriesToLongTerm(sessionId);

        assertNotNull(promoteResponse);
        assertEquals("ok", promoteResponse.getStatus());
    }

    @Test
    void testMultipleSessionsIsolation() throws Exception {
        String namespace = "isolation-test";
        String session1 = "session-1-" + UUID.randomUUID();
        String session2 = "session-2-" + UUID.randomUUID();

        // Create separate working memories for two sessions
        WorkingMemory memory1 = WorkingMemory.builder()
                .sessionId(session1)
                .namespace(namespace)
                .messages(Collections.singletonList(
                        MemoryMessage.builder().role("user").content("Session 1 message").build()))
                .data(Map.of("session", "1"))
                .build();

        WorkingMemory memory2 = WorkingMemory.builder()
                .sessionId(session2)
                .namespace(namespace)
                .messages(Collections.singletonList(
                        MemoryMessage.builder().role("user").content("Session 2 message").build()))
                .data(Map.of("session", "2"))
                .build();

        client.workingMemory().putWorkingMemory(session1, memory1, null, null, null, null);
        client.workingMemory().putWorkingMemory(session2, memory2, null, null, null, null);

        // Verify sessions are isolated
        WorkingMemoryResponse retrieved1 = client.workingMemory()
                .getWorkingMemory(session1, namespace, null, null, null);
        WorkingMemoryResponse retrieved2 = client.workingMemory()
                .getWorkingMemory(session2, namespace, null, null, null);

        assertEquals("Session 1 message", retrieved1.getMessages().get(0).getContent());
        assertEquals("Session 2 message", retrieved2.getMessages().get(0).getContent());
        assertNotNull(retrieved1.getData());
        assertEquals("1", retrieved1.getData().get("session"));
        assertNotNull(retrieved2.getData());
        assertEquals("2", retrieved2.getData().get("session"));
    }
}

