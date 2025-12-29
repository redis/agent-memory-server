package com.redis.agentmemory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.exceptions.*;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.longtermemory.*;
import com.redis.agentmemory.models.workingmemory.*;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class MemoryAPIClientTest {

    private MockWebServer mockServer;
    private MemoryAPIClient client;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() throws IOException {
        mockServer = new MockWebServer();
        mockServer.start();

        String baseUrl = mockServer.url("/").toString();
        client = MemoryAPIClient.builder(baseUrl)
                .timeout(5.0)
                .build();

        objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        objectMapper.disable(com.fasterxml.jackson.databind.SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
    }

    @AfterEach
    void tearDown() throws Exception {
        client.close();
        mockServer.shutdown();
    }


    @Test
    void testNotFoundError() {
        // Mock 404 response
        mockServer.enqueue(new MockResponse()
                .setResponseCode(404)
                .setBody("{\"detail\": \"Session not found\"}")
                .addHeader("Content-Type", "application/json"));

        // Execute and verify exception
        assertThrows(MemoryNotFoundException.class, () -> client.workingMemory().getWorkingMemory("nonexistent", null, null, null, null));
    }

    @Test
    void testServerError() {
        // Mock 500 response
        mockServer.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("{\"detail\": \"Internal server error\"}")
                .addHeader("Content-Type", "application/json"));

        // Execute and verify exception
        MemoryServerException exception = assertThrows(MemoryServerException.class, () -> client.health().healthCheck());

        assertEquals(500, exception.getStatusCode());
    }

    @Test
    void testValidationError() {
        // Mock 422 response
        mockServer.enqueue(new MockResponse()
                .setResponseCode(422)
                .setBody("{\"detail\": \"Validation error\"}")
                .addHeader("Content-Type", "application/json"));

        // Execute and verify exception
        assertThrows(MemoryServerException.class, () -> client.longTermMemory().createLongTermMemories(new ArrayList<>()));
    }

    // ===== Tests for Enhanced Working Memory Methods =====

    @Test
    void testSetWorkingMemoryData() throws Exception {
        // Mock get or create response
        WorkingMemoryResponse getResponse = new WorkingMemoryResponse();
        getResponse.setSessionId("session-123");
        getResponse.setNamespace("test-ns");
        getResponse.setMessages(new ArrayList<>());
        getResponse.setMemories(new ArrayList<>());
        getResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Mock put response
        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, Object> data = new HashMap<>();
        data.put("key1", "value1");
        data.put("key2", 42);

        WorkingMemoryResponse response = client.workingMemory().setWorkingMemoryData(
                "session-123", data, "test-ns", null);

        // Verify
        assertNotNull(response);
        assertEquals(2, mockServer.getRequestCount()); // GET + PUT
    }

    @Test
    void testAppendMessagesToWorkingMemory() throws Exception {
        // Mock get or create response
        List<MemoryMessage> existingMessages = Collections.singletonList(
                MemoryMessage.builder().role("user").content("Hello").build()
        );

        WorkingMemoryResponse getResponse = new WorkingMemoryResponse();
        getResponse.setSessionId("session-123");
        getResponse.setNamespace("test-ns");
        getResponse.setMessages(existingMessages);
        getResponse.setMemories(new ArrayList<>());
        getResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Mock put response
        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        List<MemoryMessage> newMessages = Collections.singletonList(
                MemoryMessage.builder().role("assistant").content("Hi there!").build()
        );

        WorkingMemoryResponse response = client.workingMemory().appendMessagesToWorkingMemory(
                "session-123", newMessages, "test-ns", null, null, null);

        // Verify
        assertNotNull(response);
        assertEquals(2, mockServer.getRequestCount()); // GET + PUT
    }

    // ===== Tests for Long-Term Memory CRUD =====

    @Test
    void testGetLongTermMemory() throws Exception {
        // Mock response
        MemoryRecord expectedRecord = MemoryRecord.builder()
                .id("01HQXYZ123")
                .text("Test memory")
                .memoryType(MemoryType.SEMANTIC)
                .build();

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedRecord))
                .addHeader("Content-Type", "application/json"));

        // Execute
        MemoryRecord record = client.longTermMemory().getLongTermMemory("01HQXYZ123");

        // Verify
        assertNotNull(record);
        assertEquals("01HQXYZ123", record.getId());
        assertEquals("Test memory", record.getText());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/long-term-memory/01HQXYZ123"));
    }

    @Test
    void testEditLongTermMemory() throws Exception {
        // Mock response
        AckResponse ackResponse = new AckResponse();
        ackResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(ackResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, Object> updates = new HashMap<>();
        updates.put("text", "Updated memory");

        AckResponse response = client.longTermMemory().editLongTermMemory("01HQXYZ123", updates);

        // Verify
        assertNotNull(response);
        assertEquals("ok", response.getStatus());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("PATCH", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/long-term-memory/01HQXYZ123"));
    }

    @Test
    void testDeleteLongTermMemories() throws Exception {
        // Mock response
        AckResponse expectedResponse = new AckResponse();
        expectedResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        List<String> memoryIds = Arrays.asList("01HQXYZ123", "01HQXYZ456");
        AckResponse response = client.longTermMemory().deleteLongTermMemories(memoryIds);

        // Verify
        assertNotNull(response);
        assertEquals("ok", response.getStatus());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("DELETE", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("memory_ids=01HQXYZ123"));
        assertTrue(request.getPath().contains("memory_ids=01HQXYZ456"));
    }

    // ===== Tests for Memory Hydration =====

    // ===== Tests for Validation =====

    @Test
    void testValidateMemoryRecord_Valid() {
        MemoryRecord validRecord = MemoryRecord.builder()
                .text("Valid memory text")
                .memoryType(MemoryType.SEMANTIC)
                .build();

        // Should not throw
        assertDoesNotThrow(() -> client.validateMemoryRecord(validRecord));
    }

    @Test
    void testValidateMemoryRecord_EmptyText() {
        MemoryRecord invalidRecord = MemoryRecord.builder()
                .text("")
                .memoryType(MemoryType.SEMANTIC)
                .build();

        // Should throw
        assertThrows(MemoryValidationException.class, () ->
                client.validateMemoryRecord(invalidRecord));
    }

    @Test
    void testValidateMemoryRecord_InvalidULID() {
        MemoryRecord invalidRecord = MemoryRecord.builder()
                .id("invalid-id-format")
                .text("Valid text")
                .memoryType(MemoryType.SEMANTIC)
                .build();

        // Should throw
        assertThrows(MemoryValidationException.class, () ->
                client.validateMemoryRecord(invalidRecord));
    }

    @Test
    void testValidateSearchFilters_Valid() {
        Map<String, Object> validFilters = new HashMap<>();
        validFilters.put("limit", 10);
        validFilters.put("offset", 0);
        validFilters.put("distance_threshold", 0.5);

        // Should not throw
        assertDoesNotThrow(() -> client.validateSearchFilters(validFilters));
    }

    @Test
    void testValidateSearchFilters_InvalidKey() {
        Map<String, Object> invalidFilters = new HashMap<>();
        invalidFilters.put("invalid_key", "value");

        // Should throw
        assertThrows(MemoryValidationException.class, () ->
                client.validateSearchFilters(invalidFilters));
    }

    @Test
    void testValidateSearchFilters_InvalidLimit() {
        Map<String, Object> invalidFilters = new HashMap<>();
        invalidFilters.put("limit", -1);

        // Should throw
        assertThrows(MemoryValidationException.class, () ->
                client.validateSearchFilters(invalidFilters));
    }

    // ===== Tests for Phase 2: Lifecycle Management =====

    @Test
    void testPromoteWorkingMemoriesToLongTerm_AllMemories() throws Exception {
        // Mock get or create response with memories
        WorkingMemoryResponse getResponse = new WorkingMemoryResponse();
        getResponse.setSessionId("session-123");
        getResponse.setNamespace("test-ns");
        getResponse.setMessages(new ArrayList<>());

        List<MemoryRecord> memories = Arrays.asList(
                MemoryRecord.builder().id("01HQXYZ123").text("Memory 1").build(),
                MemoryRecord.builder().id("01HQXYZ456").text("Memory 2").build()
        );
        getResponse.setMemories(memories);
        getResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Mock create long-term memories response
        AckResponse ackResponse = new AckResponse();
        ackResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(ackResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        AckResponse response = client.promoteWorkingMemoriesToLongTerm("session-123");

        // Verify
        assertNotNull(response);
        assertEquals("ok", response.getStatus());
        assertEquals(2, mockServer.getRequestCount()); // GET + CREATE
    }

    @Test
    void testPromoteWorkingMemoriesToLongTerm_SpecificMemories() throws Exception {
        // Mock get or create response with memories
        WorkingMemoryResponse getResponse = new WorkingMemoryResponse();
        getResponse.setSessionId("session-123");
        getResponse.setNamespace("test-ns");
        getResponse.setMessages(new ArrayList<>());

        List<MemoryRecord> memories = Arrays.asList(
                MemoryRecord.builder().id("01HQXYZ123").text("Memory 1").build(),
                MemoryRecord.builder().id("01HQXYZ456").text("Memory 2").build(),
                MemoryRecord.builder().id("01HQXYZ789").text("Memory 3").build()
        );
        getResponse.setMemories(memories);
        getResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Mock create long-term memories response
        AckResponse ackResponse = new AckResponse();
        ackResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(ackResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute - promote only specific memories
        List<String> memoryIds = Arrays.asList("01HQXYZ123", "01HQXYZ789");
        AckResponse response = client.promoteWorkingMemoriesToLongTerm("session-123", memoryIds);

        // Verify
        assertNotNull(response);
        assertEquals("ok", response.getStatus());
        assertEquals(2, mockServer.getRequestCount()); // GET + CREATE
    }

    @Test
    void testPromoteWorkingMemoriesToLongTerm_NoMemories() throws Exception {
        // Mock get or create response with no memories
        WorkingMemoryResponse getResponse = new WorkingMemoryResponse();
        getResponse.setSessionId("session-123");
        getResponse.setNamespace("test-ns");
        getResponse.setMessages(new ArrayList<>());
        getResponse.setMemories(new ArrayList<>()); // Empty
        getResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(getResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        AckResponse response = client.promoteWorkingMemoriesToLongTerm("session-123");

        // Verify - should return ok without making create request
        assertNotNull(response);
        assertEquals("ok", response.getStatus());
        assertEquals(1, mockServer.getRequestCount()); // Only GET, no CREATE
    }

    // ===== Tests for Phase 2: Convenience Overloads =====
}
