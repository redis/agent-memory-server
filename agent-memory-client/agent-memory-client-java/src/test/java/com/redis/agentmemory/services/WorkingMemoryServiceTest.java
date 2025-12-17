package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
import com.redis.agentmemory.models.common.AckResponse;
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

/**
 * Tests for WorkingMemoryService functionality.
 */
class WorkingMemoryServiceTest {

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
    void testListSessions() throws Exception {
        // Mock response
        SessionListResponse expectedResponse = new SessionListResponse();
        expectedResponse.setSessions(Arrays.asList("session-1", "session-2", "session-3"));
        expectedResponse.setTotal(3);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        SessionListResponse response = client.workingMemory().listSessions(10, 0, "test-namespace", "user-123");

        // Verify
        assertNotNull(response);
        assertEquals(3, response.getTotal());
        assertEquals(3, response.getSessions().size());
        assertTrue(response.getSessions().contains("session-1"));

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/"));
        assertTrue(request.getPath().contains("limit=10"));
        assertTrue(request.getPath().contains("offset=0"));
        assertTrue(request.getPath().contains("namespace=test-namespace"));
        assertTrue(request.getPath().contains("user_id=user-123"));
    }

    @Test
    void testGetWorkingMemory() throws Exception {
        // Mock response
        WorkingMemoryResponse expectedResponse = new WorkingMemoryResponse();
        expectedResponse.setSessionId("session-123");
        expectedResponse.setUserId("user-456");
        expectedResponse.setNamespace("test-namespace");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        WorkingMemoryResponse response = client.workingMemory().getWorkingMemory("session-123");

        // Verify
        assertNotNull(response);
        assertEquals("session-123", response.getSessionId());
        assertEquals("user-456", response.getUserId());
        assertEquals("test-namespace", response.getNamespace());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/session-123"));
    }

    @Test
    void testGetWorkingMemory_MinimalParams() throws Exception {
        // Mock response
        WorkingMemoryResponse expectedResponse = new WorkingMemoryResponse();
        expectedResponse.setSessionId("session-123");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute - using convenience method with minimal params
        WorkingMemoryResponse response = client.workingMemory().getWorkingMemory("session-123");

        // Verify
        assertNotNull(response);
        assertEquals("session-123", response.getSessionId());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/session-123"));
    }

    @Test
    void testPutWorkingMemory() throws Exception {
        // Prepare request
        WorkingMemory memory = new WorkingMemory("session-123");
        memory.setUserId("user-456");
        memory.setNamespace("test-namespace");
        memory.getMessages().add(new MemoryMessage("user", "Test message"));

        // Mock response
        WorkingMemoryResponse expectedResponse = new WorkingMemoryResponse();
        expectedResponse.setSessionId("session-123");
        expectedResponse.setMessages(memory.getMessages());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        WorkingMemoryResponse response = client.workingMemory().putWorkingMemory(
                "session-123", memory, "user-456", "test-namespace", null, null);

        // Verify
        assertNotNull(response);
        assertEquals("session-123", response.getSessionId());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("PUT", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/session-123"));
    }

    @Test
    void testDeleteWorkingMemory() throws Exception {
        // Mock response
        AckResponse expectedResponse = new AckResponse();
        expectedResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        AckResponse response = client.workingMemory().deleteWorkingMemory("session-123", "user-456", "test-namespace");

        // Verify
        assertNotNull(response);
        assertEquals("ok", response.getStatus());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("DELETE", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/session-123"));
    }

    @Test
    void testGetOrCreateWorkingMemory_ExistingMemory() throws Exception {
        // Mock response for existing memory
        WorkingMemoryResponse expectedResponse = new WorkingMemoryResponse();
        expectedResponse.setSessionId("session-123");
        expectedResponse.setNamespace("test-namespace");
        expectedResponse.setUserId("user-456");
        expectedResponse.setMessages(new ArrayList<>());
        expectedResponse.setMemories(new ArrayList<>());
        expectedResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        WorkingMemoryResult result = client.workingMemory().getOrCreateWorkingMemory("session-123", "test-namespace", "user-456", null, null, null);

        // Verify
        assertNotNull(result);
        assertFalse(result.isCreated()); // Should be false since it existed
        assertNotNull(result.getMemory());
        assertEquals("session-123", result.getMemory().getSessionId());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/session-123"));
    }

    @Test
    void testGetOrCreateWorkingMemory_CreateNew() throws Exception {
        // Mock 404 response for non-existent memory
        mockServer.enqueue(new MockResponse()
                .setResponseCode(404)
                .setBody("{\"detail\": \"Not found\"}")
                .addHeader("Content-Type", "application/json"));

        // Mock response for creating new memory
        WorkingMemoryResponse expectedResponse = new WorkingMemoryResponse();
        expectedResponse.setSessionId("session-123");
        expectedResponse.setNamespace("test-namespace");
        expectedResponse.setUserId("user-456");
        expectedResponse.setMessages(new ArrayList<>());
        expectedResponse.setMemories(new ArrayList<>());
        expectedResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        WorkingMemoryResult result = client.workingMemory().getOrCreateWorkingMemory("session-123", "test-namespace", "user-456", null, null, null);

        // Verify
        assertNotNull(result);
        assertTrue(result.isCreated()); // Should be true since it was created
        assertNotNull(result.getMemory());
        assertEquals("session-123", result.getMemory().getSessionId());

        RecordedRequest request1 = mockServer.takeRequest();
        assertEquals("GET", request1.getMethod());

        RecordedRequest request2 = mockServer.takeRequest();
        assertEquals("PUT", request2.getMethod());
        assertNotNull(request2.getPath());
        assertTrue(request2.getPath().contains("/v1/working-memory/session-123"));
    }

    @Test
    void testGetOrCreateWorkingMemory_MinimalParams() throws Exception {
        // Mock response for existing memory
        WorkingMemoryResponse expectedResponse = new WorkingMemoryResponse();
        expectedResponse.setSessionId("session-123");
        expectedResponse.setMessages(new ArrayList<>());
        expectedResponse.setMemories(new ArrayList<>());
        expectedResponse.setData(new HashMap<>());

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute - using convenience method with minimal params
        WorkingMemoryResult result = client.workingMemory().getOrCreateWorkingMemory("session-123");

        // Verify
        assertNotNull(result);
        assertFalse(result.isCreated());
        assertNotNull(result.getMemory());
        assertEquals("session-123", result.getMemory().getSessionId());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/working-memory/session-123"));
    }
}
