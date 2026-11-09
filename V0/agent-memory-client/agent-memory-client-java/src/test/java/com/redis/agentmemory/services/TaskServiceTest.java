package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
import com.redis.agentmemory.exceptions.MemoryNotFoundException;
import com.redis.agentmemory.models.task.Task;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TaskService functionality.
 */
class TaskServiceTest {

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
    void testGetTask_Pending() throws Exception {
        // Prepare mock response
        Task expectedTask = new Task("task-123", "summary_view_run", "pending");
        expectedTask.setViewId("view-456");
        expectedTask.setCreatedAt("2025-01-01T00:00:00Z");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedTask))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Task result = client.tasks().getTask("task-123");

        // Verify
        assertNotNull(result);
        assertEquals("task-123", result.getId());
        assertEquals("summary_view_run", result.getType());
        assertEquals("pending", result.getStatus());
        assertEquals("view-456", result.getViewId());
        assertEquals("2025-01-01T00:00:00Z", result.getCreatedAt());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/tasks/task-123"));
    }

    @Test
    void testGetTask_Completed() throws Exception {
        // Prepare mock response
        Task expectedTask = new Task("task-456", "summary_view_run", "completed");
        expectedTask.setViewId("view-789");
        expectedTask.setCreatedAt("2025-01-01T00:00:00Z");
        expectedTask.setStartedAt("2025-01-01T00:00:01Z");
        expectedTask.setCompletedAt("2025-01-01T00:05:00Z");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedTask))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Task result = client.tasks().getTask("task-456");

        // Verify
        assertNotNull(result);
        assertEquals("task-456", result.getId());
        assertEquals("completed", result.getStatus());
        assertNotNull(result.getStartedAt());
        assertNotNull(result.getCompletedAt());
        assertNull(result.getErrorMessage());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/tasks/task-456"));
    }

    @Test
    void testGetTask_Failed() throws Exception {
        // Prepare mock response
        Task expectedTask = new Task("task-789", "summary_view_run", "failed");
        expectedTask.setViewId("view-abc");
        expectedTask.setCreatedAt("2025-01-01T00:00:00Z");
        expectedTask.setStartedAt("2025-01-01T00:00:01Z");
        expectedTask.setCompletedAt("2025-01-01T00:01:00Z");
        expectedTask.setErrorMessage("Failed to generate summary: LLM error");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedTask))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Task result = client.tasks().getTask("task-789");

        // Verify
        assertNotNull(result);
        assertEquals("task-789", result.getId());
        assertEquals("failed", result.getStatus());
        assertEquals("Failed to generate summary: LLM error", result.getErrorMessage());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/tasks/task-789"));
    }

    @Test
    void testGetTask_NotFound() throws Exception {
        // Prepare mock 404 response
        mockServer.enqueue(new MockResponse()
                .setResponseCode(404)
                .setBody("{\"detail\": \"Task not found\"}")
                .addHeader("Content-Type", "application/json"));

        // Execute and verify exception
        assertThrows(MemoryNotFoundException.class, () -> {
            client.tasks().getTask("nonexistent-task");
        });

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/tasks/nonexistent-task"));
    }
}
