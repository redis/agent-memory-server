package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.summaryview.*;
import com.redis.agentmemory.models.task.Task;
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
 * Tests for SummaryViewService functionality.
 */
class SummaryViewServiceTest {

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
    void testListSummaryViews() throws Exception {
        // Prepare mock response
        List<SummaryView> views = new ArrayList<>();
        SummaryView view = new SummaryView("view-123", "long_term_memory", List.of("user_id"));
        view.setName("User Summary");
        views.add(view);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(views))
                .addHeader("Content-Type", "application/json"));

        // Execute
        List<SummaryView> result = client.summaryViews().listSummaryViews();

        // Verify
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals("view-123", result.get(0).getId());
        assertEquals("User Summary", result.get(0).getName());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views"));
    }

    @Test
    void testCreateSummaryView() throws Exception {
        // Prepare mock response
        SummaryView expectedView = new SummaryView("view-456", "long_term_memory", List.of("session_id"));
        expectedView.setName("Session Summary");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedView))
                .addHeader("Content-Type", "application/json"));

        // Execute
        CreateSummaryViewRequest createRequest = new CreateSummaryViewRequest("long_term_memory", List.of("session_id"));
        createRequest.setName("Session Summary");
        SummaryView result = client.summaryViews().createSummaryView(createRequest);

        // Verify
        assertNotNull(result);
        assertEquals("view-456", result.getId());
        assertEquals("Session Summary", result.getName());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views"));
    }

    @Test
    void testGetSummaryView() throws Exception {
        // Prepare mock response
        SummaryView expectedView = new SummaryView("view-789", "long_term_memory", List.of("user_id", "namespace"));
        expectedView.setName("Test View");
        expectedView.setTimeWindowDays(30);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedView))
                .addHeader("Content-Type", "application/json"));

        // Execute
        SummaryView result = client.summaryViews().getSummaryView("view-789");

        // Verify
        assertNotNull(result);
        assertEquals("view-789", result.getId());
        assertEquals("Test View", result.getName());
        assertEquals(30, result.getTimeWindowDays());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views/view-789"));
    }

    @Test
    void testDeleteSummaryView() throws Exception {
        // Prepare mock response
        AckResponse expectedResponse = new AckResponse();
        expectedResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        AckResponse result = client.summaryViews().deleteSummaryView("view-123");

        // Verify
        assertNotNull(result);
        assertEquals("ok", result.getStatus());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("DELETE", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views/view-123"));
    }

    @Test
    void testRunSummaryViewPartition() throws Exception {
        // Prepare mock response
        SummaryViewPartitionResult expectedResult = new SummaryViewPartitionResult(
                "view-123",
                Map.of("user_id", "user-456"),
                "User has been working on project X.",
                10
        );
        expectedResult.setComputedAt("2025-01-01T00:00:00Z");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResult))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, String> group = new HashMap<>();
        group.put("user_id", "user-456");
        SummaryViewPartitionResult result = client.summaryViews().runSummaryViewPartition("view-123", group);

        // Verify
        assertNotNull(result);
        assertEquals("view-123", result.getViewId());
        assertEquals("User has been working on project X.", result.getSummary());
        assertEquals(10, result.getMemoryCount());
        assertEquals("user-456", result.getGroup().get("user_id"));

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views/view-123/partitions/run"));
    }

    @Test
    void testListSummaryViewPartitions() throws Exception {
        // Prepare mock response
        List<SummaryViewPartitionResult> partitions = new ArrayList<>();
        SummaryViewPartitionResult partition1 = new SummaryViewPartitionResult(
                "view-123", Map.of("user_id", "user-1"), "Summary 1", 5);
        SummaryViewPartitionResult partition2 = new SummaryViewPartitionResult(
                "view-123", Map.of("user_id", "user-2"), "Summary 2", 8);
        partitions.add(partition1);
        partitions.add(partition2);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(partitions))
                .addHeader("Content-Type", "application/json"));

        // Execute
        List<SummaryViewPartitionResult> result = client.summaryViews().listSummaryViewPartitions("view-123", 10, 0);

        // Verify
        assertNotNull(result);
        assertEquals(2, result.size());
        assertEquals("Summary 1", result.get(0).getSummary());
        assertEquals("Summary 2", result.get(1).getSummary());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views/view-123/partitions"));
        assertTrue(request.getPath().contains("limit=10"));
        assertTrue(request.getPath().contains("offset=0"));
    }

    @Test
    void testRunSummaryView() throws Exception {
        // Prepare mock response
        Task expectedTask = new Task("task-abc", "summary_view_run", "pending");
        expectedTask.setViewId("view-123");
        expectedTask.setCreatedAt("2025-01-01T00:00:00Z");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedTask))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Task result = client.summaryViews().runSummaryView("view-123", true);

        // Verify
        assertNotNull(result);
        assertEquals("task-abc", result.getId());
        assertEquals("summary_view_run", result.getType());
        assertEquals("pending", result.getStatus());
        assertEquals("view-123", result.getViewId());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertTrue(request.getPath().contains("/v1/summary-views/view-123/run"));
        assertTrue(request.getPath().contains("force=true"));
    }
}
