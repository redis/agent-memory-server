package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.longtermemory.*;
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
 * Tests for LongTermMemoryService functionality.
 */
class LongTermMemoryServiceTest {

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
    void testCreateLongTermMemories() throws Exception {
        // Prepare request
        List<MemoryRecord> memories = new ArrayList<>();

        MemoryRecord memory1 = new MemoryRecord("Test memory 1");
        memory1.setUserId("user-456");
        memory1.setNamespace("test-namespace");
        memory1.setMemoryType(MemoryType.SEMANTIC);
        memories.add(memory1);

        MemoryRecord memory2 = new MemoryRecord("Test memory 2");
        memory2.setUserId("user-456");
        memory2.setNamespace("test-namespace");
        memory2.setMemoryType(MemoryType.EPISODIC);
        memories.add(memory2);

        // Mock response
        AckResponse expectedResponse = new AckResponse();
        expectedResponse.setStatus("ok");

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        AckResponse response = client.longTermMemory().createLongTermMemories(memories);

        // Verify
        assertNotNull(response);
        assertEquals("ok", response.getStatus());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/long-term-memory"));
    }

    @Test
    void testSearchLongTermMemories() throws Exception {
        // Mock response
        MemoryRecordResults expectedResponse = new MemoryRecordResults();

        List<MemoryRecordResult> memories = new ArrayList<>();
        MemoryRecordResult result1 = new MemoryRecordResult();
        result1.setText("Test memory 1");
        result1.setDist(0.1);
        result1.setTopics(Arrays.asList("topic1", "topic2"));
        memories.add(result1);

        expectedResponse.setMemories(memories);
        expectedResponse.setTotal(1);
        expectedResponse.setNextOffset(null);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        SearchRequest searchRequest = SearchRequest.builder()
                .text("test query")
                .limit(10)
                .offset(0)
                .namespace("test-namespace")
                .userId("user-456")
                .topics(List.of("topic1"))
                .build();
        MemoryRecordResults response = client.longTermMemory().searchLongTermMemories(searchRequest);

        // Verify
        assertNotNull(response);
        assertEquals(1, response.getTotal());
        assertEquals(1, response.getMemories().size());
        assertEquals("Test memory 1", response.getMemories().get(0).getText());
        assertEquals(0.1, response.getMemories().get(0).getDist());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/long-term-memory/search"));
    }

    @Test
    void testSearchLongTermMemories_MinimalParams() throws Exception {
        // Mock response
        MemoryRecordResults expectedResponse = new MemoryRecordResults();
        expectedResponse.setMemories(new ArrayList<>());
        expectedResponse.setTotal(0);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute - using convenience method with minimal params
        MemoryRecordResults response = client.longTermMemory().searchLongTermMemories("test query");

        // Verify
        assertNotNull(response);
        assertEquals(0, response.getTotal());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/long-term-memory/search"));
    }

    @Test
    void testForgetLongTermMemories_DryRun() throws Exception {
        // Mock response
        ForgetResponse expectedResponse = new ForgetResponse();
        expectedResponse.setScanned(100);
        expectedResponse.setDeleted(10);
        expectedResponse.setDeletedIds(Arrays.asList("01HQXYZ123", "01HQXYZ456"));
        expectedResponse.setDryRun(true);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, Object> policy = new HashMap<>();
        policy.put("max_age_days", 180);
        policy.put("max_inactive_days", 90);
        policy.put("budget", null);
        policy.put("memory_type_allowlist", null);

        ForgetResponse response = client.longTermMemory().forgetLongTermMemories(
                policy, "test-ns", "user-123", null, 1000, true, null);

        // Verify
        assertNotNull(response);
        assertEquals(100, response.getScanned());
        assertEquals(10, response.getDeleted());
        assertEquals(2, response.getDeletedIds().size());
        assertTrue(response.isDryRun());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/long-term-memory/forget"));
        assertTrue(request.getPath().contains("namespace=test-ns"));
        assertTrue(request.getPath().contains("user_id=user-123"));
        assertTrue(request.getPath().contains("limit=1000"));
        assertTrue(request.getPath().contains("dry_run=true"));
    }

    @Test
    void testForgetLongTermMemories_ActualDeletion() throws Exception {
        // Mock response
        ForgetResponse expectedResponse = new ForgetResponse();
        expectedResponse.setScanned(50);
        expectedResponse.setDeleted(5);
        expectedResponse.setDeletedIds(List.of("01HQXYZ789"));
        expectedResponse.setDryRun(false);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, Object> policy = new HashMap<>();
        policy.put("max_age_days", 365);

        ForgetResponse response = client.longTermMemory().forgetLongTermMemories(
                policy, null, null, null, 500, false, null);

        // Verify
        assertNotNull(response);
        assertEquals(50, response.getScanned());
        assertEquals(5, response.getDeleted());
        assertFalse(response.isDryRun());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("dry_run=false"));
    }

    @Test
    void testForgetLongTermMemories_WithPinnedIds() throws Exception {
        // Mock response
        ForgetResponse expectedResponse = new ForgetResponse();
        expectedResponse.setScanned(100);
        expectedResponse.setDeleted(8);
        expectedResponse.setDeletedIds(Arrays.asList("01HQXYZ111", "01HQXYZ222"));
        expectedResponse.setDryRun(true);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, Object> policy = new HashMap<>();
        policy.put("budget", 50);

        List<String> pinnedIds = Arrays.asList("01HQXYZ999", "01HQXYZ888");
        ForgetResponse response = client.longTermMemory().forgetLongTermMemories(
                policy, "test-ns", null, null, 1000, true, pinnedIds);

        // Verify
        assertNotNull(response);
        assertEquals(100, response.getScanned());
        assertEquals(8, response.getDeleted());
        assertTrue(response.isDryRun());

        RecordedRequest request = mockServer.takeRequest();
        String requestBody = request.getBody().readUtf8();
        assertTrue(requestBody.contains("pinned_ids"));
        assertTrue(requestBody.contains("01HQXYZ999"));
        assertTrue(requestBody.contains("01HQXYZ888"));
    }

    @Test
    void testForgetLongTermMemories_MinimalParams() throws Exception {
        // Mock response
        ForgetResponse expectedResponse = new ForgetResponse();
        expectedResponse.setScanned(10);
        expectedResponse.setDeleted(2);
        expectedResponse.setDeletedIds(List.of("01HQXYZ001"));
        expectedResponse.setDryRun(true);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute - using convenience method
        Map<String, Object> policy = new HashMap<>();
        policy.put("max_age_days", 90);

        ForgetResponse response = client.longTermMemory().forgetLongTermMemories(policy);

        // Verify
        assertNotNull(response);
        assertEquals(10, response.getScanned());
        assertTrue(response.isDryRun());

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("limit=1000"));
        assertTrue(request.getPath().contains("dry_run=true"));
    }

}
