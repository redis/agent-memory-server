package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
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
 * Tests for MemoryHydrationService functionality.
 */
class MemoryHydrationServiceTest {
    
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
    void testMemoryPrompt() throws Exception {
        // Mock response
        Map<String, Object> expectedResponse = new HashMap<>();
        expectedResponse.put("messages", List.of(
                Map.of("role", "system", "content", "Context from memory")
        ));

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        Map<String, Object> longTermSearch = new HashMap<>();
        longTermSearch.put("limit", 10);

        Map<String, Object> response = client.hydration().memoryPrompt(
                "What are my preferences?",
                "session-123",
                "test-ns",
                "gpt-4",
                8000,
                longTermSearch,
                "user-123",
                true
        );

        // Verify
        assertNotNull(response);
        assertTrue(response.containsKey("messages"));

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("POST", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/memory/prompt"));
        assertTrue(request.getPath().contains("optimize_query=true"));
    }
}

