package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
import com.redis.agentmemory.models.health.HealthCheckResponse;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for HealthService functionality.
 */
class HealthServiceTest {
    
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
    void testHealthCheck() throws Exception {
        // Mock response
        HealthCheckResponse expectedResponse = new HealthCheckResponse();
        expectedResponse.setNow(System.currentTimeMillis() / 1000.0);

        mockServer.enqueue(new MockResponse()
                .setBody(objectMapper.writeValueAsString(expectedResponse))
                .addHeader("Content-Type", "application/json"));

        // Execute
        HealthCheckResponse response = client.health().healthCheck();

        // Verify
        assertNotNull(response);
        assertTrue(response.getNow() > 0);

        RecordedRequest request = mockServer.takeRequest();
        assertEquals("GET", request.getMethod());
        assertNotNull(request.getPath());
        assertTrue(request.getPath().contains("/v1/health"));
    }
}

