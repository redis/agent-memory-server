package com.redis.agentmemory.models;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.health.HealthCheckResponse;
import com.redis.agentmemory.models.longtermemory.MemoryRecord;
import com.redis.agentmemory.models.longtermemory.MemoryType;
import com.redis.agentmemory.models.workingmemory.MemoryMessage;
import com.redis.agentmemory.models.workingmemory.SessionListResponse;
import com.redis.agentmemory.models.workingmemory.WorkingMemory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonSerializationTest {
    
    private ObjectMapper objectMapper;
    
    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        objectMapper.disable(com.fasterxml.jackson.databind.SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        objectMapper.disable(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    }
    
    @Test
    void testMemoryMessageSerialization() throws Exception {
        MemoryMessage message = new MemoryMessage("user", "Hello, world!");
        
        String json = objectMapper.writeValueAsString(message);
        assertNotNull(json);
        assertTrue(json.contains("\"role\":\"user\""));
        assertTrue(json.contains("\"content\":\"Hello, world!\""));
        
        MemoryMessage deserialized = objectMapper.readValue(json, MemoryMessage.class);
        assertEquals("user", deserialized.getRole());
        assertEquals("Hello, world!", deserialized.getContent());
    }
    
    @Test
    void testMemoryRecordSerialization() throws Exception {
        MemoryRecord record = new MemoryRecord("Test memory");
        record.setUserId("user-123");
        record.setNamespace("test-namespace");
        record.setSessionId("session-456");
        record.setMemoryType(MemoryType.SEMANTIC);
        record.setTopics(Arrays.asList("topic1", "topic2"));
        record.setEntities(Arrays.asList("entity1", "entity2"));
        
        String json = objectMapper.writeValueAsString(record);
        assertNotNull(json);
        assertTrue(json.contains("\"text\":\"Test memory\""));
        assertTrue(json.contains("\"user_id\":\"user-123\""));
        assertTrue(json.contains("\"memory_type\":\"semantic\""));
        
        MemoryRecord deserialized = objectMapper.readValue(json, MemoryRecord.class);
        assertEquals("Test memory", deserialized.getText());
        assertEquals("user-123", deserialized.getUserId());
        assertEquals(MemoryType.SEMANTIC, deserialized.getMemoryType());
        assertNotNull(deserialized.getTopics());
        assertEquals(2, deserialized.getTopics().size());
    }
    
    @Test
    void testMemoryTypeSerialization() throws Exception {
        // Test enum serialization
        assertEquals("\"message\"", objectMapper.writeValueAsString(MemoryType.MESSAGE));
        assertEquals("\"semantic\"", objectMapper.writeValueAsString(MemoryType.SEMANTIC));
        assertEquals("\"episodic\"", objectMapper.writeValueAsString(MemoryType.EPISODIC));
        
        // Test enum deserialization
        assertEquals(MemoryType.MESSAGE, objectMapper.readValue("\"message\"", MemoryType.class));
        assertEquals(MemoryType.SEMANTIC, objectMapper.readValue("\"semantic\"", MemoryType.class));
        assertEquals(MemoryType.EPISODIC, objectMapper.readValue("\"episodic\"", MemoryType.class));
    }
    
    @Test
    void testWorkingMemorySerialization() throws Exception {
        WorkingMemory memory = new WorkingMemory("session-123");
        memory.setUserId("user-456");
        memory.setNamespace("test-namespace");
        memory.setContext("Previous conversation");
        memory.setTokens(1000);
        
        MemoryMessage message = new MemoryMessage("user", "Hello");
        memory.getMessages().add(message);
        
        MemoryRecord record = new MemoryRecord("User said hello");
        memory.getMemories().add(record);
        
        Map<String, Object> data = new HashMap<>();
        data.put("key1", "value1");
        data.put("key2", 42);
        memory.setData(data);
        
        String json = objectMapper.writeValueAsString(memory);
        assertNotNull(json);
        assertTrue(json.contains("\"session_id\":\"session-123\""));
        assertTrue(json.contains("\"user_id\":\"user-456\""));
        
        WorkingMemory deserialized = objectMapper.readValue(json, WorkingMemory.class);
        assertEquals("session-123", deserialized.getSessionId());
        assertEquals("user-456", deserialized.getUserId());
        assertEquals(1, deserialized.getMessages().size());
        assertEquals(1, deserialized.getMemories().size());
        assertNotNull(deserialized.getData());
        assertEquals("value1", deserialized.getData().get("key1"));
    }
    
    @Test
    void testInstantSerialization() throws Exception {
        MemoryMessage message = new MemoryMessage("user", "Test");
        Instant now = Instant.now();
        message.setCreatedAt(now);
        
        String json = objectMapper.writeValueAsString(message);
        assertNotNull(json);
        
        // Should be in ISO-8601 format, not timestamp
        assertFalse(json.contains("\"created_at\":" + now.toEpochMilli()));
        assertTrue(json.contains("\"created_at\":\""));
        
        MemoryMessage deserialized = objectMapper.readValue(json, MemoryMessage.class);
        assertNotNull(deserialized.getCreatedAt());
    }
    
    @Test
    void testHealthCheckResponseDeserialization() throws Exception {
        String json = "{\"now\":1705318200.0}";

        HealthCheckResponse response = objectMapper.readValue(json, HealthCheckResponse.class);
        assertNotNull(response);
        assertTrue(response.getNow() > 0);
    }
    
    @Test
    void testSessionListResponseDeserialization() throws Exception {
        String json = "{\"sessions\":[\"session-1\",\"session-2\"],\"total\":2}";
        
        SessionListResponse response = objectMapper.readValue(json, SessionListResponse.class);
        assertNotNull(response);
        assertEquals(2, response.getTotal());
        assertEquals(2, response.getSessions().size());
        assertTrue(response.getSessions().contains("session-1"));
    }
    
    @Test
    void testAckResponseDeserialization() throws Exception {
        String json = "{\"status\":\"ok\"}";
        
        AckResponse response = objectMapper.readValue(json, AckResponse.class);
        assertNotNull(response);
        assertEquals("ok", response.getStatus());
    }
}

