package com.redis.agentmemory.models.longtermemory;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MemoryRecordTest {

    @Test
    void testDefaultConstructor() {
        MemoryRecord record = new MemoryRecord();

        assertNotNull(record.getId());
        assertNotNull(record.getCreatedAt());
        assertNotNull(record.getLastAccessed());
        assertNotNull(record.getUpdatedAt());
        assertEquals("f", record.getDiscreteMemoryExtracted());
        assertEquals(MemoryType.MESSAGE, record.getMemoryType());
    }

    @Test
    void testConstructorWithText() {
        MemoryRecord record = new MemoryRecord("Test memory");

        assertEquals("Test memory", record.getText());
        assertNotNull(record.getId());
    }

    @Test
    void testSettersAndGetters() {
        MemoryRecord record = new MemoryRecord();

        record.setText("Test memory");
        record.setSessionId("session-123");
        record.setUserId("user-456");
        record.setNamespace("test-namespace");

        List<String> topics = Arrays.asList("topic1", "topic2");
        record.setTopics(topics);

        List<String> entities = Arrays.asList("entity1", "entity2");
        record.setEntities(entities);

        record.setMemoryType(MemoryType.SEMANTIC);

        assertEquals("Test memory", record.getText());
        assertEquals("session-123", record.getSessionId());
        assertEquals("user-456", record.getUserId());
        assertEquals("test-namespace", record.getNamespace());
        assertEquals(topics, record.getTopics());
        assertEquals(entities, record.getEntities());
        assertEquals(MemoryType.SEMANTIC, record.getMemoryType());
    }
}
