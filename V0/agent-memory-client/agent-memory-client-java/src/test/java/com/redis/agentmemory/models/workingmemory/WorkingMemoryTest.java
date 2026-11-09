package com.redis.agentmemory.models.workingmemory;

import com.redis.agentmemory.models.longtermemory.MemoryRecord;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WorkingMemoryTest {

    @Test
    void testDefaultConstructor() {
        WorkingMemory memory = new WorkingMemory();

        assertNotNull(memory.getMessages());
        assertTrue(memory.getMessages().isEmpty());
        assertNotNull(memory.getMemories());
        assertTrue(memory.getMemories().isEmpty());
        assertNotNull(memory.getData());
        assertEquals(0, memory.getTokens());
        assertNotNull(memory.getLongTermMemoryStrategy());
        assertNotNull(memory.getLastAccessed());
    }

    @Test
    void testConstructorWithSessionId() {
        WorkingMemory memory = new WorkingMemory("session-123");

        assertEquals("session-123", memory.getSessionId());
        assertNotNull(memory.getMessages());
        assertNotNull(memory.getMemories());
    }

    @Test
    void testAddingMessages() {
        WorkingMemory memory = new WorkingMemory("session-123");

        MemoryMessage message1 = new MemoryMessage("user", "Hello");
        MemoryMessage message2 = new MemoryMessage("assistant", "Hi there!");

        memory.getMessages().add(message1);
        memory.getMessages().add(message2);

        assertEquals(2, memory.getMessages().size());
        assertEquals("Hello", memory.getMessages().get(0).getContent());
        assertEquals("Hi there!", memory.getMessages().get(1).getContent());
    }

    @Test
    void testAddingMemories() {
        WorkingMemory memory = new WorkingMemory("session-123");

        MemoryRecord record1 = new MemoryRecord("Memory 1");
        MemoryRecord record2 = new MemoryRecord("Memory 2");

        memory.getMemories().add(record1);
        memory.getMemories().add(record2);

        assertEquals(2, memory.getMemories().size());
        assertEquals("Memory 1", memory.getMemories().get(0).getText());
        assertEquals("Memory 2", memory.getMemories().get(1).getText());
    }

    @Test
    void testSettersAndGetters() {
        WorkingMemory memory = new WorkingMemory();

        memory.setSessionId("session-456");
        memory.setUserId("user-789");
        memory.setNamespace("test-namespace");
        memory.setContext("Previous conversation summary");
        memory.setTokens(1000);
        memory.setTtlSeconds(3600);

        assertEquals("session-456", memory.getSessionId());
        assertEquals("user-789", memory.getUserId());
        assertEquals("test-namespace", memory.getNamespace());
        assertEquals("Previous conversation summary", memory.getContext());
        assertEquals(1000, memory.getTokens());
        assertEquals(3600, memory.getTtlSeconds());
    }
}
