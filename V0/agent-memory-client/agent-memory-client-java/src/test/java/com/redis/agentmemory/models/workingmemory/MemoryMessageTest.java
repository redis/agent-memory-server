package com.redis.agentmemory.models.workingmemory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MemoryMessageTest {

    @Test
    void testDefaultConstructor() {
        MemoryMessage message = new MemoryMessage();

        assertNotNull(message.getId());
        assertNotNull(message.getCreatedAt());
        assertEquals("f", message.getDiscreteMemoryExtracted());
        assertNull(message.getPersistedAt());
    }

    @Test
    void testConstructorWithRoleAndContent() {
        MemoryMessage message = new MemoryMessage("user", "Hello, world!");

        assertEquals("user", message.getRole());
        assertEquals("Hello, world!", message.getContent());
        assertNotNull(message.getId());
        assertNotNull(message.getCreatedAt());
        assertEquals("f", message.getDiscreteMemoryExtracted());
    }

    @Test
    void testSettersAndGetters() {
        MemoryMessage message = new MemoryMessage();

        message.setRole("assistant");
        message.setContent("Test content");

        assertEquals("assistant", message.getRole());
        assertEquals("Test content", message.getContent());
    }
}
