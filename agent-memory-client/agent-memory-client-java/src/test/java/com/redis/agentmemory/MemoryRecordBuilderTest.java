package com.redis.agentmemory;

import com.redis.agentmemory.models.longtermemory.MemoryRecord;
import com.redis.agentmemory.models.longtermemory.MemoryType;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test MemoryRecord builder pattern and default values
 */
class MemoryRecordBuilderTest {

    @Test
    void testMemoryRecordBuilderDefaults() {
        // Test that builder creates memories with correct defaults for long-term storage
        MemoryRecord memory = MemoryRecord.builder()
                .text("User prefers dark mode")
                .memoryType(MemoryType.SEMANTIC)
                .topics(Arrays.asList("preferences", "ui"))
                .build();

        assertNotNull(memory.getId());
        assertEquals("User prefers dark mode", memory.getText());
        assertEquals(MemoryType.SEMANTIC, memory.getMemoryType());
        assertEquals("t", memory.getDiscreteMemoryExtracted()); // Should be "t" for extracted memories
        assertNotNull(memory.getCreatedAt());
        assertNotNull(memory.getLastAccessed());
        assertNotNull(memory.getUpdatedAt());
        assertEquals(Arrays.asList("preferences", "ui"), memory.getTopics());
    }

    @Test
    void testMemoryRecordBuilderWithEpisodicType() {
        MemoryRecord memory = MemoryRecord.builder()
                .text("User completed onboarding on 2024-01-15")
                .memoryType(MemoryType.EPISODIC)
                .topics(Arrays.asList("onboarding", "milestones"))
                .build();

        assertEquals(MemoryType.EPISODIC, memory.getMemoryType());
        assertEquals("t", memory.getDiscreteMemoryExtracted());
    }

    @Test
    void testMemoryRecordBuilderRequiresText() {
        // Test that builder throws exception when text is not provided
        assertThrows(IllegalStateException.class, () -> MemoryRecord.builder()
                .memoryType(MemoryType.SEMANTIC)
                .build());
    }

    @Test
    void testMemoryRecordDefaultConstructor() {
        // Test that default constructor still uses old defaults (for deserialization)
        MemoryRecord memory = new MemoryRecord();
        
        assertNotNull(memory.getId());
        assertEquals("f", memory.getDiscreteMemoryExtracted()); // Should be "f" for default constructor
        assertEquals(MemoryType.MESSAGE, memory.getMemoryType()); // Should be MESSAGE for default constructor
    }

    @Test
    void testMemoryRecordConstructorWithText() {
        // Test that text constructor uses old defaults (for deserialization)
        MemoryRecord memory = new MemoryRecord("Test text");
        
        assertEquals("Test text", memory.getText());
        assertEquals("f", memory.getDiscreteMemoryExtracted());
        assertEquals(MemoryType.MESSAGE, memory.getMemoryType());
    }
}

