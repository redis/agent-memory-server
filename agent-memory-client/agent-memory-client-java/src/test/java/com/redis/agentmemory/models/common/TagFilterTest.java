package com.redis.agentmemory.models.common;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TagFilterTest {

    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
    }

    @Test
    void eq_serializesToEqOperator() throws Exception {
        String json = objectMapper.writeValueAsString(TagFilter.eq("user-123"));
        assertEquals("{\"eq\":\"user-123\"}", json);
    }

    @Test
    void ne_serializesToNeOperator() throws Exception {
        String json = objectMapper.writeValueAsString(TagFilter.ne("user-123"));
        assertEquals("{\"ne\":\"user-123\"}", json);
    }

    @Test
    void any_varargs_serializesToAnyOperator() throws Exception {
        String json = objectMapper.writeValueAsString(TagFilter.any("user-123", "__account__"));
        assertEquals("{\"any\":[\"user-123\",\"__account__\"]}", json);
    }

    @Test
    void any_list_serializesToAnyOperator() throws Exception {
        String json = objectMapper.writeValueAsString(TagFilter.any(List.of("a", "b", "c")));
        assertEquals("{\"any\":[\"a\",\"b\",\"c\"]}", json);
    }

    @Test
    void all_serializesToAllOperator() throws Exception {
        String json = objectMapper.writeValueAsString(TagFilter.all("x", "y"));
        assertEquals("{\"all\":[\"x\",\"y\"]}", json);
    }

    @Test
    void startsWith_serializesToStartswithOperator() throws Exception {
        String json = objectMapper.writeValueAsString(TagFilter.startsWith("tenant-"));
        assertEquals("{\"startswith\":\"tenant-\"}", json);
    }

    @Test
    void any_emptyVarargs_throws() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> TagFilter.any());
        assertEquals("any cannot be an empty list", ex.getMessage());
    }

    @Test
    void any_emptyList_throws() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> TagFilter.any(List.of()));
        assertEquals("any cannot be an empty list", ex.getMessage());
    }

    @Test
    void all_emptyVarargs_throws() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> TagFilter.all());
        assertEquals("all cannot be an empty list", ex.getMessage());
    }

    @Test
    void all_emptyList_throws() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> TagFilter.all(List.of()));
        assertEquals("all cannot be an empty list", ex.getMessage());
    }

    @Test
    void startsWith_emptyString_throws() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> TagFilter.startsWith(""));
        assertEquals("startswith cannot be an empty string", ex.getMessage());
    }
}
