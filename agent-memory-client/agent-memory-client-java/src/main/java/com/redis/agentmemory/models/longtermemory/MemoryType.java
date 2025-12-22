package com.redis.agentmemory.models.longtermemory;

import com.fasterxml.jackson.annotation.JsonValue;

/**
 * Enum for memory types.
 */
public enum MemoryType {
    EPISODIC("episodic"),
    SEMANTIC("semantic"),
    MESSAGE("message");
    
    private final String value;
    
    MemoryType(String value) {
        this.value = value;
    }
    
    @JsonValue
    public String getValue() {
        return value;
    }
    
    public static MemoryType fromValue(String value) {
        for (MemoryType type : values()) {
            if (type.value.equals(value)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown memory type: " + value);
    }
}

