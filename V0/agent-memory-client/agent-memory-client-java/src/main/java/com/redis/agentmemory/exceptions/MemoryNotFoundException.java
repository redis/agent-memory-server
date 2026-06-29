package com.redis.agentmemory.exceptions;

/**
 * Raised when a requested memory or session is not found.
 */
public class MemoryNotFoundException extends MemoryClientException {

    public MemoryNotFoundException(String message) {
        super(message);
    }

    public MemoryNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}
