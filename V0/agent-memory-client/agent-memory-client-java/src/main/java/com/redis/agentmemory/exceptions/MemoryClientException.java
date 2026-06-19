package com.redis.agentmemory.exceptions;

/**
 * Base exception for all memory client errors.
 */
public class MemoryClientException extends Exception {

    public MemoryClientException(String message) {
        super(message);
    }

    public MemoryClientException(String message, Throwable cause) {
        super(message, cause);
    }
}
