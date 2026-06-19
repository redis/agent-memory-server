package com.redis.agentmemory.exceptions;

/**
 * Raised when memory record or filter validation fails.
 * <p>
 * This exception signals validation issues that occur before sending
 * requests to the server, allowing for early error detection.
 */
public class MemoryValidationException extends MemoryClientException {

    public MemoryValidationException(String message) {
        super(message);
    }

    public MemoryValidationException(String message, Throwable cause) {
        super(message, cause);
    }
}
