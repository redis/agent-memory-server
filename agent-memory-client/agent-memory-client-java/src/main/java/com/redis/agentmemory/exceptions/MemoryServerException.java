package com.redis.agentmemory.exceptions;

import org.jetbrains.annotations.Nullable;

/**
 * Raised when the memory server returns an error.
 */
public class MemoryServerException extends MemoryClientException {

    @Nullable
    private final Integer statusCode;

    public MemoryServerException(String message) {
        this(message, null);
    }

    public MemoryServerException(String message, @Nullable Integer statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public MemoryServerException(String message, @Nullable Integer statusCode, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }

    @Nullable
    public Integer getStatusCode() {
        return statusCode;
    }
}
