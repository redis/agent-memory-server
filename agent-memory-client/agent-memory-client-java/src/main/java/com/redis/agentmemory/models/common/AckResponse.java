package com.redis.agentmemory.models.common;

import org.jetbrains.annotations.NotNull;

/**
 * Generic acknowledgement response.
 */
public class AckResponse {
    
    @NotNull
    private String status;
    
    public AckResponse() {
    }
    
    public AckResponse(@NotNull String status) {
        this.status = status;
    }
    
    @NotNull
    public String getStatus() {
        return status;
    }
    
    public void setStatus(@NotNull String status) {
        this.status = status;
    }

    @Override
    public String toString() {
        return "AckResponse{" +
                "status='" + status + '\'' +
                '}';
    }
}

