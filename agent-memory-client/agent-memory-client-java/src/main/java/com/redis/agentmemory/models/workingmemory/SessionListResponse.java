package com.redis.agentmemory.models.workingmemory;

import org.jetbrains.annotations.NotNull;

import java.util.List;

/**
 * Response containing a list of sessions.
 */
public class SessionListResponse {
    
    @NotNull
    private List<String> sessions;
    
    private int total;
    
    public SessionListResponse() {
    }
    
    public SessionListResponse(@NotNull List<String> sessions, int total) {
        this.sessions = sessions;
        this.total = total;
    }
    
    @NotNull
    public List<String> getSessions() {
        return sessions;
    }
    
    public void setSessions(@NotNull List<String> sessions) {
        this.sessions = sessions;
    }
    
    public int getTotal() {
        return total;
    }
    
    public void setTotal(int total) {
        this.total = total;
    }

    @Override
    public String toString() {
        return "SessionListResponse{" +
                "sessions=" + sessions +
                ", total=" + total +
                '}';
    }
}

