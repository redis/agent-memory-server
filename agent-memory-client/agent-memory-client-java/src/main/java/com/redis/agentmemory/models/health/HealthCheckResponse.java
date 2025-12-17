package com.redis.agentmemory.models.health;

/**
 * Health check response from the server.
 */
public class HealthCheckResponse {
    
    private double now;
    
    public HealthCheckResponse() {
    }
    
    public HealthCheckResponse(double now) {
        this.now = now;
    }
    
    public double getNow() {
        return now;
    }
    
    public void setNow(double now) {
        this.now = now;
    }

    @Override
    public String toString() {
        return "HealthCheckResponse{" +
                "now=" + now +
                '}';
    }
}

