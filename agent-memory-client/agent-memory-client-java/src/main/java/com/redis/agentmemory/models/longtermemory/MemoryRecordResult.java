package com.redis.agentmemory.models.longtermemory;

/**
 * Result from a memory search operation.
 */
public class MemoryRecordResult extends MemoryRecord {
    
    private double dist;
    
    public MemoryRecordResult() {
        super();
    }
    
    public double getDist() {
        return dist;
    }
    
    public void setDist(double dist) {
        this.dist = dist;
    }

    @Override
    public String toString() {
        return "MemoryRecordResult{" +
                "dist=" + dist +
                ", " + super.toString() +
                '}';
    }
}

