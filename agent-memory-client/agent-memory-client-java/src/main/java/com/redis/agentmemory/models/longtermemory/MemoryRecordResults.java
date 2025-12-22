package com.redis.agentmemory.models.longtermemory;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;

/**
 * Results from memory search operations.
 */
public class MemoryRecordResults {
    
    @NotNull
    private List<MemoryRecordResult> memories;
    
    private int total;
    
    @Nullable
    @JsonProperty("next_offset")
    private Integer nextOffset;
    
    public MemoryRecordResults() {
    }
    
    public MemoryRecordResults(@NotNull List<MemoryRecordResult> memories, int total) {
        this.memories = memories;
        this.total = total;
    }
    
    @NotNull
    public List<MemoryRecordResult> getMemories() {
        return memories;
    }
    
    public void setMemories(@NotNull List<MemoryRecordResult> memories) {
        this.memories = memories;
    }
    
    public int getTotal() {
        return total;
    }
    
    public void setTotal(int total) {
        this.total = total;
    }
    
    @Nullable
    public Integer getNextOffset() {
        return nextOffset;
    }
    
    public void setNextOffset(@Nullable Integer nextOffset) {
        this.nextOffset = nextOffset;
    }

    @Override
    public String toString() {
        return "MemoryRecordResults{" +
                "memories=" + memories +
                ", total=" + total +
                ", nextOffset=" + nextOffset +
                '}';
    }
}

