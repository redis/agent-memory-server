package com.redis.agentmemory.models.workingmemory;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.Nullable;

/**
 * Response from working memory operations.
 */
public class WorkingMemoryResponse extends WorkingMemory {
    
    @Nullable
    @JsonProperty("context_percentage_total_used")
    private Double contextPercentageTotalUsed;
    
    @Nullable
    @JsonProperty("context_percentage_until_summarization")
    private Double contextPercentageUntilSummarization;
    
    @Nullable
    @JsonProperty("new_session")
    private Boolean newSession;
    
    @Nullable
    private Boolean unsaved;
    
    public WorkingMemoryResponse() {
        super();
    }
    
    @Nullable
    public Double getContextPercentageTotalUsed() {
        return contextPercentageTotalUsed;
    }
    
    public void setContextPercentageTotalUsed(@Nullable Double contextPercentageTotalUsed) {
        this.contextPercentageTotalUsed = contextPercentageTotalUsed;
    }
    
    @Nullable
    public Double getContextPercentageUntilSummarization() {
        return contextPercentageUntilSummarization;
    }
    
    public void setContextPercentageUntilSummarization(@Nullable Double contextPercentageUntilSummarization) {
        this.contextPercentageUntilSummarization = contextPercentageUntilSummarization;
    }
    
    @Nullable
    public Boolean getNewSession() {
        return newSession;
    }
    
    public void setNewSession(@Nullable Boolean newSession) {
        this.newSession = newSession;
    }
    
    @Nullable
    public Boolean getUnsaved() {
        return unsaved;
    }
    
    public void setUnsaved(@Nullable Boolean unsaved) {
        this.unsaved = unsaved;
    }

    @Override
    public String toString() {
        return "WorkingMemoryResponse{" +
                "contextPercentageTotalUsed=" + contextPercentageTotalUsed +
                ", contextPercentageUntilSummarization=" + contextPercentageUntilSummarization +
                ", newSession=" + newSession +
                ", unsaved=" + unsaved +
                ", " + super.toString() +
                '}';
    }
}

