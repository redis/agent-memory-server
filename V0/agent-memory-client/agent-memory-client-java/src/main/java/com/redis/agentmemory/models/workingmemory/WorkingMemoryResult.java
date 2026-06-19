package com.redis.agentmemory.models.workingmemory;

/**
 * Result of getOrCreateWorkingMemory operation.
 * Contains a flag indicating if the memory was created and the memory itself.
 */
public class WorkingMemoryResult {
    private final boolean created;
    private final WorkingMemoryResponse memory;

    public WorkingMemoryResult(boolean created, WorkingMemoryResponse memory) {
        this.created = created;
        this.memory = memory;
    }

    /**
     * @return true if the memory was created, false if it already existed
     */
    public boolean isCreated() {
        return created;
    }

    /**
     * @return the working memory (either newly created or existing)
     */
    public WorkingMemoryResponse getMemory() {
        return memory;
    }

    @Override
    public String toString() {
        return "WorkingMemoryResult{" +
                "created=" + created +
                ", memory=" + memory +
                '}';
    }
}
