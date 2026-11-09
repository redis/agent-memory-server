package com.redis.agentmemory.models.workingmemory;

/**
 * Strategy for merging data when updating working memory.
 */
public enum MergeStrategy {
    /**
     * Replace existing data entirely with new data.
     */
    REPLACE,

    /**
     * Shallow merge - top-level keys from new data override existing keys.
     */
    MERGE,

    /**
     * Deep merge - recursively merge nested maps.
     */
    DEEP_MERGE
}
