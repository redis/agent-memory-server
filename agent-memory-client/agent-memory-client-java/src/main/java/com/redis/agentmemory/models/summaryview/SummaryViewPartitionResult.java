package com.redis.agentmemory.models.summaryview;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Map;

/**
 * Result of summarizing one partition.
 */
public class SummaryViewPartitionResult {

    @NotNull
    @JsonProperty("view_id")
    private String viewId;

    @NotNull
    private Map<String, String> group;

    @NotNull
    private String summary;

    @JsonProperty("memory_count")
    private int memoryCount;

    @Nullable
    @JsonProperty("computed_at")
    private String computedAt;

    public SummaryViewPartitionResult() {
    }

    public SummaryViewPartitionResult(
            @NotNull String viewId,
            @NotNull Map<String, String> group,
            @NotNull String summary,
            int memoryCount) {
        this.viewId = viewId;
        this.group = group;
        this.summary = summary;
        this.memoryCount = memoryCount;
    }

    @NotNull
    public String getViewId() {
        return viewId;
    }

    public void setViewId(@NotNull String viewId) {
        this.viewId = viewId;
    }

    @NotNull
    public Map<String, String> getGroup() {
        return group;
    }

    public void setGroup(@NotNull Map<String, String> group) {
        this.group = group;
    }

    @NotNull
    public String getSummary() {
        return summary;
    }

    public void setSummary(@NotNull String summary) {
        this.summary = summary;
    }

    public int getMemoryCount() {
        return memoryCount;
    }

    public void setMemoryCount(int memoryCount) {
        this.memoryCount = memoryCount;
    }

    @Nullable
    public String getComputedAt() {
        return computedAt;
    }

    public void setComputedAt(@Nullable String computedAt) {
        this.computedAt = computedAt;
    }

    @Override
    public String toString() {
        return "SummaryViewPartitionResult{" +
                "viewId='" + viewId + '\'' +
                ", group=" + group +
                ", memoryCount=" + memoryCount +
                '}';
    }
}
