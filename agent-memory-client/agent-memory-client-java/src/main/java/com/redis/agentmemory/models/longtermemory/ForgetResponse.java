package com.redis.agentmemory.models.longtermemory;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;

import java.util.List;

/**
 * Response from the "forget" endpoint.
 */
public class ForgetResponse {

    private int scanned;

    private int deleted;

    @NotNull
    @JsonProperty("deleted_ids")
    private List<String> deletedIds;

    @JsonProperty("dry_run")
    private boolean dryRun;

    public ForgetResponse() {
    }

    public ForgetResponse(int scanned, int deleted, @NotNull List<String> deletedIds, boolean dryRun) {
        this.scanned = scanned;
        this.deleted = deleted;
        this.deletedIds = deletedIds;
        this.dryRun = dryRun;
    }

    public int getScanned() {
        return scanned;
    }

    public void setScanned(int scanned) {
        this.scanned = scanned;
    }

    public int getDeleted() {
        return deleted;
    }

    public void setDeleted(int deleted) {
        this.deleted = deleted;
    }

    @NotNull
    public List<String> getDeletedIds() {
        return deletedIds;
    }

    public void setDeletedIds(@NotNull List<String> deletedIds) {
        this.deletedIds = deletedIds;
    }

    public boolean isDryRun() {
        return dryRun;
    }

    public void setDryRun(boolean dryRun) {
        this.dryRun = dryRun;
    }

    @Override
    public String toString() {
        return "ForgetResponse{" +
                "scanned=" + scanned +
                ", deleted=" + deleted +
                ", deletedIds=" + deletedIds +
                ", dryRun=" + dryRun +
                '}';
    }
}
