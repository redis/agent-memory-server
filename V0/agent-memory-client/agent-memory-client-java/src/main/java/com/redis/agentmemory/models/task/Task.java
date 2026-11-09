package com.redis.agentmemory.models.task;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Background task representation.
 */
public class Task {

    @NotNull
    private String id;

    @NotNull
    private String type;

    @NotNull
    private String status;

    @Nullable
    @JsonProperty("view_id")
    private String viewId;

    @Nullable
    @JsonProperty("created_at")
    private String createdAt;

    @Nullable
    @JsonProperty("started_at")
    private String startedAt;

    @Nullable
    @JsonProperty("completed_at")
    private String completedAt;

    @Nullable
    @JsonProperty("error_message")
    private String errorMessage;

    public Task() {
    }

    public Task(@NotNull String id, @NotNull String type, @NotNull String status) {
        this.id = id;
        this.type = type;
        this.status = status;
    }

    @NotNull
    public String getId() {
        return id;
    }

    public void setId(@NotNull String id) {
        this.id = id;
    }

    @NotNull
    public String getType() {
        return type;
    }

    public void setType(@NotNull String type) {
        this.type = type;
    }

    @NotNull
    public String getStatus() {
        return status;
    }

    public void setStatus(@NotNull String status) {
        this.status = status;
    }

    @Nullable
    public String getViewId() {
        return viewId;
    }

    public void setViewId(@Nullable String viewId) {
        this.viewId = viewId;
    }

    @Nullable
    public String getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(@Nullable String createdAt) {
        this.createdAt = createdAt;
    }

    @Nullable
    public String getStartedAt() {
        return startedAt;
    }

    public void setStartedAt(@Nullable String startedAt) {
        this.startedAt = startedAt;
    }

    @Nullable
    public String getCompletedAt() {
        return completedAt;
    }

    public void setCompletedAt(@Nullable String completedAt) {
        this.completedAt = completedAt;
    }

    @Nullable
    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(@Nullable String errorMessage) {
        this.errorMessage = errorMessage;
    }

    @Override
    public String toString() {
        return "Task{" +
                "id='" + id + '\'' +
                ", type='" + type + '\'' +
                ", status='" + status + '\'' +
                ", viewId='" + viewId + '\'' +
                '}';
    }
}
