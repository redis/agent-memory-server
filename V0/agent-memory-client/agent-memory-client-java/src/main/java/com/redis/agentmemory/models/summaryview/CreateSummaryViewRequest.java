package com.redis.agentmemory.models.summaryview;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Map;

/**
 * Request to create a summary view.
 */
public class CreateSummaryViewRequest {

    @Nullable
    private String name;

    @NotNull
    private String source;

    @NotNull
    @JsonProperty("group_by")
    private List<String> groupBy;

    @Nullable
    private Map<String, Object> filters;

    @Nullable
    @JsonProperty("time_window_days")
    private Integer timeWindowDays;

    @Nullable
    private Boolean continuous;

    @Nullable
    private String prompt;

    @Nullable
    @JsonProperty("model_name")
    private String modelName;

    public CreateSummaryViewRequest() {
    }

    public CreateSummaryViewRequest(@NotNull String source, @NotNull List<String> groupBy) {
        this.source = source;
        this.groupBy = groupBy;
    }

    @Nullable
    public String getName() {
        return name;
    }

    public void setName(@Nullable String name) {
        this.name = name;
    }

    @NotNull
    public String getSource() {
        return source;
    }

    public void setSource(@NotNull String source) {
        this.source = source;
    }

    @NotNull
    public List<String> getGroupBy() {
        return groupBy;
    }

    public void setGroupBy(@NotNull List<String> groupBy) {
        this.groupBy = groupBy;
    }

    @Nullable
    public Map<String, Object> getFilters() {
        return filters;
    }

    public void setFilters(@Nullable Map<String, Object> filters) {
        this.filters = filters;
    }

    @Nullable
    public Integer getTimeWindowDays() {
        return timeWindowDays;
    }

    public void setTimeWindowDays(@Nullable Integer timeWindowDays) {
        this.timeWindowDays = timeWindowDays;
    }

    @Nullable
    public Boolean getContinuous() {
        return continuous;
    }

    public void setContinuous(@Nullable Boolean continuous) {
        this.continuous = continuous;
    }

    @Nullable
    public String getPrompt() {
        return prompt;
    }

    public void setPrompt(@Nullable String prompt) {
        this.prompt = prompt;
    }

    @Nullable
    public String getModelName() {
        return modelName;
    }

    public void setModelName(@Nullable String modelName) {
        this.modelName = modelName;
    }
}
