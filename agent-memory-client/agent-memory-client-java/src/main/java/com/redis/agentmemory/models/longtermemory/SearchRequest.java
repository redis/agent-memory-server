package com.redis.agentmemory.models.longtermemory;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.Nullable;

import java.util.List;

/**
 * Request payload for long-term memory search operations.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SearchRequest {

    @Nullable
    private String text;

    @Nullable
    @JsonProperty("session_id")
    private String sessionId;

    @Nullable
    private String namespace;

    @Nullable
    private List<String> topics;

    @Nullable
    private List<String> entities;

    @Nullable
    @JsonProperty("user_id")
    private String userId;

    @Nullable
    @JsonProperty("distance_threshold")
    private Double distanceThreshold;

    private int limit = 10;

    private int offset = 0;

    public SearchRequest() {
    }

    @Nullable
    public String getText() {
        return text;
    }

    public void setText(@Nullable String text) {
        this.text = text;
    }

    @Nullable
    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(@Nullable String sessionId) {
        this.sessionId = sessionId;
    }

    @Nullable
    public String getNamespace() {
        return namespace;
    }

    public void setNamespace(@Nullable String namespace) {
        this.namespace = namespace;
    }

    @Nullable
    public List<String> getTopics() {
        return topics;
    }

    public void setTopics(@Nullable List<String> topics) {
        this.topics = topics;
    }

    @Nullable
    public List<String> getEntities() {
        return entities;
    }

    public void setEntities(@Nullable List<String> entities) {
        this.entities = entities;
    }

    @Nullable
    public String getUserId() {
        return userId;
    }

    public void setUserId(@Nullable String userId) {
        this.userId = userId;
    }

    @Nullable
    public Double getDistanceThreshold() {
        return distanceThreshold;
    }

    public void setDistanceThreshold(@Nullable Double distanceThreshold) {
        this.distanceThreshold = distanceThreshold;
    }

    public int getLimit() {
        return limit;
    }

    public void setLimit(int limit) {
        this.limit = limit;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }

    @Override
    public String toString() {
        return "SearchRequest{" +
                "text='" + text + '\'' +
                ", sessionId='" + sessionId + '\'' +
                ", namespace='" + namespace + '\'' +
                ", topics=" + topics +
                ", entities=" + entities +
                ", userId='" + userId + '\'' +
                ", distanceThreshold=" + distanceThreshold +
                ", limit=" + limit +
                ", offset=" + offset +
                '}';
    }

    /**
     * Creates a new builder for SearchRequest.
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for SearchRequest.
     */
    public static class Builder {
        private final SearchRequest request = new SearchRequest();

        public Builder text(@Nullable String text) {
            request.text = text;
            return this;
        }

        public Builder sessionId(@Nullable String sessionId) {
            request.sessionId = sessionId;
            return this;
        }

        public Builder namespace(@Nullable String namespace) {
            request.namespace = namespace;
            return this;
        }

        public Builder topics(@Nullable List<String> topics) {
            request.topics = topics;
            return this;
        }

        public Builder entities(@Nullable List<String> entities) {
            request.entities = entities;
            return this;
        }

        public Builder userId(@Nullable String userId) {
            request.userId = userId;
            return this;
        }

        public Builder distanceThreshold(@Nullable Double distanceThreshold) {
            request.distanceThreshold = distanceThreshold;
            return this;
        }

        public Builder limit(int limit) {
            request.limit = limit;
            return this;
        }

        public Builder offset(int offset) {
            request.offset = offset;
            return this;
        }

        public SearchRequest build() {
            return request;
        }
    }
}
