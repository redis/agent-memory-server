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
    @JsonProperty("search_mode")
    private String searchMode;

    @Nullable
    @JsonProperty("hybrid_alpha")
    private Double hybridAlpha;

    @Nullable
    @JsonProperty("text_scorer")
    private String textScorer;

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

    // Recency boost parameters
    @Nullable
    @JsonProperty("recency_boost")
    private Boolean recencyBoost;

    @Nullable
    @JsonProperty("recency_semantic_weight")
    private Double recencySemanticWeight;

    @Nullable
    @JsonProperty("recency_recency_weight")
    private Double recencyRecencyWeight;

    @Nullable
    @JsonProperty("recency_freshness_weight")
    private Double recencyFreshnessWeight;

    @Nullable
    @JsonProperty("recency_novelty_weight")
    private Double recencyNoveltyWeight;

    @Nullable
    @JsonProperty("recency_half_life_last_access_days")
    private Double recencyHalfLifeLastAccessDays;

    @Nullable
    @JsonProperty("recency_half_life_created_days")
    private Double recencyHalfLifeCreatedDays;

    @Nullable
    @JsonProperty("server_side_recency")
    private Boolean serverSideRecency;

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
    public String getSearchMode() {
        return searchMode;
    }

    public void setSearchMode(@Nullable String searchMode) {
        this.searchMode = searchMode;
    }

    @Nullable
    public Double getHybridAlpha() {
        return hybridAlpha;
    }

    public void setHybridAlpha(@Nullable Double hybridAlpha) {
        this.hybridAlpha = hybridAlpha;
    }

    @Nullable
    public String getTextScorer() {
        return textScorer;
    }

    public void setTextScorer(@Nullable String textScorer) {
        this.textScorer = textScorer;
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

    @Nullable
    public Boolean getRecencyBoost() {
        return recencyBoost;
    }

    public void setRecencyBoost(@Nullable Boolean recencyBoost) {
        this.recencyBoost = recencyBoost;
    }

    @Nullable
    public Double getRecencySemanticWeight() {
        return recencySemanticWeight;
    }

    public void setRecencySemanticWeight(@Nullable Double recencySemanticWeight) {
        this.recencySemanticWeight = recencySemanticWeight;
    }

    @Nullable
    public Double getRecencyRecencyWeight() {
        return recencyRecencyWeight;
    }

    public void setRecencyRecencyWeight(@Nullable Double recencyRecencyWeight) {
        this.recencyRecencyWeight = recencyRecencyWeight;
    }

    @Nullable
    public Double getRecencyFreshnessWeight() {
        return recencyFreshnessWeight;
    }

    public void setRecencyFreshnessWeight(@Nullable Double recencyFreshnessWeight) {
        this.recencyFreshnessWeight = recencyFreshnessWeight;
    }

    @Nullable
    public Double getRecencyNoveltyWeight() {
        return recencyNoveltyWeight;
    }

    public void setRecencyNoveltyWeight(@Nullable Double recencyNoveltyWeight) {
        this.recencyNoveltyWeight = recencyNoveltyWeight;
    }

    @Nullable
    public Double getRecencyHalfLifeLastAccessDays() {
        return recencyHalfLifeLastAccessDays;
    }

    public void setRecencyHalfLifeLastAccessDays(@Nullable Double recencyHalfLifeLastAccessDays) {
        this.recencyHalfLifeLastAccessDays = recencyHalfLifeLastAccessDays;
    }

    @Nullable
    public Double getRecencyHalfLifeCreatedDays() {
        return recencyHalfLifeCreatedDays;
    }

    public void setRecencyHalfLifeCreatedDays(@Nullable Double recencyHalfLifeCreatedDays) {
        this.recencyHalfLifeCreatedDays = recencyHalfLifeCreatedDays;
    }

    @Nullable
    public Boolean getServerSideRecency() {
        return serverSideRecency;
    }

    public void setServerSideRecency(@Nullable Boolean serverSideRecency) {
        this.serverSideRecency = serverSideRecency;
    }

    @Override
    public String toString() {
        return "SearchRequest{" +
                "text='" + text + '\'' +
                ", searchMode='" + searchMode + '\'' +
                ", hybridAlpha=" + hybridAlpha +
                ", textScorer='" + textScorer + '\'' +
                ", sessionId='" + sessionId + '\'' +
                ", namespace='" + namespace + '\'' +
                ", topics=" + topics +
                ", entities=" + entities +
                ", userId='" + userId + '\'' +
                ", distanceThreshold=" + distanceThreshold +
                ", limit=" + limit +
                ", offset=" + offset +
                ", recencyBoost=" + recencyBoost +
                ", recencySemanticWeight=" + recencySemanticWeight +
                ", recencyRecencyWeight=" + recencyRecencyWeight +
                ", recencyFreshnessWeight=" + recencyFreshnessWeight +
                ", recencyNoveltyWeight=" + recencyNoveltyWeight +
                ", recencyHalfLifeLastAccessDays=" + recencyHalfLifeLastAccessDays +
                ", recencyHalfLifeCreatedDays=" + recencyHalfLifeCreatedDays +
                ", serverSideRecency=" + serverSideRecency +
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

        public Builder searchMode(@Nullable String searchMode) {
            request.searchMode = searchMode;
            return this;
        }

        public Builder hybridAlpha(@Nullable Double hybridAlpha) {
            request.hybridAlpha = hybridAlpha;
            return this;
        }

        public Builder textScorer(@Nullable String textScorer) {
            request.textScorer = textScorer;
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

        public Builder recencyBoost(@Nullable Boolean recencyBoost) {
            request.recencyBoost = recencyBoost;
            return this;
        }

        public Builder recencySemanticWeight(@Nullable Double recencySemanticWeight) {
            request.recencySemanticWeight = recencySemanticWeight;
            return this;
        }

        public Builder recencyRecencyWeight(@Nullable Double recencyRecencyWeight) {
            request.recencyRecencyWeight = recencyRecencyWeight;
            return this;
        }

        public Builder recencyFreshnessWeight(@Nullable Double recencyFreshnessWeight) {
            request.recencyFreshnessWeight = recencyFreshnessWeight;
            return this;
        }

        public Builder recencyNoveltyWeight(@Nullable Double recencyNoveltyWeight) {
            request.recencyNoveltyWeight = recencyNoveltyWeight;
            return this;
        }

        public Builder recencyHalfLifeLastAccessDays(@Nullable Double recencyHalfLifeLastAccessDays) {
            request.recencyHalfLifeLastAccessDays = recencyHalfLifeLastAccessDays;
            return this;
        }

        public Builder recencyHalfLifeCreatedDays(@Nullable Double recencyHalfLifeCreatedDays) {
            request.recencyHalfLifeCreatedDays = recencyHalfLifeCreatedDays;
            return this;
        }

        public Builder serverSideRecency(@Nullable Boolean serverSideRecency) {
            request.serverSideRecency = serverSideRecency;
            return this;
        }

        public SearchRequest build() {
            return request;
        }
    }
}
