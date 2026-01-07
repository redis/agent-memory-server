package com.redis.agentmemory.models.longtermemory;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.f4b6a3.ulid.UlidCreator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.time.Instant;
import java.util.List;

/**
 * A memory record in the system.
 */
public class MemoryRecord {

    @NotNull
    private String id;

    @NotNull
    private String text;

    @Nullable
    @JsonProperty("session_id")
    private String sessionId;

    @Nullable
    @JsonProperty("user_id")
    private String userId;

    @Nullable
    private String namespace;

    @NotNull
    @JsonProperty("last_accessed")
    private Instant lastAccessed;

    @NotNull
    @JsonProperty("created_at")
    private Instant createdAt;

    @NotNull
    @JsonProperty("updated_at")
    private Instant updatedAt;

    @Nullable
    private List<String> topics;

    @Nullable
    private List<String> entities;

    @Nullable
    @JsonProperty("memory_hash")
    private String memoryHash;

    @NotNull
    @JsonProperty("discrete_memory_extracted")
    private String discreteMemoryExtracted;

    @NotNull
    @JsonProperty("memory_type")
    private MemoryType memoryType;

    @Nullable
    @JsonProperty("persisted_at")
    private Instant persistedAt;

    @Nullable
    @JsonProperty("extracted_from")
    private List<String> extractedFrom;

    @Nullable
    @JsonProperty("event_date")
    private Instant eventDate;

    public MemoryRecord() {
        this.id = UlidCreator.getUlid().toString();
        Instant now = Instant.now();
        this.lastAccessed = now;
        this.createdAt = now;
        this.updatedAt = now;
        this.discreteMemoryExtracted = "f";
        this.memoryType = MemoryType.MESSAGE;
    }

    public MemoryRecord(@NotNull String text) {
        this();
        this.text = text;
    }

    // Getters and setters

    @NotNull
    public String getId() {
        return id;
    }

    public void setId(@NotNull String id) {
        this.id = id;
    }

    @NotNull
    public String getText() {
        return text;
    }

    public void setText(@NotNull String text) {
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
    public String getUserId() {
        return userId;
    }

    public void setUserId(@Nullable String userId) {
        this.userId = userId;
    }

    @Nullable
    public String getNamespace() {
        return namespace;
    }

    public void setNamespace(@Nullable String namespace) {
        this.namespace = namespace;
    }

    @NotNull
    public Instant getLastAccessed() {
        return lastAccessed;
    }

    public void setLastAccessed(@NotNull Instant lastAccessed) {
        this.lastAccessed = lastAccessed;
    }

    @NotNull
    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(@NotNull Instant createdAt) {
        this.createdAt = createdAt;
    }

    @NotNull
    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(@NotNull Instant updatedAt) {
        this.updatedAt = updatedAt;
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
    public String getMemoryHash() {
        return memoryHash;
    }

    public void setMemoryHash(@Nullable String memoryHash) {
        this.memoryHash = memoryHash;
    }

    @NotNull
    public String getDiscreteMemoryExtracted() {
        return discreteMemoryExtracted;
    }

    public void setDiscreteMemoryExtracted(@NotNull String discreteMemoryExtracted) {
        this.discreteMemoryExtracted = discreteMemoryExtracted;
    }

    @NotNull
    public MemoryType getMemoryType() {
        return memoryType;
    }

    public void setMemoryType(@NotNull MemoryType memoryType) {
        this.memoryType = memoryType;
    }

    @Nullable
    public Instant getPersistedAt() {
        return persistedAt;
    }

    public void setPersistedAt(@Nullable Instant persistedAt) {
        this.persistedAt = persistedAt;
    }

    @Nullable
    public List<String> getExtractedFrom() {
        return extractedFrom;
    }

    public void setExtractedFrom(@Nullable List<String> extractedFrom) {
        this.extractedFrom = extractedFrom;
    }

    @Nullable
    public Instant getEventDate() {
        return eventDate;
    }

    public void setEventDate(@Nullable Instant eventDate) {
        this.eventDate = eventDate;
    }

    @Override
    public String toString() {
        return "MemoryRecord{" +
                "id='" + id + '\'' +
                ", text='" + text + '\'' +
                ", sessionId='" + sessionId + '\'' +
                ", userId='" + userId + '\'' +
                ", namespace='" + namespace + '\'' +
                ", lastAccessed=" + lastAccessed +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                ", topics=" + topics +
                ", entities=" + entities +
                ", memoryHash='" + memoryHash + '\'' +
                ", discreteMemoryExtracted='" + discreteMemoryExtracted + '\'' +
                ", memoryType=" + memoryType +
                ", extractedFrom=" + extractedFrom +
                ", eventDate=" + eventDate +
                '}';
    }

    /**
     * Creates a new builder for MemoryRecord.
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for MemoryRecord.
     */
    public static class Builder {
        private String id;
        private String text;
        private String sessionId;
        private String userId;
        private String namespace;
        private Instant lastAccessed;
        private Instant createdAt;
        private Instant updatedAt;
        private List<String> topics;
        private List<String> entities;
        private String memoryHash;
        private String discreteMemoryExtracted;
        private MemoryType memoryType;
        private Instant persistedAt;
        private List<String> extractedFrom;
        private Instant eventDate;

        private Builder() {
            // Initialize with defaults for extracted memories (client-created long-term memories)
            this.id = UlidCreator.getUlid().toString();
            Instant now = Instant.now();
            this.lastAccessed = now;
            this.createdAt = now;
            this.updatedAt = now;
            this.discreteMemoryExtracted = "t";  // "t" for extracted memories
            this.memoryType = MemoryType.SEMANTIC;  // SEMANTIC for long-term memories
        }

        /**
         * Initialize builder from an existing MemoryRecord.
         * @param record the record to copy from
         * @return this builder
         */
        public Builder from(MemoryRecord record) {
            this.id = record.id;
            this.text = record.text;
            this.sessionId = record.sessionId;
            this.userId = record.userId;
            this.namespace = record.namespace;
            this.lastAccessed = record.lastAccessed;
            this.createdAt = record.createdAt;
            this.updatedAt = record.updatedAt;
            this.topics = record.topics;
            this.entities = record.entities;
            this.memoryHash = record.memoryHash;
            this.discreteMemoryExtracted = record.discreteMemoryExtracted;
            this.memoryType = record.memoryType;
            this.persistedAt = record.persistedAt;
            this.extractedFrom = record.extractedFrom;
            this.eventDate = record.eventDate;
            return this;
        }

        /**
         * Sets the text content of the memory.
         * @param text the memory text
         * @return this builder
         */
        public Builder text(@NotNull String text) {
            this.text = text;
            return this;
        }

        /**
         * Sets the ID of the memory. If not set, a ULID will be generated.
         * @param id the memory ID
         * @return this builder
         */
        public Builder id(@NotNull String id) {
            this.id = id;
            return this;
        }

        /**
         * Sets the session ID.
         * @param sessionId the session ID
         * @return this builder
         */
        public Builder sessionId(@Nullable String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        /**
         * Sets the user ID.
         * @param userId the user ID
         * @return this builder
         */
        public Builder userId(@Nullable String userId) {
            this.userId = userId;
            return this;
        }

        /**
         * Sets the namespace.
         * @param namespace the namespace
         * @return this builder
         */
        public Builder namespace(@Nullable String namespace) {
            this.namespace = namespace;
            return this;
        }

        /**
         * Sets the last accessed timestamp.
         * @param lastAccessed the last accessed timestamp
         * @return this builder
         */
        public Builder lastAccessed(@NotNull Instant lastAccessed) {
            this.lastAccessed = lastAccessed;
            return this;
        }

        /**
         * Sets the creation timestamp.
         * @param createdAt the creation timestamp
         * @return this builder
         */
        public Builder createdAt(@NotNull Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        /**
         * Sets the update timestamp.
         * @param updatedAt the update timestamp
         * @return this builder
         */
        public Builder updatedAt(@NotNull Instant updatedAt) {
            this.updatedAt = updatedAt;
            return this;
        }

        /**
         * Sets the topics associated with this memory.
         * @param topics the list of topics
         * @return this builder
         */
        public Builder topics(@Nullable List<String> topics) {
            this.topics = topics;
            return this;
        }

        /**
         * Sets the entities associated with this memory.
         * @param entities the list of entities
         * @return this builder
         */
        public Builder entities(@Nullable List<String> entities) {
            this.entities = entities;
            return this;
        }

        /**
         * Sets the memory hash.
         * @param memoryHash the memory hash
         * @return this builder
         */
        public Builder memoryHash(@Nullable String memoryHash) {
            this.memoryHash = memoryHash;
            return this;
        }

        /**
         * Sets whether discrete memory has been extracted.
         * @param discreteMemoryExtracted "t" for true, "f" for false
         * @return this builder
         */
        public Builder discreteMemoryExtracted(@NotNull String discreteMemoryExtracted) {
            this.discreteMemoryExtracted = discreteMemoryExtracted;
            return this;
        }

        /**
         * Sets the memory type.
         * @param memoryType the memory type
         * @return this builder
         */
        public Builder memoryType(@NotNull MemoryType memoryType) {
            this.memoryType = memoryType;
            return this;
        }

        /**
         * Sets the persisted timestamp.
         * @param persistedAt the persisted timestamp
         * @return this builder
         */
        public Builder persistedAt(@Nullable Instant persistedAt) {
            this.persistedAt = persistedAt;
            return this;
        }

        /**
         * Sets the list of IDs this memory was extracted from.
         * @param extractedFrom the list of source IDs
         * @return this builder
         */
        public Builder extractedFrom(@Nullable List<String> extractedFrom) {
            this.extractedFrom = extractedFrom;
            return this;
        }

        /**
         * Sets the event date for this memory.
         * @param eventDate the event date
         * @return this builder
         */
        public Builder eventDate(@Nullable Instant eventDate) {
            this.eventDate = eventDate;
            return this;
        }

        /**
         * Builds the MemoryRecord instance.
         * @return a new MemoryRecord
         * @throws IllegalStateException if required fields are not set
         */
        public MemoryRecord build() {
            if (text == null) {
                throw new IllegalStateException("text is required");
            }

            MemoryRecord record = new MemoryRecord();
            record.id = this.id;
            record.text = this.text;
            record.sessionId = this.sessionId;
            record.userId = this.userId;
            record.namespace = this.namespace;
            record.lastAccessed = this.lastAccessed;
            record.createdAt = this.createdAt;
            record.updatedAt = this.updatedAt;
            record.topics = this.topics;
            record.entities = this.entities;
            record.memoryHash = this.memoryHash;
            record.discreteMemoryExtracted = this.discreteMemoryExtracted;
            record.memoryType = this.memoryType;
            record.persistedAt = this.persistedAt;
            record.extractedFrom = this.extractedFrom;
            record.eventDate = this.eventDate;
            return record;
        }
    }
}
