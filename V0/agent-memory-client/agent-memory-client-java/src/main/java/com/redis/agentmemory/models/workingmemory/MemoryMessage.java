package com.redis.agentmemory.models.workingmemory;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.f4b6a3.ulid.UlidCreator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.time.Instant;

/**
 * A message in the memory system.
 */
public class MemoryMessage {

    @NotNull
    private String role;

    @NotNull
    private String content;

    @NotNull
    private String id;

    @NotNull
    @JsonProperty("created_at")
    private Instant createdAt;

    @Nullable
    @JsonProperty("persisted_at")
    private Instant persistedAt;

    @NotNull
    @JsonProperty("discrete_memory_extracted")
    private String discreteMemoryExtracted;

    public MemoryMessage() {
        this.id = UlidCreator.getUlid().toString();
        this.createdAt = Instant.now();
        this.discreteMemoryExtracted = "f";
    }

    public MemoryMessage(@NotNull String role, @NotNull String content) {
        this();
        this.role = role;
        this.content = content;
    }

    // Getters and setters

    @NotNull
    public String getRole() {
        return role;
    }

    public void setRole(@NotNull String role) {
        this.role = role;
    }

    @NotNull
    public String getContent() {
        return content;
    }

    public void setContent(@NotNull String content) {
        this.content = content;
    }

    @NotNull
    public String getId() {
        return id;
    }

    public void setId(@NotNull String id) {
        this.id = id;
    }

    @NotNull
    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(@NotNull Instant createdAt) {
        this.createdAt = createdAt;
    }

    @Nullable
    public Instant getPersistedAt() {
        return persistedAt;
    }

    public void setPersistedAt(@Nullable Instant persistedAt) {
        this.persistedAt = persistedAt;
    }

    @NotNull
    public String getDiscreteMemoryExtracted() {
        return discreteMemoryExtracted;
    }

    public void setDiscreteMemoryExtracted(@NotNull String discreteMemoryExtracted) {
        this.discreteMemoryExtracted = discreteMemoryExtracted;
    }

    @Override
    public String toString() {
        return "MemoryMessage{" +
                "role='" + role + '\'' +
                ", content='" + content + '\'' +
                ", id='" + id + '\'' +
                ", createdAt=" + createdAt +
                ", persistedAt=" + persistedAt +
                ", discreteMemoryExtracted='" + discreteMemoryExtracted + '\'' +
                '}';
    }

    /**
     * Creates a new builder for MemoryMessage.
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for MemoryMessage.
     */
    public static class Builder {
        private String role;
        private String content;
        private String id;
        private Instant createdAt;
        private Instant persistedAt;
        private String discreteMemoryExtracted;

        private Builder() {
            // Initialize with defaults
            this.id = UlidCreator.getUlid().toString();
            this.createdAt = Instant.now();
            this.discreteMemoryExtracted = "f";
        }

        /**
         * Sets the role of the message.
         * @param role the role (e.g., "user", "assistant", "system")
         * @return this builder
         */
        public Builder role(@NotNull String role) {
            this.role = role;
            return this;
        }

        /**
         * Sets the content of the message.
         * @param content the message content
         * @return this builder
         */
        public Builder content(@NotNull String content) {
            this.content = content;
            return this;
        }

        /**
         * Sets the ID of the message. If not set, a ULID will be generated.
         * @param id the message ID
         * @return this builder
         */
        public Builder id(@NotNull String id) {
            this.id = id;
            return this;
        }

        /**
         * Sets the creation timestamp. If not set, current time will be used.
         * @param createdAt the creation timestamp
         * @return this builder
         */
        public Builder createdAt(@NotNull Instant createdAt) {
            this.createdAt = createdAt;
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
         * Sets whether discrete memory has been extracted.
         * @param discreteMemoryExtracted "t" for true, "f" for false
         * @return this builder
         */
        public Builder discreteMemoryExtracted(@NotNull String discreteMemoryExtracted) {
            this.discreteMemoryExtracted = discreteMemoryExtracted;
            return this;
        }

        /**
         * Builds the MemoryMessage instance.
         * @return a new MemoryMessage
         * @throws IllegalStateException if required fields are not set
         */
        public MemoryMessage build() {
            if (role == null) {
                throw new IllegalStateException("role is required");
            }
            if (content == null) {
                throw new IllegalStateException("content is required");
            }

            MemoryMessage message = new MemoryMessage();
            message.role = this.role;
            message.content = this.content;
            message.id = this.id;
            message.createdAt = this.createdAt;
            message.persistedAt = this.persistedAt;
            message.discreteMemoryExtracted = this.discreteMemoryExtracted;
            return message;
        }
    }
}
