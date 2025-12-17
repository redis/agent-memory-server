package com.redis.agentmemory.models.workingmemory;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.redis.agentmemory.models.longtermemory.MemoryRecord;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Working memory for a session - contains both messages and structured memory records.
 */
public class WorkingMemory {
    
    @NotNull
    private List<MemoryMessage> messages;
    
    @NotNull
    private List<MemoryRecord> memories;
    
    @Nullable
    private Map<String, Object> data;
    
    @Nullable
    private String context;
    
    @Nullable
    @JsonProperty("user_id")
    private String userId;
    
    private int tokens;
    
    @NotNull
    @JsonProperty("session_id")
    private String sessionId;
    
    @Nullable
    private String namespace;
    
    @NotNull
    @JsonProperty("long_term_memory_strategy")
    private MemoryStrategyConfig longTermMemoryStrategy;
    
    @Nullable
    @JsonProperty("ttl_seconds")
    private Integer ttlSeconds;
    
    @NotNull
    @JsonProperty("last_accessed")
    private Instant lastAccessed;
    
    public WorkingMemory() {
        this.messages = new ArrayList<>();
        this.memories = new ArrayList<>();
        this.data = new HashMap<>();
        this.tokens = 0;
        this.longTermMemoryStrategy = new MemoryStrategyConfig();
        this.lastAccessed = Instant.now();
    }
    
    public WorkingMemory(@NotNull String sessionId) {
        this();
        this.sessionId = sessionId;
    }
    
    // Getters and setters
    
    @NotNull
    public List<MemoryMessage> getMessages() {
        return messages;
    }
    
    public void setMessages(@NotNull List<MemoryMessage> messages) {
        this.messages = messages;
    }
    
    @NotNull
    public List<MemoryRecord> getMemories() {
        return memories;
    }
    
    public void setMemories(@NotNull List<MemoryRecord> memories) {
        this.memories = memories;
    }
    
    @Nullable
    public Map<String, Object> getData() {
        return data;
    }
    
    public void setData(@Nullable Map<String, Object> data) {
        this.data = data;
    }
    
    @Nullable
    public String getContext() {
        return context;
    }
    
    public void setContext(@Nullable String context) {
        this.context = context;
    }
    
    @Nullable
    public String getUserId() {
        return userId;
    }
    
    public void setUserId(@Nullable String userId) {
        this.userId = userId;
    }
    
    public int getTokens() {
        return tokens;
    }
    
    public void setTokens(int tokens) {
        this.tokens = tokens;
    }
    
    @NotNull
    public String getSessionId() {
        return sessionId;
    }
    
    public void setSessionId(@NotNull String sessionId) {
        this.sessionId = sessionId;
    }
    
    @Nullable
    public String getNamespace() {
        return namespace;
    }
    
    public void setNamespace(@Nullable String namespace) {
        this.namespace = namespace;
    }
    
    @NotNull
    public MemoryStrategyConfig getLongTermMemoryStrategy() {
        return longTermMemoryStrategy;
    }

    public void setLongTermMemoryStrategy(@NotNull MemoryStrategyConfig longTermMemoryStrategy) {
        this.longTermMemoryStrategy = longTermMemoryStrategy;
    }

    @Nullable
    public Integer getTtlSeconds() {
        return ttlSeconds;
    }

    public void setTtlSeconds(@Nullable Integer ttlSeconds) {
        this.ttlSeconds = ttlSeconds;
    }

    @NotNull
    public Instant getLastAccessed() {
        return lastAccessed;
    }

    public void setLastAccessed(@NotNull Instant lastAccessed) {
        this.lastAccessed = lastAccessed;
    }

    @Override
    public String toString() {
        return "WorkingMemory{" +
                "messages=" + messages +
                ", memories=" + memories +
                ", data=" + data +
                ", context='" + context + '\'' +
                ", userId='" + userId + '\'' +
                ", tokens=" + tokens +
                ", sessionId='" + sessionId + '\'' +
                ", namespace='" + namespace + '\'' +
                ", longTermMemoryStrategy=" + longTermMemoryStrategy +
                ", ttlSeconds=" + ttlSeconds +
                ", lastAccessed=" + lastAccessed +
                '}';
    }

    /**
     * Creates a new builder for WorkingMemory.
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for WorkingMemory.
     */
    public static class Builder {
        private List<MemoryMessage> messages;
        private List<MemoryRecord> memories;
        private Map<String, Object> data;
        private String context;
        private String userId;
        private int tokens;
        private String sessionId;
        private String namespace;
        private MemoryStrategyConfig longTermMemoryStrategy;
        private Integer ttlSeconds;
        private Instant lastAccessed;

        private Builder() {
            // Initialize with defaults
            this.messages = new ArrayList<>();
            this.memories = new ArrayList<>();
            this.data = new HashMap<>();
            this.tokens = 0;
            this.longTermMemoryStrategy = new MemoryStrategyConfig();
            this.lastAccessed = Instant.now();
        }

        /**
         * Sets the session ID.
         * @param sessionId the session ID
         * @return this builder
         */
        public Builder sessionId(@NotNull String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        /**
         * Sets the messages list.
         * @param messages the list of messages
         * @return this builder
         */
        public Builder messages(@NotNull List<MemoryMessage> messages) {
            this.messages = messages;
            return this;
        }

        /**
         * Adds a single message.
         * @param message the message to add
         * @return this builder
         */
        public Builder addMessage(@NotNull MemoryMessage message) {
            this.messages.add(message);
            return this;
        }

        /**
         * Sets the memories list.
         * @param memories the list of memory records
         * @return this builder
         */
        public Builder memories(@NotNull List<MemoryRecord> memories) {
            this.memories = memories;
            return this;
        }

        /**
         * Adds a single memory record.
         * @param memory the memory record to add
         * @return this builder
         */
        public Builder addMemory(@NotNull MemoryRecord memory) {
            this.memories.add(memory);
            return this;
        }

        /**
         * Sets the data map.
         * @param data the data map
         * @return this builder
         */
        public Builder data(@Nullable Map<String, Object> data) {
            this.data = data;
            return this;
        }

        /**
         * Adds a single data entry.
         * @param key the data key
         * @param value the data value
         * @return this builder
         */
        public Builder addData(@NotNull String key, @Nullable Object value) {
            if (this.data == null) {
                this.data = new HashMap<>();
            }
            this.data.put(key, value);
            return this;
        }

        /**
         * Sets the context.
         * @param context the context string
         * @return this builder
         */
        public Builder context(@Nullable String context) {
            this.context = context;
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
         * Sets the token count.
         * @param tokens the token count
         * @return this builder
         */
        public Builder tokens(int tokens) {
            this.tokens = tokens;
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
         * Sets the long-term memory strategy configuration.
         * @param longTermMemoryStrategy the strategy configuration
         * @return this builder
         */
        public Builder longTermMemoryStrategy(@NotNull MemoryStrategyConfig longTermMemoryStrategy) {
            this.longTermMemoryStrategy = longTermMemoryStrategy;
            return this;
        }

        /**
         * Sets the TTL in seconds.
         * @param ttlSeconds the TTL in seconds
         * @return this builder
         */
        public Builder ttlSeconds(@Nullable Integer ttlSeconds) {
            this.ttlSeconds = ttlSeconds;
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
         * Builds the WorkingMemory instance.
         * @return a new WorkingMemory
         * @throws IllegalStateException if required fields are not set
         */
        public WorkingMemory build() {
            if (sessionId == null) {
                throw new IllegalStateException("sessionId is required");
            }

            WorkingMemory workingMemory = new WorkingMemory();
            workingMemory.messages = this.messages;
            workingMemory.memories = this.memories;
            workingMemory.data = this.data;
            workingMemory.context = this.context;
            workingMemory.userId = this.userId;
            workingMemory.tokens = this.tokens;
            workingMemory.sessionId = this.sessionId;
            workingMemory.namespace = this.namespace;
            workingMemory.longTermMemoryStrategy = this.longTermMemoryStrategy;
            workingMemory.ttlSeconds = this.ttlSeconds;
            workingMemory.lastAccessed = this.lastAccessed;
            return workingMemory;
        }
    }
}

