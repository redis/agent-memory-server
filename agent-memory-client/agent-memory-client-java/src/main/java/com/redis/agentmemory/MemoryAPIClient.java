package com.redis.agentmemory;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.exceptions.MemoryValidationException;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.longtermemory.*;
import com.redis.agentmemory.models.workingmemory.*;
import com.redis.agentmemory.services.*;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Client for the Agent Memory Server REST API.
 * <p>
 * This client provides methods to interact with all server endpoints:
 * - Health check
 * - Session management (list, get, put, delete)
 * - Long-term memory (create, search)
 */
public class MemoryAPIClient implements AutoCloseable {

    private static final String VERSION = "0.1.0";

    private final String baseUrl;
    private final double timeout;
    private final String defaultNamespace;
    private final String defaultModelName;
    private final Integer defaultContextWindowMax;
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;

    // Service instances
    private final HealthService healthService;
    private final WorkingMemoryService workingMemoryService;
    private final LongTermMemoryService longTermMemoryService;
    private final MemoryHydrationService memoryHydrationService;

    private MemoryAPIClient(Builder builder) {
        this.baseUrl = builder.baseUrl;
        this.timeout = builder.timeout;
        this.defaultNamespace = builder.defaultNamespace;
        this.defaultModelName = builder.defaultModelName;
        this.defaultContextWindowMax = builder.defaultContextWindowMax;

        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout((long) timeout, TimeUnit.SECONDS)
                .readTimeout((long) timeout, TimeUnit.SECONDS)
                .writeTimeout((long) timeout, TimeUnit.SECONDS)
                .addInterceptor(chain -> {
                    Request original = chain.request();
                    Request request = original.newBuilder()
                            .header("User-Agent", "agent-memory-client-java/" + VERSION)
                            .header("X-Client-Version", VERSION)
                            .build();
                    return chain.proceed(request);
                })
                .build();

        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule())
                .configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false)
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        // Initialize services
        this.healthService = new HealthService(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
        this.workingMemoryService = new WorkingMemoryService(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
        this.longTermMemoryService = new LongTermMemoryService(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
        this.memoryHydrationService = new MemoryHydrationService(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }

    @NotNull
    public String getBaseUrl() {
        return baseUrl;
    }

    public double getTimeout() {
        return timeout;
    }

    @Nullable
    public String getDefaultNamespace() {
        return defaultNamespace;
    }

    @Nullable
    public String getDefaultModelName() {
        return defaultModelName;
    }

    @Nullable
    public Integer getDefaultContextWindowMax() {
        return defaultContextWindowMax;
    }

    /**
     * Get the health service for health check operations.
     * @return HealthService instance
     */
    @NotNull
    public HealthService health() {
        return healthService;
    }

    /**
     * Get the working memory service for session management operations.
     * @return WorkingMemoryService instance
     */
    @NotNull
    public WorkingMemoryService workingMemory() {
        return workingMemoryService;
    }

    /**
     * Get the long-term memory service for persistent memory operations.
     * @return LongTermMemoryService instance
     */
    @NotNull
    public LongTermMemoryService longTermMemory() {
        return longTermMemoryService;
    }

    /**
     * Get the memory hydration service for prompt hydration operations.
     * @return MemoryHydrationService instance
     */
    @NotNull
    public MemoryHydrationService hydration() {
        return memoryHydrationService;
    }

    @Override
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }

    /**
     * Creates a new builder for MemoryAPIClient.
     * @param baseUrl the base URL of the memory server
     * @return a new Builder instance
     */
    public static Builder builder(@NotNull String baseUrl) {
        return new Builder(baseUrl);
    }

    // ===== Memory Lifecycle Management =====

    /**
     * Explicitly promote specific working memories to long-term storage.
     * <p>
     * Note: Memory promotion normally happens automatically when working memory
     * is saved. This method is for cases where you need manual control over
     * the promotion timing or want to promote specific memories immediately.
     *
     * @param sessionId The session containing memories to promote
     * @param memoryIds Specific memory IDs to promote (if null, promotes all unpromoted)
     * @param namespace Optional namespace filter
     * @return Acknowledgement of promotion operation
     * @throws MemoryClientException if the operation fails
     */
    public AckResponse promoteWorkingMemoriesToLongTerm(
            @NotNull String sessionId,
            @Nullable List<String> memoryIds,
            @Nullable String namespace) throws MemoryClientException {

        // Get current working memory
        WorkingMemoryResult result = workingMemoryService.getOrCreateWorkingMemory(
                sessionId, namespace, null, null, null, null);

        WorkingMemoryResponse workingMemory = result.getMemory();

        // Filter memories if specific IDs are requested
        List<MemoryRecord> memoriesToPromote = workingMemory.getMemories();
        if (memoryIds != null && !memoryIds.isEmpty()) {
            memoriesToPromote = workingMemory.getMemories().stream()
                    .filter(memory -> memoryIds.contains(memory.getId()))
                    .collect(java.util.stream.Collectors.toList());
        }

        if (memoriesToPromote.isEmpty()) {
            AckResponse response = new AckResponse();
            response.setStatus("ok");
            return response;
        }

        // Create long-term memories
        return longTermMemoryService.createLongTermMemories(memoriesToPromote);
    }

    /**
     * Promote all working memories to long-term storage for a session.
     *
     * @param sessionId The session containing memories to promote
     * @return Acknowledgement of promotion operation
     * @throws MemoryClientException if the operation fails
     */
    public AckResponse promoteWorkingMemoriesToLongTerm(@NotNull String sessionId)
            throws MemoryClientException {
        return promoteWorkingMemoriesToLongTerm(sessionId, null, null);
    }

    /**
     * Promote specific working memories to long-term storage.
     *
     * @param sessionId The session containing memories to promote
     * @param memoryIds Specific memory IDs to promote
     * @return Acknowledgement of promotion operation
     * @throws MemoryClientException if the operation fails
     */
    public AckResponse promoteWorkingMemoriesToLongTerm(
            @NotNull String sessionId,
            @NotNull List<String> memoryIds) throws MemoryClientException {
        return promoteWorkingMemoriesToLongTerm(sessionId, memoryIds, null);
    }

    // ===== Client-Side Validation =====

    /**
     * Validate memory record before sending to server.
     * <p>
     * Checks:
     * - Required fields are present
     * - Memory type is valid
     * - Text content is not empty
     * - ID format is valid (ULID)
     *
     * @param memory The memory record to validate
     * @throws MemoryValidationException if validation fails with descriptive message
     */
    public void validateMemoryRecord(@NotNull MemoryRecord memory) throws MemoryValidationException {
        // Check text is not empty
        if (memory.getText() == null || memory.getText().trim().isEmpty()) {
            throw new MemoryValidationException("Memory text cannot be empty");
        }

        // Check memory type is valid
        if (memory.getMemoryType() == null) {
            throw new MemoryValidationException("Memory type cannot be null");
        }

        // Validate memory type enum values
        MemoryType type = memory.getMemoryType();
        if (type != MemoryType.EPISODIC && type != MemoryType.SEMANTIC && type != MemoryType.MESSAGE) {
            throw new MemoryValidationException("Invalid memory type: " + type);
        }

        // Check ID format if present
        if (memory.getId() != null && !memory.getId().isEmpty() && !isValidULID(memory.getId())) {
            throw new MemoryValidationException("Invalid ID format: " + memory.getId());
        }
    }

    /**
     * Validate search filter parameters before API call.
     *
     * @param filters Map of filter parameters
     * @throws MemoryValidationException if validation fails
     */
    public void validateSearchFilters(@NotNull Map<String, Object> filters) throws MemoryValidationException {
        Set<String> validFilterKeys = new HashSet<>(Arrays.asList(
                "session_id", "namespace", "topics", "entities", "created_at",
                "last_accessed", "user_id", "distance_threshold", "memory_type",
                "limit", "offset"
        ));

        // Check for invalid keys
        for (String key : filters.keySet()) {
            if (!validFilterKeys.contains(key)) {
                throw new MemoryValidationException("Invalid filter key: " + key);
            }
        }

        // Validate limit
        if (filters.containsKey("limit")) {
            Object limit = filters.get("limit");
            if (!(limit instanceof Integer) || (Integer) limit <= 0) {
                throw new MemoryValidationException("Limit must be a positive integer");
            }
        }

        // Validate offset
        if (filters.containsKey("offset")) {
            Object offset = filters.get("offset");
            if (!(offset instanceof Integer) || (Integer) offset < 0) {
                throw new MemoryValidationException("Offset must be a non-negative integer");
            }
        }

        // Validate distance_threshold
        if (filters.containsKey("distance_threshold")) {
            Object threshold = filters.get("distance_threshold");
            if (!(threshold instanceof Number) || ((Number) threshold).doubleValue() < 0) {
                throw new MemoryValidationException("Distance threshold must be a non-negative number");
            }
        }
    }

    // ===== Helper Methods =====

    /**
     * ULID regex pattern for validation.
     * ULIDs are 26 characters using Crockford's base32 alphabet.
     */
    private static final Pattern ULID_PATTERN = Pattern.compile("[0-7][0-9A-HJKMNP-TV-Z]{25}");

    /**
     * Check if a string is a valid ULID format.
     *
     * @param ulidStr The string to check
     * @return true if valid ULID format, false otherwise
     */
    private boolean isValidULID(String ulidStr) {
        return ULID_PATTERN.matcher(ulidStr).matches();
    }



    /**
     * Builder for MemoryAPIClient.
     */
    public static class Builder {
        private final String baseUrl;
        private double timeout = 30.0;
        private String defaultNamespace = null;
        private String defaultModelName = null;
        private Integer defaultContextWindowMax = null;

        private Builder(@NotNull String baseUrl) {
            this.baseUrl = baseUrl;
        }

        /**
         * Sets the timeout for HTTP requests in seconds.
         * @param timeout the timeout in seconds (default: 30.0)
         * @return this builder
         */
        public Builder timeout(double timeout) {
            this.timeout = timeout;
            return this;
        }

        /**
         * Sets the default namespace for operations.
         * @param defaultNamespace the default namespace
         * @return this builder
         */
        public Builder defaultNamespace(@Nullable String defaultNamespace) {
            this.defaultNamespace = defaultNamespace;
            return this;
        }

        /**
         * Sets the default model name for operations.
         * @param defaultModelName the default model name
         * @return this builder
         */
        public Builder defaultModelName(@Nullable String defaultModelName) {
            this.defaultModelName = defaultModelName;
            return this;
        }

        /**
         * Sets the default context window maximum.
         * @param defaultContextWindowMax the default context window maximum
         * @return this builder
         */
        public Builder defaultContextWindowMax(@Nullable Integer defaultContextWindowMax) {
            this.defaultContextWindowMax = defaultContextWindowMax;
            return this;
        }

        /**
         * Builds the MemoryAPIClient instance.
         * @return a new MemoryAPIClient
         */
        public MemoryAPIClient build() {
            return new MemoryAPIClient(this);
        }
    }
}
