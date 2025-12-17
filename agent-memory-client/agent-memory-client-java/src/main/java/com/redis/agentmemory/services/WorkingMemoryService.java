package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.exceptions.MemoryNotFoundException;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.longtermemory.MemoryRecord;
import com.redis.agentmemory.models.workingmemory.*;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.util.*;

/**
 * Service for working memory operations.
 */
public class WorkingMemoryService extends BaseService {
    
    public WorkingMemoryService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        super(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }
    
    /**
     * List available sessions with optional pagination and filtering.
     * 
     * @param limit Maximum number of sessions to return
     * @param offset Offset for pagination
     * @param namespace Optional namespace filter
     * @param userId Optional user ID filter
     * @return SessionListResponse containing session IDs and total count
     * @throws MemoryClientException if the request fails
     */
    public SessionListResponse listSessions(
            int limit,
            int offset,
            @Nullable String namespace,
            @Nullable String userId
    ) throws MemoryClientException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(baseUrl + "/v1/working-memory/")
                .newBuilder()
                .addQueryParameter("limit", String.valueOf(limit))
                .addQueryParameter("offset", String.valueOf(offset));
        
        if (namespace != null) {
            urlBuilder.addQueryParameter("namespace", namespace);
        } else if (defaultNamespace != null) {
            urlBuilder.addQueryParameter("namespace", defaultNamespace);
        }
        
        if (userId != null) {
            urlBuilder.addQueryParameter("user_id", userId);
        }
        
        Request request = new Request.Builder()
                .url(urlBuilder.build())
                .get()
                .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);
            
            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }
            
            return objectMapper.readValue(body.string(), SessionListResponse.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to list sessions", e);
        }
    }
    
    /**
     * List sessions with default pagination.
     */
    public SessionListResponse listSessions() throws MemoryClientException {
        return listSessions(100, 0, null, null);
    }
    
    /**
     * Get working memory for a session.
     *
     * @param sessionId The session ID to retrieve working memory for
     * @param userId The user ID to retrieve working memory for
     * @param namespace Optional namespace for the session
     * @param modelName Optional model name to determine context window size
     * @param contextWindowMax Optional direct specification of context window tokens
     * @return WorkingMemoryResponse containing messages, context and metadata
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResponse getWorkingMemory(
            @NotNull String sessionId,
            @Nullable String userId,
            @Nullable String namespace,
            @Nullable String modelName,
            @Nullable Integer contextWindowMax
    ) throws MemoryClientException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(
                baseUrl + "/v1/working-memory/" + sessionId
        ).newBuilder();
        
        if (userId != null) {
            urlBuilder.addQueryParameter("user_id", userId);
        }
        
        if (namespace != null) {
            urlBuilder.addQueryParameter("namespace", namespace);
        } else if (defaultNamespace != null) {
            urlBuilder.addQueryParameter("namespace", defaultNamespace);
        }
        
        String effectiveModelName = modelName != null ? modelName : defaultModelName;
        if (effectiveModelName != null) {
            urlBuilder.addQueryParameter("model_name", effectiveModelName);
        }
        
        Integer effectiveContextWindowMax = contextWindowMax != null
                ? contextWindowMax
                : defaultContextWindowMax;
        if (effectiveContextWindowMax != null) {
            urlBuilder.addQueryParameter("context_window_max", String.valueOf(effectiveContextWindowMax));
        }
        
        Request request = new Request.Builder()
                .url(urlBuilder.build())
                .get()
                .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);
            
            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }
            
            return objectMapper.readValue(body.string(), WorkingMemoryResponse.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to get working memory", e);
        }
    }

    /**
     * Get working memory with minimal parameters.
     */
    public WorkingMemoryResponse getWorkingMemory(@NotNull String sessionId) throws MemoryClientException {
        return getWorkingMemory(sessionId, null, null, null, null);
    }

    /**
     * Put (create or update) working memory for a session.
     *
     * @param sessionId The session ID
     * @param memory The working memory to store
     * @param userId Optional user ID
     * @param namespace Optional namespace
     * @param modelName Optional model name
     * @param contextWindowMax Optional context window max
     * @return WorkingMemoryResponse with the stored memory
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResponse putWorkingMemory(
            @NotNull String sessionId,
            @NotNull WorkingMemory memory,
            @Nullable String userId,
            @Nullable String namespace,
            @Nullable String modelName,
            @Nullable Integer contextWindowMax
    ) throws MemoryClientException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(
                baseUrl + "/v1/working-memory/" + sessionId
        ).newBuilder();

        if (userId != null) {
            urlBuilder.addQueryParameter("user_id", userId);
        }

        if (namespace != null) {
            urlBuilder.addQueryParameter("namespace", namespace);
        } else if (defaultNamespace != null) {
            urlBuilder.addQueryParameter("namespace", defaultNamespace);
        }

        String effectiveModelName = modelName != null ? modelName : defaultModelName;
        if (effectiveModelName != null) {
            urlBuilder.addQueryParameter("model_name", effectiveModelName);
        }

        Integer effectiveContextWindowMax = contextWindowMax != null
                ? contextWindowMax
                : defaultContextWindowMax;
        if (effectiveContextWindowMax != null) {
            urlBuilder.addQueryParameter("context_window_max", String.valueOf(effectiveContextWindowMax));
        }

        try {
            String json = objectMapper.writeValueAsString(memory);
            RequestBody body = RequestBody.create(json, JSON);

            Request request = new Request.Builder()
                    .url(urlBuilder.build())
                    .put(body)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), WorkingMemoryResponse.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to put working memory", e);
        }
    }

    /**
     * Put working memory with minimal parameters.
     */
    public WorkingMemoryResponse putWorkingMemory(
            @NotNull String sessionId,
            @NotNull WorkingMemory memory) throws MemoryClientException {
        return putWorkingMemory(sessionId, memory, null, null, null, null);
    }

    /**
     * Delete working memory for a session.
     *
     * @param sessionId The session ID to delete
     * @param userId Optional user ID
     * @param namespace Optional namespace
     * @return AckResponse indicating success
     * @throws MemoryClientException if the request fails
     */
    public AckResponse deleteWorkingMemory(
            @NotNull String sessionId,
            @Nullable String userId,
            @Nullable String namespace
    ) throws MemoryClientException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(
                baseUrl + "/v1/working-memory/" + sessionId
        ).newBuilder();

        if (userId != null) {
            urlBuilder.addQueryParameter("user_id", userId);
        }

        if (namespace != null) {
            urlBuilder.addQueryParameter("namespace", namespace);
        } else if (defaultNamespace != null) {
            urlBuilder.addQueryParameter("namespace", defaultNamespace);
        }

        Request request = new Request.Builder()
                .url(urlBuilder.build())
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);

            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }

            return objectMapper.readValue(body.string(), AckResponse.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to delete working memory", e);
        }
    }

    /**
     * Delete working memory with minimal parameters.
     */
    public AckResponse deleteWorkingMemory(@NotNull String sessionId) throws MemoryClientException {
        return deleteWorkingMemory(sessionId, null, null);
    }

    /**
     * Get working memory for a session, creating it if it doesn't exist.
     *
     * @param sessionId The session ID
     * @param namespace Optional namespace
     * @param userId Optional user ID
     * @param modelName Optional model name for context window management
     * @param contextWindowMax Optional context window max tokens
     * @param longTermMemoryStrategy Optional long-term memory strategy
     * @return WorkingMemoryResult containing a flag indicating if created and the memory
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResult getOrCreateWorkingMemory(
            @NotNull String sessionId,
            @Nullable String namespace,
            @Nullable String userId,
            @Nullable String modelName,
            @Nullable Integer contextWindowMax,
            @Nullable MemoryStrategyConfig longTermMemoryStrategy) throws MemoryClientException {
        try {
            // Try to get existing memory
            WorkingMemoryResponse existing = getWorkingMemory(sessionId, userId, namespace, modelName, contextWindowMax);
            return new WorkingMemoryResult(false, existing);
        } catch (MemoryNotFoundException e) {
            // Memory doesn't exist, create it
            WorkingMemory emptyMemory = WorkingMemory.builder()
                    .sessionId(sessionId)
                    .namespace(namespace != null ? namespace : defaultNamespace)
                    .messages(new ArrayList<>())
                    .memories(new ArrayList<>())
                    .data(new HashMap<>())
                    .userId(userId)
                    .longTermMemoryStrategy(longTermMemoryStrategy != null ? longTermMemoryStrategy : MemoryStrategyConfig.builder().build())
                    .build();

            WorkingMemoryResponse created = putWorkingMemory(sessionId, emptyMemory, userId, namespace, modelName, contextWindowMax);
            return new WorkingMemoryResult(true, created);
        }
    }

    /**
     * Get or create working memory with minimal parameters.
     */
    public WorkingMemoryResult getOrCreateWorkingMemory(@NotNull String sessionId) throws MemoryClientException {
        return getOrCreateWorkingMemory(sessionId, null, null, null, null, null);
    }

    // ===== Enhanced Working Memory Methods =====

    /**
     * Convenience method for setting JSON data in working memory.
     *
     * @param sessionId The session ID
     * @param data The data to set
     * @param namespace Optional namespace
     * @param userId Optional user ID
     * @return WorkingMemoryResponse with updated memory
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResponse setWorkingMemoryData(
            @NotNull String sessionId,
            @NotNull Map<String, Object> data,
            @Nullable String namespace,
            @Nullable String userId) throws MemoryClientException {
        // Get or create existing memory
        WorkingMemoryResult result = getOrCreateWorkingMemory(sessionId, namespace, userId, null, null, null);
        WorkingMemoryResponse existing = result.getMemory();

        // Create updated memory with new data
        WorkingMemory updated = WorkingMemory.builder()
                .sessionId(sessionId)
                .namespace(namespace != null ? namespace : defaultNamespace)
                .messages(existing.getMessages())
                .memories(existing.getMemories())
                .data(data)
                .context(existing.getContext())
                .userId(existing.getUserId())
                .longTermMemoryStrategy(existing.getLongTermMemoryStrategy())
                .build();

        return putWorkingMemory(sessionId, updated, userId, namespace, null, null);
    }

    /**
     * Set working memory data with minimal parameters.
     *
     * @param sessionId Session ID
     * @param data Data to set
     * @return Working memory response
     * @throws MemoryClientException if the operation fails
     */
    public WorkingMemoryResponse setWorkingMemoryData(
            @NotNull String sessionId,
            @NotNull Map<String, Object> data) throws MemoryClientException {
        return setWorkingMemoryData(sessionId, data, defaultNamespace, null);
    }

    /**
     * Add structured memories to working memory without replacing existing ones.
     *
     * @param sessionId The session ID
     * @param memories List of memories to add
     * @param replace If true, replace all existing memories; if false, append
     * @param namespace Optional namespace
     * @return WorkingMemoryResponse with updated memory
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResponse addMemoriesToWorkingMemory(
            @NotNull String sessionId,
            @NotNull List<MemoryRecord> memories,
            boolean replace,
            @Nullable String namespace) throws MemoryClientException {
        // Get or create existing memory
        WorkingMemoryResult result = getOrCreateWorkingMemory(sessionId, namespace, null, null, null, null);
        WorkingMemoryResponse existing = result.getMemory();

        // Determine final memories list
        List<MemoryRecord> finalMemories;
        if (replace || result.isCreated()) {
            finalMemories = new ArrayList<>(memories);
        } else {
            finalMemories = new ArrayList<>(existing.getMemories());
            finalMemories.addAll(memories);
        }

        // Auto-generate IDs for memories that don't have them
        for (int i = 0; i < finalMemories.size(); i++) {
            MemoryRecord memory = finalMemories.get(i);
            if (memory.getId() == null || memory.getId().isEmpty()) {
                // Generate ULID - using a simple UUID for now
                finalMemories.set(i, MemoryRecord.builder()
                        .from(memory)
                        .id(UUID.randomUUID().toString().replace("-", "").toUpperCase())
                        .build());
            }
        }

        // Create updated memory
        WorkingMemory updated = WorkingMemory.builder()
                .sessionId(sessionId)
                .namespace(namespace != null ? namespace : defaultNamespace)
                .messages(existing.getMessages())
                .memories(finalMemories)
                .data(existing.getData())
                .context(existing.getContext())
                .userId(existing.getUserId())
                .longTermMemoryStrategy(existing.getLongTermMemoryStrategy())
                .build();

        return putWorkingMemory(sessionId, updated, null, namespace, null, null);
    }

    /**
     * Add memories to working memory with minimal parameters.
     *
     * @param sessionId Session ID
     * @param memories Memories to add
     * @return Working memory response
     * @throws MemoryClientException if the operation fails
     */
    public WorkingMemoryResponse addMemoriesToWorkingMemory(
            @NotNull String sessionId,
            @NotNull List<MemoryRecord> memories) throws MemoryClientException {
        return addMemoriesToWorkingMemory(sessionId, memories, false, defaultNamespace);
    }

    /**
     * Update specific data fields in working memory without replacing everything.
     *
     * @param sessionId The session ID
     * @param dataUpdates Dictionary of updates to apply
     * @param namespace Optional namespace
     * @param mergeStrategy How to handle existing data
     * @param userId Optional user ID
     * @return WorkingMemoryResponse with updated memory
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResponse updateWorkingMemoryData(
            @NotNull String sessionId,
            @NotNull Map<String, Object> dataUpdates,
            @Nullable String namespace,
            @NotNull MergeStrategy mergeStrategy,
            @Nullable String userId) throws MemoryClientException {
        // Get existing memory
        WorkingMemoryResult result = getOrCreateWorkingMemory(sessionId, namespace, userId, null, null, null);
        WorkingMemoryResponse existing = result.getMemory();

        // Determine final data based on merge strategy
        Map<String, Object> finalData;
        if (existing.getData() != null && !existing.getData().isEmpty()) {
            switch (mergeStrategy) {
                case REPLACE:
                    finalData = new HashMap<>(dataUpdates);
                    break;
                case MERGE:
                    finalData = new HashMap<>(existing.getData());
                    finalData.putAll(dataUpdates);
                    break;
                case DEEP_MERGE:
                    finalData = deepMergeMaps(existing.getData(), dataUpdates);
                    break;
                default:
                    throw new IllegalArgumentException("Invalid merge strategy: " + mergeStrategy);
            }
        } else {
            finalData = new HashMap<>(dataUpdates);
        }

        // Create updated working memory
        WorkingMemory updated = WorkingMemory.builder()
                .sessionId(sessionId)
                .namespace(namespace != null ? namespace : defaultNamespace)
                .messages(existing.getMessages())
                .memories(existing.getMemories())
                .data(finalData)
                .context(existing.getContext())
                .userId(existing.getUserId())
                .longTermMemoryStrategy(existing.getLongTermMemoryStrategy())
                .build();

        return putWorkingMemory(sessionId, updated, userId, namespace, null, null);
    }

    /**
     * Append new messages to existing working memory.
     * More efficient than retrieving, modifying, and setting full memory.
     *
     * @param sessionId The session ID
     * @param messages List of messages to append
     * @param namespace Optional namespace
     * @param modelName Optional model name for token-based summarization
     * @param contextWindowMax Optional context window max tokens
     * @param userId Optional user ID
     * @return WorkingMemoryResponse with updated memory (potentially summarized if token limit exceeded)
     * @throws MemoryClientException if the request fails
     */
    public WorkingMemoryResponse appendMessagesToWorkingMemory(
            @NotNull String sessionId,
            @NotNull List<MemoryMessage> messages,
            @Nullable String namespace,
            @Nullable String modelName,
            @Nullable Integer contextWindowMax,
            @Nullable String userId) throws MemoryClientException {
        // Get existing memory
        WorkingMemoryResult result = getOrCreateWorkingMemory(sessionId, namespace, userId, null, null, null);
        WorkingMemoryResponse existing = result.getMemory();

        // Get existing messages
        List<MemoryMessage> existingMessages = new ArrayList<>(existing.getMessages());

        // Append new messages
        existingMessages.addAll(messages);

        // Create updated working memory
        WorkingMemory updated = WorkingMemory.builder()
                .sessionId(sessionId)
                .namespace(namespace != null ? namespace : defaultNamespace)
                .messages(existingMessages)
                .memories(existing.getMemories())
                .data(existing.getData())
                .context(existing.getContext())
                .userId(userId != null ? userId : existing.getUserId())
                .longTermMemoryStrategy(existing.getLongTermMemoryStrategy())
                .build();

        return putWorkingMemory(sessionId, updated, userId, namespace, modelName, contextWindowMax);
    }

    /**
     * Append messages to working memory with minimal parameters.
     *
     * @param sessionId Session ID
     * @param messages Messages to append
     * @return Working memory response
     * @throws MemoryClientException if the operation fails
     */
    public WorkingMemoryResponse appendMessagesToWorkingMemory(
            @NotNull String sessionId,
            @NotNull List<MemoryMessage> messages) throws MemoryClientException {
        return appendMessagesToWorkingMemory(sessionId, messages, defaultNamespace,
                null, null, null);
    }

    // ===== Helper Methods =====

    /**
     * Deep merge two maps recursively.
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> deepMergeMaps(Map<String, Object> base, Map<String, Object> updates) {
        Map<String, Object> result = new HashMap<>(base);

        for (Map.Entry<String, Object> entry : updates.entrySet()) {
            String key = entry.getKey();
            Object updateValue = entry.getValue();

            if (updateValue instanceof Map && result.get(key) instanceof Map) {
                // Both are maps, recursively merge
                result.put(key, deepMergeMaps(
                        (Map<String, Object>) result.get(key),
                        (Map<String, Object>) updateValue
                ));
            } else {
                // Otherwise, just replace
                result.put(key, updateValue);
            }
        }

        return result;
    }
}
