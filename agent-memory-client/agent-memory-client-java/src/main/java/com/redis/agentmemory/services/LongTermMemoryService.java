package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.longtermemory.*;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Service for long-term memory operations.
 */
public class LongTermMemoryService extends BaseService {

    public LongTermMemoryService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        super(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }

    /**
     * Create long-term memories.
     *
     * @param memories List of memory records to create
     * @return AckResponse indicating success
     * @throws MemoryClientException if the request fails
     */
    public AckResponse createLongTermMemories(@NotNull List<MemoryRecord> memories) throws MemoryClientException {
        Map<String, Object> payload = new HashMap<>();
        payload.put("memories", memories);

        try {
            String json = objectMapper.writeValueAsString(payload);
            RequestBody body = RequestBody.create(json, JSON);

            Request request = new Request.Builder()
                    .url(baseUrl + "/v1/long-term-memory/")
                    .post(body)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), AckResponse.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to create long-term memories", e);
        }
    }

    /**
     * Search long-term memories.
     *
     * @param request Search request with query and filters
     * @return MemoryRecordResults containing matching memories
     * @throws MemoryClientException if the request fails
     */
    public MemoryRecordResults searchLongTermMemories(@NotNull SearchRequest request) throws MemoryClientException {
        // Build payload
        Map<String, Object> payload = new HashMap<>();
        payload.put("text", request.getText());
        payload.put("limit", request.getLimit());
        payload.put("offset", request.getOffset());

        // Add filters if present
        if (request.getSessionId() != null) {
            payload.put("session_id", Map.of("eq", request.getSessionId()));
        }
        if (request.getUserId() != null) {
            payload.put("user_id", Map.of("eq", request.getUserId()));
        }
        if (request.getNamespace() != null) {
            payload.put("namespace", Map.of("eq", request.getNamespace()));
        } else if (defaultNamespace != null) {
            payload.put("namespace", Map.of("eq", defaultNamespace));
        }

        if (request.getTopics() != null && !request.getTopics().isEmpty()) {
            payload.put("topics", Map.of("any", request.getTopics()));
        }
        if (request.getEntities() != null && !request.getEntities().isEmpty()) {
            payload.put("entities", Map.of("any", request.getEntities()));
        }

        try {
            String json = objectMapper.writeValueAsString(payload);
            RequestBody body = RequestBody.create(json, JSON);

            Request httpRequest = new Request.Builder()
                    .url(baseUrl + "/v1/long-term-memory/search")
                    .post(body)
                    .build();

            try (Response response = httpClient.newCall(httpRequest).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), MemoryRecordResults.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to search long-term memories", e);
        }
    }

    /**
     * Search long-term memories with simple text query.
     */
    public MemoryRecordResults searchLongTermMemories(@NotNull String text) throws MemoryClientException {
        SearchRequest request = SearchRequest.builder()
                .text(text)
                .build();
        return searchLongTermMemories(request);
    }

    /**
     * Get a single long-term memory by ID.
     *
     * @param memoryId The memory ID to retrieve
     * @return MemoryRecord if found
     * @throws MemoryClientException if the request fails
     */
    public MemoryRecord getLongTermMemory(@NotNull String memoryId) throws MemoryClientException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/long-term-memory/" + memoryId)
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);

            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }

            return objectMapper.readValue(body.string(), MemoryRecord.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to get long-term memory", e);
        }
    }

    /**
     * Edit a long-term memory.
     *
     * @param memoryId The memory ID to edit
     * @param updates Map of fields to update
     * @return AckResponse indicating success
     * @throws MemoryClientException if the request fails
     */
    public AckResponse editLongTermMemory(
            @NotNull String memoryId,
            @NotNull Map<String, Object> updates) throws MemoryClientException {
        try {
            String json = objectMapper.writeValueAsString(updates);
            RequestBody body = RequestBody.create(json, JSON);

            Request request = new Request.Builder()
                    .url(baseUrl + "/v1/long-term-memory/" + memoryId)
                    .patch(body)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), AckResponse.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to edit long-term memory", e);
        }
    }

    /**
     * Delete long-term memories by IDs.
     *
     * @param memoryIds List of memory IDs to delete
     * @return AckResponse indicating success
     * @throws MemoryClientException if the request fails
     */
    public AckResponse deleteLongTermMemories(@NotNull List<String> memoryIds) throws MemoryClientException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(baseUrl + "/v1/long-term-memory").newBuilder();

        // Add memory_ids as query parameters
        for (String memoryId : memoryIds) {
            urlBuilder.addQueryParameter("memory_ids", memoryId);
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
            throw new MemoryClientException("Failed to delete long-term memories", e);
        }
    }

    /**
     * Run a forgetting pass with the provided policy. Returns summary data.
     * This is an admin-style endpoint for managing memory lifecycle.
     *
     * @param policy Policy configuration for forgetting (max_age_days, max_inactive_days, budget, memory_type_allowlist)
     * @param namespace Optional namespace filter
     * @param userId Optional user ID filter
     * @param sessionId Optional session ID filter
     * @param limit Maximum number of memories to scan (default: 1000)
     * @param dryRun If true, only simulate deletion without actually deleting (default: true)
     * @param pinnedIds Optional list of memory IDs to protect from deletion
     * @return ForgetResponse with scanned count, deleted count, deleted IDs, and dry_run flag
     * @throws MemoryClientException if the request fails
     */
    public ForgetResponse forgetLongTermMemories(
            @NotNull Map<String, Object> policy,
            @Nullable String namespace,
            @Nullable String userId,
            @Nullable String sessionId,
            int limit,
            boolean dryRun,
            @Nullable List<String> pinnedIds) throws MemoryClientException {

        HttpUrl.Builder urlBuilder = HttpUrl.parse(baseUrl + "/v1/long-term-memory/forget").newBuilder();

        // Add query parameters
        if (namespace != null) {
            urlBuilder.addQueryParameter("namespace", namespace);
        }
        if (userId != null) {
            urlBuilder.addQueryParameter("user_id", userId);
        }
        if (sessionId != null) {
            urlBuilder.addQueryParameter("session_id", sessionId);
        }
        urlBuilder.addQueryParameter("limit", String.valueOf(limit));
        urlBuilder.addQueryParameter("dry_run", String.valueOf(dryRun));

        // Build request body
        Map<String, Object> payload = new HashMap<>();
        payload.put("policy", policy);
        if (pinnedIds != null) {
            payload.put("pinned_ids", pinnedIds);
        }

        try {
            String json = objectMapper.writeValueAsString(payload);
            RequestBody body = RequestBody.create(json, JSON);

            Request request = new Request.Builder()
                    .url(urlBuilder.build())
                    .post(body)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), ForgetResponse.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to forget long-term memories", e);
        }
    }

    /**
     * Run a forgetting pass with the provided policy using default parameters.
     * This is a convenience method with dry_run=true and limit=1000.
     *
     * @param policy Policy configuration for forgetting
     * @return ForgetResponse with scanned count, deleted count, deleted IDs, and dry_run flag
     * @throws MemoryClientException if the request fails
     */
    public ForgetResponse forgetLongTermMemories(@NotNull Map<String, Object> policy) throws MemoryClientException {
        return forgetLongTermMemories(policy, null, null, null, 1000, true, null);
    }

    // ===== Batch Operations =====

    /**
     * Create multiple batches of memories with proper rate limiting.
     *
     * @param memoryBatches List of memory record batches
     * @param batchSize Maximum memories per batch request
     * @param delayBetweenBatchesMs Delay in milliseconds between batches
     * @return List of acknowledgement responses for each batch
     * @throws MemoryClientException if the request fails
     */
    public List<AckResponse> bulkCreateLongTermMemories(
            @NotNull List<List<MemoryRecord>> memoryBatches,
            int batchSize,
            long delayBetweenBatchesMs) throws MemoryClientException {
        List<AckResponse> results = new ArrayList<>();

        for (List<MemoryRecord> batch : memoryBatches) {
            // Split large batches into smaller chunks
            for (int i = 0; i < batch.size(); i += batchSize) {
                int end = Math.min(i + batchSize, batch.size());
                List<MemoryRecord> chunk = batch.subList(i, end);

                AckResponse response = createLongTermMemories(chunk);
                results.add(response);

                // Rate limiting delay
                if (delayBetweenBatchesMs > 0 && (i + batchSize) < batch.size()) {
                    try {
                        Thread.sleep(delayBetweenBatchesMs);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new MemoryClientException("Interrupted during batch delay", e);
                    }
                }
            }
        }

        return results;
    }

    // ===== Pagination Utilities =====

    /**
     * Auto-paginating search that yields all matching long-term memory results.
     * Automatically handles pagination to retrieve all results without requiring manual offset management.
     *
     * @param text Search query text
     * @param sessionId Optional session ID filter
     * @param namespace Optional namespace filter
     * @param topics Optional topics filter (comma-separated or list)
     * @param entities Optional entities filter (comma-separated or list)
     * @param userId Optional user ID filter
     * @param batchSize Number of results to fetch per API call
     * @return Iterator over all matching memory records
     */
    public Iterator<MemoryRecord> searchAllLongTermMemories(
            @NotNull String text,
            @Nullable String sessionId,
            @Nullable String namespace,
            @Nullable List<String> topics,
            @Nullable List<String> entities,
            @Nullable String userId,
            int batchSize) {

        return new Iterator<>() {
            private int offset = 0;
            private List<MemoryRecord> currentBatch = new ArrayList<>();
            private int currentIndex = 0;
            private boolean hasMore = true;

            @Override
            public boolean hasNext() {
                // If we have items in current batch, return true
                if (currentIndex < currentBatch.size()) {
                    return true;
                }

                // If we've exhausted all results, return false
                if (!hasMore) {
                    return false;
                }

                // Try to fetch next batch
                try {
                    SearchRequest request = SearchRequest.builder()
                            .text(text)
                            .limit(batchSize)
                            .offset(offset)
                            .namespace(namespace)
                            .userId(userId)
                            .sessionId(sessionId)
                            .topics(topics)
                            .entities(entities)
                            .build();
                    MemoryRecordResults results = searchLongTermMemories(request);

                    // Convert MemoryRecordResult to MemoryRecord (MemoryRecordResult extends MemoryRecord)
                    currentBatch = new ArrayList<>(results.getMemories());
                    currentIndex = 0;
                    offset += batchSize;

                    // If we got fewer results than batch size, we've reached the end
                    if (currentBatch.isEmpty() || currentBatch.size() < batchSize) {
                        hasMore = false;
                    }

                    return !currentBatch.isEmpty();
                } catch (MemoryClientException e) {
                    throw new RuntimeException("Failed to fetch next batch", e);
                }
            }

            @Override
            public MemoryRecord next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return currentBatch.get(currentIndex++);
            }
        };
    }

    /**
     * Auto-paginating search that returns a Stream of all matching long-term memory results.
     *
     * @param text Search query text
     * @param sessionId Optional session ID filter
     * @param namespace Optional namespace filter
     * @param topics Optional topics filter (comma-separated or list)
     * @param entities Optional entities filter (comma-separated or list)
     * @param userId Optional user ID filter
     * @param batchSize Number of results to fetch per API call
     * @return Stream of all matching memory records
     */
    public Stream<MemoryRecord> searchAllLongTermMemoriesStream(
            @NotNull String text,
            @Nullable String sessionId,
            @Nullable String namespace,
            @Nullable List<String> topics,
            @Nullable List<String> entities,
            @Nullable String userId,
            int batchSize) {

        Iterator<MemoryRecord> iterator = searchAllLongTermMemories(
                text, sessionId, namespace, topics, entities, userId, batchSize
        );

        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(iterator, Spliterator.ORDERED),
                false
        );
    }
}
