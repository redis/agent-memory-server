package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Service for memory hydration operations (memory prompt).
 */
public class MemoryHydrationService extends BaseService {

    public MemoryHydrationService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        super(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }

    /**
     * Hydrate a user query with memory context and return a prompt ready to send to an LLM.
     *
     * @param query The query for vector search to find relevant context for
     * @param sessionId Optional session ID to include session messages
     * @param namespace Optional namespace for the session
     * @param modelName Optional model name to determine context window size
     * @param contextWindowMax Optional direct specification of context window tokens
     * @param longTermSearch Optional search parameters for long-term memory
     * @param userId Optional user ID for the session
     * @param optimizeQuery Whether to optimize the query for vector search using a fast model
     * @return Map with messages hydrated with relevant memory context
     * @throws MemoryClientException if the request fails
     */
    public Map<String, Object> memoryPrompt(
            @NotNull String query,
            @Nullable String sessionId,
            @Nullable String namespace,
            @Nullable String modelName,
            @Nullable Integer contextWindowMax,
            @Nullable Map<String, Object> longTermSearch,
            @Nullable String userId,
            boolean optimizeQuery) throws MemoryClientException {

        Map<String, Object> payload = new HashMap<>();
        payload.put("query", query);

        // Add session parameters if provided
        if (sessionId != null) {
            Map<String, Object> sessionParams = new HashMap<>();
            sessionParams.put("session_id", sessionId);

            if (namespace != null) {
                sessionParams.put("namespace", namespace);
            } else if (defaultNamespace != null) {
                sessionParams.put("namespace", defaultNamespace);
            }

            String effectiveModelName = modelName != null ? modelName : defaultModelName;
            if (effectiveModelName != null) {
                sessionParams.put("model_name", effectiveModelName);
            }

            Integer effectiveContextWindowMax = contextWindowMax != null
                    ? contextWindowMax
                    : defaultContextWindowMax;
            if (effectiveContextWindowMax != null) {
                sessionParams.put("context_window_max", effectiveContextWindowMax);
            }

            if (userId != null) {
                sessionParams.put("user_id", userId);
            }

            payload.put("session", sessionParams);
        }

        // Add long-term search parameters if provided
        if (longTermSearch != null) {
            Map<String, Object> searchParams = new HashMap<>(longTermSearch);

            // Add namespace to long-term search if not present
            if (!searchParams.containsKey("namespace")) {
                if (namespace != null) {
                    Map<String, String> namespaceFilter = new HashMap<>();
                    namespaceFilter.put("eq", namespace);
                    searchParams.put("namespace", namespaceFilter);
                } else if (defaultNamespace != null) {
                    Map<String, String> namespaceFilter = new HashMap<>();
                    namespaceFilter.put("eq", defaultNamespace);
                    searchParams.put("namespace", namespaceFilter);
                }
            }

            payload.put("long_term_search", searchParams);
        }

        HttpUrl.Builder urlBuilder = HttpUrl.parse(baseUrl + "/v1/memory/prompt").newBuilder();
        urlBuilder.addQueryParameter("optimize_query", String.valueOf(optimizeQuery));

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

                @SuppressWarnings("unchecked")
                Map<String, Object> result = objectMapper.readValue(responseBody.string(), Map.class);
                return result;
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to hydrate memory prompt: " + e.getMessage(), e);
        }
    }

    /**
     * Hydrate a query with minimal parameters.
     */
    public Map<String, Object> memoryPrompt(@NotNull String query) throws MemoryClientException {
        return memoryPrompt(query, null, null, null, null, null, null, false);
    }
}
