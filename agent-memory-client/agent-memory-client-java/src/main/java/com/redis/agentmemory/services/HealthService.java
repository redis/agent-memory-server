package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.exceptions.MemoryServerException;
import com.redis.agentmemory.models.health.HealthCheckResponse;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;

/**
 * Service for health check operations.
 */
public class HealthService extends BaseService {
    
    public HealthService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        super(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }
    
    /**
     * Check the health of the memory server.
     * 
     * @return HealthCheckResponse with current server timestamp
     * @throws MemoryClientException if the request fails
     */
    public HealthCheckResponse healthCheck() throws MemoryClientException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/health")
                .get()
                .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                handleHttpError(response);
            }
            
            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryServerException("Empty response body");
            }
            
            return objectMapper.readValue(body.string(), HealthCheckResponse.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to execute health check", e);
        }
    }
}

