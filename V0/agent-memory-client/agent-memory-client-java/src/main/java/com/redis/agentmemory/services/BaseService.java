package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.exceptions.MemoryNotFoundException;
import com.redis.agentmemory.exceptions.MemoryServerException;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;

/**
 * Base service class providing common functionality for all service classes.
 */
public abstract class BaseService {

    protected static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    protected final String baseUrl;
    protected final OkHttpClient httpClient;
    protected final ObjectMapper objectMapper;
    protected final String defaultNamespace;
    protected final String defaultModelName;
    protected final Integer defaultContextWindowMax;

    protected BaseService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        this.baseUrl = baseUrl;
        this.httpClient = httpClient;
        this.objectMapper = objectMapper;
        this.defaultNamespace = defaultNamespace;
        this.defaultModelName = defaultModelName;
        this.defaultContextWindowMax = defaultContextWindowMax;
    }

    /**
     * Handle HTTP errors and throw appropriate exceptions.
     */
    protected void handleHttpError(@NotNull Response response) throws MemoryClientException {
        int statusCode = response.code();

        if (statusCode == 404) {
            throw new MemoryNotFoundException("Resource not found: " + response.request().url());
        }

        if (statusCode >= 400) {
            String message = "HTTP " + statusCode;
            try {
                ResponseBody body = response.body();
                if (body != null) {
                    String bodyString = body.string();
                    // Try to parse error detail from JSON
                    try {
                        var errorData = objectMapper.readTree(bodyString);

                        if (errorData.has("detail")
                                && errorData.get("detail").isArray()
                                && !errorData.get("detail").isEmpty()
                                && errorData.get("detail").get(0).has("msg")) {

                            message = errorData.get("detail").get(0).get("msg").asText();
                        } else {
                            message = "HTTP " + statusCode + ": " + bodyString;
                        }
                    } catch (Exception e) {
                        message = "HTTP " + statusCode + ": " + bodyString;
                    }
                }
            } catch (IOException e) {
                // Ignore, use default message
            }
            throw new MemoryServerException(message, statusCode);
        }
    }
}
