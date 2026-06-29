package com.redis.agentmemory.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.models.task.Task;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;

/**
 * Service for task operations.
 */
public class TaskService extends BaseService {

    public TaskService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        super(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }

    /**
     * Get a task by ID.
     *
     * @param taskId The task ID
     * @return The task
     * @throws MemoryClientException if the request fails
     */
    public Task getTask(@NotNull String taskId) throws MemoryClientException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/tasks/" + taskId)
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);

            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }

            return objectMapper.readValue(body.string(), Task.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to get task", e);
        }
    }
}
