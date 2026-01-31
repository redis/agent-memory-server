package com.redis.agentmemory.services;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.agentmemory.exceptions.MemoryClientException;
import com.redis.agentmemory.models.common.AckResponse;
import com.redis.agentmemory.models.summaryview.*;
import com.redis.agentmemory.models.task.Task;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Service for summary view operations.
 */
public class SummaryViewService extends BaseService {

    public SummaryViewService(
            @NotNull String baseUrl,
            @NotNull OkHttpClient httpClient,
            @NotNull ObjectMapper objectMapper,
            @Nullable String defaultNamespace,
            @Nullable String defaultModelName,
            @Nullable Integer defaultContextWindowMax) {
        super(baseUrl, httpClient, objectMapper, defaultNamespace, defaultModelName, defaultContextWindowMax);
    }

    /**
     * List all summary views.
     *
     * @return List of summary views
     * @throws MemoryClientException if the request fails
     */
    public List<SummaryView> listSummaryViews() throws MemoryClientException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/summary-views")
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);

            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }

            return objectMapper.readValue(body.string(), new TypeReference<List<SummaryView>>() {});
        } catch (IOException e) {
            throw new MemoryClientException("Failed to list summary views", e);
        }
    }

    /**
     * Create a new summary view.
     *
     * @param createRequest The creation request
     * @return The created summary view
     * @throws MemoryClientException if the request fails
     */
    public SummaryView createSummaryView(@NotNull CreateSummaryViewRequest createRequest) throws MemoryClientException {
        try {
            String json = objectMapper.writeValueAsString(createRequest);
            RequestBody body = RequestBody.create(json, JSON);

            Request request = new Request.Builder()
                    .url(baseUrl + "/v1/summary-views")
                    .post(body)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), SummaryView.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to create summary view", e);
        }
    }

    /**
     * Get a summary view by ID.
     *
     * @param viewId The view ID
     * @return The summary view
     * @throws MemoryClientException if the request fails
     */
    public SummaryView getSummaryView(@NotNull String viewId) throws MemoryClientException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/summary-views/" + viewId)
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            handleHttpError(response);

            ResponseBody body = response.body();
            if (body == null) {
                throw new MemoryClientException("Empty response body");
            }

            return objectMapper.readValue(body.string(), SummaryView.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to get summary view", e);
        }
    }

    /**
     * Delete a summary view.
     *
     * @param viewId The view ID to delete
     * @return Acknowledgement response
     * @throws MemoryClientException if the request fails
     */
    public AckResponse deleteSummaryView(@NotNull String viewId) throws MemoryClientException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/summary-views/" + viewId)
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
            throw new MemoryClientException("Failed to delete summary view", e);
        }
    }

    /**
     * Run a summary view for a specific partition.
     *
     * @param viewId The view ID
     * @param group Map of group_by field to value for the partition
     * @return The partition summary result
     * @throws MemoryClientException if the request fails
     */
    public SummaryViewPartitionResult runSummaryViewPartition(
            @NotNull String viewId,
            @NotNull Map<String, String> group) throws MemoryClientException {
        try {
            String json = objectMapper.writeValueAsString(group);
            RequestBody body = RequestBody.create(json, JSON);

            Request request = new Request.Builder()
                    .url(baseUrl + "/v1/summary-views/" + viewId + "/partitions/run")
                    .post(body)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                handleHttpError(response);

                ResponseBody responseBody = response.body();
                if (responseBody == null) {
                    throw new MemoryClientException("Empty response body");
                }

                return objectMapper.readValue(responseBody.string(), SummaryViewPartitionResult.class);
            }
        } catch (IOException e) {
            throw new MemoryClientException("Failed to run summary view partition", e);
        }
    }

    /**
     * List partitions for a summary view.
     *
     * @param viewId The view ID
     * @param limit Maximum number of results
     * @param offset Offset for pagination
     * @return List of partition results
     * @throws MemoryClientException if the request fails
     */
    public List<SummaryViewPartitionResult> listSummaryViewPartitions(
            @NotNull String viewId,
            int limit,
            int offset) throws MemoryClientException {

        HttpUrl.Builder urlBuilder = HttpUrl.parse(baseUrl + "/v1/summary-views/" + viewId + "/partitions").newBuilder();
        urlBuilder.addQueryParameter("limit", String.valueOf(limit));
        urlBuilder.addQueryParameter("offset", String.valueOf(offset));

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

            return objectMapper.readValue(body.string(), new TypeReference<List<SummaryViewPartitionResult>>() {});
        } catch (IOException e) {
            throw new MemoryClientException("Failed to list summary view partitions", e);
        }
    }

    /**
     * Run a full summary view (all partitions) as a background task.
     *
     * @param viewId The view ID
     * @param force Whether to force re-summarization of all partitions
     * @return The background task
     * @throws MemoryClientException if the request fails
     */
    public Task runSummaryView(@NotNull String viewId, boolean force) throws MemoryClientException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(baseUrl + "/v1/summary-views/" + viewId + "/run").newBuilder();
        urlBuilder.addQueryParameter("force", String.valueOf(force));

        // Empty POST body
        RequestBody body = RequestBody.create("", JSON);

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

            return objectMapper.readValue(responseBody.string(), Task.class);
        } catch (IOException e) {
            throw new MemoryClientException("Failed to run summary view", e);
        }
    }
}
