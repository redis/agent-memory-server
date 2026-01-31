import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryAPIClient, MemoryClientConfig } from "./client";
import {
  MemoryClientError,
  MemoryNotFoundError,
  MemoryServerError,
} from "./errors";

// Mock fetch helper
function createMockFetch(response: unknown, options: { ok?: boolean; status?: number } = {}) {
  return vi.fn().mockResolvedValue({
    ok: options.ok ?? true,
    status: options.status ?? 200,
    json: vi.fn().mockResolvedValue(response),
    text: vi.fn().mockResolvedValue(JSON.stringify(response)),
  });
}

function createErrorFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: false,
    status,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  });
}

describe("MemoryAPIClient", () => {
  let client: MemoryAPIClient;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch = createMockFetch({});
    client = new MemoryAPIClient({
      baseUrl: "http://localhost:8000",
      fetch: mockFetch,
    });
  });

  describe("constructor", () => {
    it("should create client with base URL", () => {
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
        fetch: mockFetch,
      });
      expect(client).toBeInstanceOf(MemoryAPIClient);
    });

    it("should use global fetch when no custom fetch provided", () => {
      // This tests the ?? fetch fallback branch
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
      });
      expect(client).toBeDefined();
      // The fetchFn should be the global fetch
      expect(client["fetchFn"]).toBe(globalThis.fetch);
    });

    it("should remove trailing slash from base URL", async () => {
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000/",
        fetch: mockFetch,
      });
      mockFetch = createMockFetch({ now: 123 });
      client["fetchFn"] = mockFetch;
      await client.healthCheck();
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/v1/health",
        expect.any(Object)
      );
    });

    it("should use default timeout of 30000ms", () => {
      expect(client["config"].timeout).toBe(30000);
    });

    it("should allow custom timeout", () => {
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
        timeout: 5000,
        fetch: mockFetch,
      });
      expect(client["config"].timeout).toBe(5000);
    });
  });

  describe("headers", () => {
    it("should include API key header when configured", async () => {
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
        apiKey: "test-api-key",
        fetch: createMockFetch({ now: 123 }),
      });
      await client.healthCheck();
      expect(client["fetchFn"]).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-API-Key": "test-api-key",
          }),
        })
      );
    });

    it("should include bearer token when configured", async () => {
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
        bearerToken: "test-token",
        fetch: createMockFetch({ now: 123 }),
      });
      await client.healthCheck();
      expect(client["fetchFn"]).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: "Bearer test-token",
          }),
        })
      );
    });
  });

  describe("healthCheck", () => {
    it("should return health response", async () => {
      const mockResponse = { now: 1234567890 };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.healthCheck();
      expect(result).toEqual(mockResponse);
    });

    it("should call correct endpoint", async () => {
      mockFetch = createMockFetch({ now: 123 });
      client["fetchFn"] = mockFetch;
      await client.healthCheck();
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/v1/health",
        expect.objectContaining({ method: "GET" })
      );
    });
  });

  describe("error handling", () => {
    it("should throw MemoryNotFoundError on 404", async () => {
      client["fetchFn"] = createErrorFetch(404, { detail: "Not found" });
      await expect(client.healthCheck()).rejects.toThrow(MemoryNotFoundError);
    });

    it("should throw MemoryServerError on 500", async () => {
      client["fetchFn"] = createErrorFetch(500, { detail: "Server error" });
      await expect(client.healthCheck()).rejects.toThrow(MemoryServerError);
    });

    it("should include status code in MemoryServerError", async () => {
      client["fetchFn"] = createErrorFetch(503, { detail: "Service unavailable" });
      try {
        await client.healthCheck();
      } catch (error) {
        expect(error).toBeInstanceOf(MemoryServerError);
        expect((error as MemoryServerError).statusCode).toBe(503);
      }
    });

    it("should handle error response with message field", async () => {
      client["fetchFn"] = createErrorFetch(400, { message: "Bad request" });
      await expect(client.healthCheck()).rejects.toThrow("Bad request");
    });

    it("should handle timeout", async () => {
      const abortError = new Error("Aborted");
      abortError.name = "AbortError";
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
        timeout: 1,
        fetch: vi.fn().mockRejectedValue(abortError),
      });
      await expect(client.healthCheck()).rejects.toThrow("timeout");
    });

    it("should handle network errors", async () => {
      client["fetchFn"] = vi.fn().mockRejectedValue(new Error("Network error"));
      await expect(client.healthCheck()).rejects.toThrow(MemoryClientError);
    });

    it("should re-throw MemoryClientError directly", async () => {
      const customError = new MemoryClientError("Custom client error");
      client["fetchFn"] = vi.fn().mockRejectedValue(customError);
      await expect(client.healthCheck()).rejects.toThrow(customError);
    });

    it("should handle error response with message field", async () => {
      client["fetchFn"] = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: vi.fn().mockResolvedValue({ message: "Server message error" }),
      });
      await expect(client.healthCheck()).rejects.toThrow("Server message error");
    });

    it("should handle error response with unknown body structure", async () => {
      client["fetchFn"] = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: vi.fn().mockResolvedValue({ code: 500, type: "error" }),
      });
      await expect(client.healthCheck()).rejects.toThrow('{"code":500,"type":"error"}');
    });

    it("should handle non-JSON error response", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: vi.fn().mockRejectedValue(new Error("Invalid JSON")),
        text: vi.fn().mockResolvedValue("Internal Server Error"),
      });
      client["fetchFn"] = mockFetch;
      await expect(client.healthCheck()).rejects.toThrow("Internal Server Error");
    });
  });

  describe("listSessions", () => {
    it("should list sessions", async () => {
      const mockResponse = { sessions: ["s1", "s2"], total: 2 };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.listSessions();
      expect(result).toEqual(mockResponse);
    });

    it("should pass namespace parameter", async () => {
      mockFetch = createMockFetch({ sessions: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.listSessions({ namespace: "test-ns" });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("namespace=test-ns"),
        expect.any(Object)
      );
    });

    it("should pass limit and offset parameters", async () => {
      mockFetch = createMockFetch({ sessions: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.listSessions({ limit: 10, offset: 20 });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringMatching(/limit=10.*offset=20|offset=20.*limit=10/),
        expect.any(Object)
      );
    });

    it("should use default namespace from config", async () => {
      const client = new MemoryAPIClient({
        baseUrl: "http://localhost:8000",
        defaultNamespace: "default-ns",
        fetch: createMockFetch({ sessions: [], total: 0 }),
      });
      await client.listSessions();
      expect(client["fetchFn"]).toHaveBeenCalledWith(
        expect.stringContaining("namespace=default-ns"),
        expect.any(Object)
      );
    });
  });

  describe("getWorkingMemory", () => {
    it("should get working memory", async () => {
      const mockResponse = { session_id: "test", messages: [] };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.getWorkingMemory("test-session");
      expect(result).toEqual(mockResponse);
    });

    it("should return null on 404", async () => {
      client["fetchFn"] = createErrorFetch(404, { detail: "Not found" });
      const result = await client.getWorkingMemory("nonexistent");
      expect(result).toBeNull();
    });

    it("should encode session ID in URL", async () => {
      mockFetch = createMockFetch({ session_id: "test" });
      client["fetchFn"] = mockFetch;
      await client.getWorkingMemory("session/with/slashes");
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("session%2Fwith%2Fslashes"),
        expect.any(Object)
      );
    });

    it("should pass model_name parameter", async () => {
      mockFetch = createMockFetch({ session_id: "test" });
      client["fetchFn"] = mockFetch;
      await client.getWorkingMemory("test", { modelName: "gpt-4o" });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("model_name=gpt-4o"),
        expect.any(Object)
      );
    });

    it("should re-throw non-404 errors", async () => {
      client["fetchFn"] = createErrorFetch(500, { detail: "Server error" });
      await expect(client.getWorkingMemory("test")).rejects.toThrow(MemoryServerError);
    });
  });

  describe("putWorkingMemory", () => {
    it("should create working memory", async () => {
      const mockResponse = { session_id: "test", messages: [], new_session: true };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.putWorkingMemory("test", {
        messages: [{ role: "user", content: "Hello" }],
      });
      expect(result).toEqual(mockResponse);
    });

    it("should use PUT method", async () => {
      mockFetch = createMockFetch({ session_id: "test" });
      client["fetchFn"] = mockFetch;
      await client.putWorkingMemory("test", {});
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: "PUT" })
      );
    });

    it("should pass background parameter", async () => {
      mockFetch = createMockFetch({ session_id: "test" });
      client["fetchFn"] = mockFetch;
      await client.putWorkingMemory("test", {}, { background: true });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("background=true"),
        expect.any(Object)
      );
    });
  });

  describe("getOrCreateWorkingMemory", () => {
    it("should return existing working memory", async () => {
      const mockResponse = { session_id: "test", messages: [] };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.getOrCreateWorkingMemory("test");
      expect(result).toEqual(mockResponse);
    });

    it("should create new working memory if not found", async () => {
      let callCount = 0;
      client["fetchFn"] = vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            ok: false,
            status: 404,
            json: () => Promise.resolve({ detail: "Not found" }),
            text: () => Promise.resolve("Not found"),
          });
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ session_id: "test", new_session: true }),
        });
      });
      const result = await client.getOrCreateWorkingMemory("test");
      expect(result.session_id).toBe("test");
      expect(callCount).toBe(2);
    });
  });

  describe("deleteWorkingMemory", () => {
    it("should delete working memory", async () => {
      const mockResponse = { status: "ok" };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.deleteWorkingMemory("test");
      expect(result).toEqual(mockResponse);
    });

    it("should use DELETE method", async () => {
      mockFetch = createMockFetch({ status: "ok" });
      client["fetchFn"] = mockFetch;
      await client.deleteWorkingMemory("test");
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: "DELETE" })
      );
    });
  });

  describe("createLongTermMemory", () => {
    it("should create long term memory", async () => {
      const mockResponse = { status: "ok" };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.createLongTermMemory([
        { id: "mem-1", text: "Test memory" },
      ]);
      expect(result).toEqual(mockResponse);
    });

    it("should use POST method", async () => {
      mockFetch = createMockFetch({ status: "ok" });
      client["fetchFn"] = mockFetch;
      await client.createLongTermMemory([{ id: "mem-1", text: "Test" }]);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/long-term-memory/"),
        expect.objectContaining({ method: "POST" })
      );
    });

    it("should send memories in request body", async () => {
      mockFetch = createMockFetch({ status: "ok" });
      client["fetchFn"] = mockFetch;
      const memories = [{ id: "mem-1", text: "Test memory" }];
      await client.createLongTermMemory(memories);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: JSON.stringify({ memories }),
        })
      );
    });
  });

  describe("searchLongTermMemory", () => {
    it("should search long term memory", async () => {
      const mockResponse = { memories: [], total: 0 };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.searchLongTermMemory({ text: "test query" });
      expect(result).toEqual(mockResponse);
    });

    it("should use POST method", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({ text: "test" });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/long-term-memory/search"),
        expect.objectContaining({ method: "POST" })
      );
    });

    it("should include filters in request body", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        limit: 10,
        offset: 5,
        distanceThreshold: 0.5,
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.text).toBe("test");
      expect(callBody.limit).toBe(10);
      expect(callBody.offset).toBe(5);
      expect(callBody.distance_threshold).toBe(0.5);
    });

    it("should handle SessionId filter class", async () => {
      const { SessionId } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        sessionId: new SessionId({ eq: "sess-1" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.session_id).toEqual({ eq: "sess-1" });
    });

    it("should handle plain object sessionId filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        sessionId: { eq: "sess-1" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.session_id).toEqual({ eq: "sess-1" });
    });

    it("should handle Namespace filter class", async () => {
      const { Namespace } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        namespace: new Namespace({ eq: "ns-1" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.namespace).toEqual({ eq: "ns-1" });
    });

    it("should handle Topics filter class", async () => {
      const { Topics } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        topics: new Topics({ any: ["topic1"] }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.topics).toEqual({ any: ["topic1"] });
    });

    it("should handle Entities filter class", async () => {
      const { Entities } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        entities: new Entities({ all: ["entity1"] }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.entities).toEqual({ all: ["entity1"] });
    });

    it("should handle CreatedAt filter class", async () => {
      const { CreatedAt } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        createdAt: new CreatedAt({ gte: "2024-01-01" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.created_at).toEqual({ gte: "2024-01-01" });
    });

    it("should handle LastAccessed filter class", async () => {
      const { LastAccessed } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        lastAccessed: new LastAccessed({ lte: "2024-12-31" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.last_accessed).toEqual({ lte: "2024-12-31" });
    });

    it("should handle UserId filter class", async () => {
      const { UserId } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        userId: new UserId({ eq: "user-1" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.user_id).toEqual({ eq: "user-1" });
    });

    it("should handle MemoryType filter class", async () => {
      const { MemoryType } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        memoryType: new MemoryType({ eq: "episodic" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.memory_type).toEqual({ eq: "episodic" });
    });

    it("should handle EventDate filter class", async () => {
      const { EventDate } = await import("./filters");
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        eventDate: new EventDate({ gte: "2024-01-01", lte: "2024-12-31" }),
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.event_date).toEqual({ gte: "2024-01-01", lte: "2024-12-31" });
    });

    it("should handle recency config", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        recency: {
          recency_boost: true,
          semantic_weight: 0.7,
          recency_weight: 0.3,
          freshness_weight: 0.5,
          novelty_weight: 0.5,
          half_life_last_access_days: 7,
          half_life_created_days: 30,
          server_side_recency: true,
        },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.recency_boost).toBe(true);
      expect(callBody.recency_semantic_weight).toBe(0.7);
      expect(callBody.recency_recency_weight).toBe(0.3);
      expect(callBody.recency_freshness_weight).toBe(0.5);
      expect(callBody.recency_novelty_weight).toBe(0.5);
      expect(callBody.recency_half_life_last_access_days).toBe(7);
      expect(callBody.recency_half_life_created_days).toBe(30);
      expect(callBody.server_side_recency).toBe(true);
    });

    it("should handle plain object namespace filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        namespace: { eq: "ns-1" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.namespace).toEqual({ eq: "ns-1" });
    });

    it("should handle plain object topics filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        topics: { any: ["topic1"] },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.topics).toEqual({ any: ["topic1"] });
    });

    it("should handle plain object entities filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        entities: { all: ["entity1"] },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.entities).toEqual({ all: ["entity1"] });
    });

    it("should handle plain object createdAt filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        createdAt: { gte: "2024-01-01" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.created_at).toEqual({ gte: "2024-01-01" });
    });

    it("should handle plain object lastAccessed filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        lastAccessed: { lte: "2024-12-31" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.last_accessed).toEqual({ lte: "2024-12-31" });
    });

    it("should handle plain object userId filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        userId: { eq: "user-1" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.user_id).toEqual({ eq: "user-1" });
    });

    it("should handle plain object memoryType filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        memoryType: { eq: "episodic" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.memory_type).toEqual({ eq: "episodic" });
    });

    it("should handle plain object eventDate filter", async () => {
      mockFetch = createMockFetch({ memories: [], total: 0 });
      client["fetchFn"] = mockFetch;
      await client.searchLongTermMemory({
        text: "test",
        eventDate: { gte: "2024-01-01" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.event_date).toEqual({ gte: "2024-01-01" });
    });
  });

  describe("getLongTermMemory", () => {
    it("should get long term memory by ID", async () => {
      const mockResponse = { id: "mem-1", text: "Test memory" };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.getLongTermMemory("mem-1");
      expect(result).toEqual(mockResponse);
    });

    it("should return null on 404", async () => {
      client["fetchFn"] = createErrorFetch(404, { detail: "Not found" });
      const result = await client.getLongTermMemory("nonexistent");
      expect(result).toBeNull();
    });

    it("should encode memory ID in URL", async () => {
      mockFetch = createMockFetch({ id: "test" });
      client["fetchFn"] = mockFetch;
      await client.getLongTermMemory("mem/with/slashes");
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("mem%2Fwith%2Fslashes"),
        expect.any(Object)
      );
    });

    it("should re-throw non-404 errors", async () => {
      client["fetchFn"] = createErrorFetch(500, { detail: "Server error" });
      await expect(client.getLongTermMemory("test")).rejects.toThrow(MemoryServerError);
    });
  });

  describe("deleteLongTermMemories", () => {
    it("should delete long term memories", async () => {
      const mockResponse = { status: "ok" };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.deleteLongTermMemories(["mem-1", "mem-2"]);
      expect(result).toEqual(mockResponse);
    });

    it("should use DELETE method", async () => {
      mockFetch = createMockFetch({ status: "ok" });
      client["fetchFn"] = mockFetch;
      await client.deleteLongTermMemories(["mem-1"]);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: "DELETE" })
      );
    });

    it("should send memory_ids as query params not body", async () => {
      mockFetch = createMockFetch({ status: "ok" });
      client["fetchFn"] = mockFetch;
      await client.deleteLongTermMemories(["mem-1", "mem-2"]);
      const calledUrl = (mockFetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(calledUrl).toContain("memory_ids=mem-1");
      expect(calledUrl).toContain("memory_ids=mem-2");
      // Verify no body is sent
      const calledOptions = (mockFetch as ReturnType<typeof vi.fn>).mock.calls[0][1];
      expect(calledOptions.body).toBeUndefined();
    });
  });

  describe("memoryPrompt", () => {
    it("should get memory prompt", async () => {
      const mockResponse = { messages: [{ role: "system", content: "Context" }] };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.memoryPrompt({ query: "test query" });
      expect(result).toEqual(mockResponse);
    });

    it("should use POST method", async () => {
      mockFetch = createMockFetch({ messages: [] });
      client["fetchFn"] = mockFetch;
      await client.memoryPrompt({ query: "test" });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/memory/prompt"),
        expect.objectContaining({ method: "POST" })
      );
    });

    it("should include session params", async () => {
      mockFetch = createMockFetch({ messages: [] });
      client["fetchFn"] = mockFetch;
      await client.memoryPrompt({
        query: "test",
        session: { session_id: "sess-1", namespace: "ns" },
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.session.session_id).toBe("sess-1");
      expect(callBody.session.namespace).toBe("ns");
    });
  });

  describe("editLongTermMemory", () => {
    it("should edit long term memory", async () => {
      const mockResponse = { id: "mem-1", text: "Updated memory" };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.editLongTermMemory("mem-1", { text: "Updated memory" });
      expect(result).toEqual(mockResponse);
    });

    it("should use PATCH method", async () => {
      mockFetch = createMockFetch({ id: "mem-1", text: "Updated" });
      client["fetchFn"] = mockFetch;
      await client.editLongTermMemory("mem-1", { text: "Updated" });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/long-term-memory/mem-1"),
        expect.objectContaining({ method: "PATCH" })
      );
    });

    it("should send updates in request body", async () => {
      mockFetch = createMockFetch({ id: "mem-1", text: "Updated" });
      client["fetchFn"] = mockFetch;
      await client.editLongTermMemory("mem-1", { text: "Updated", topics: ["new-topic"] });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.text).toBe("Updated");
      expect(callBody.topics).toEqual(["new-topic"]);
    });

    it("should encode memory ID in URL", async () => {
      mockFetch = createMockFetch({ id: "test" });
      client["fetchFn"] = mockFetch;
      await client.editLongTermMemory("mem/with/slashes", { text: "test" });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("mem%2Fwith%2Fslashes"),
        expect.any(Object)
      );
    });
  });

  describe("forgetLongTermMemories", () => {
    it("should run forget pass with policy", async () => {
      const mockResponse = { scanned: 100, deleted: 5, deleted_ids: ["m1", "m2"], dry_run: false };
      client["fetchFn"] = createMockFetch(mockResponse);
      const result = await client.forgetLongTermMemories({
        policy: { max_age_days: 30 },
      });
      expect(result).toEqual(mockResponse);
    });

    it("should use POST method", async () => {
      mockFetch = createMockFetch({ scanned: 0, deleted: 0, deleted_ids: [], dry_run: true });
      client["fetchFn"] = mockFetch;
      await client.forgetLongTermMemories({ policy: {} });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/long-term-memory/forget"),
        expect.objectContaining({ method: "POST" })
      );
    });

    it("should send query params", async () => {
      mockFetch = createMockFetch({ scanned: 0, deleted: 0, deleted_ids: [], dry_run: true });
      client["fetchFn"] = mockFetch;
      await client.forgetLongTermMemories({
        policy: { max_age_days: 30 },
        namespace: "test-ns",
        userId: "user-1",
        sessionId: "sess-1",
        limit: 500,
        dryRun: false,
      });
      const calledUrl = mockFetch.mock.calls[0][0];
      expect(calledUrl).toContain("namespace=test-ns");
      expect(calledUrl).toContain("user_id=user-1");
      expect(calledUrl).toContain("session_id=sess-1");
      expect(calledUrl).toContain("limit=500");
      expect(calledUrl).toContain("dry_run=false");
    });

    it("should send policy and pinned_ids in body", async () => {
      mockFetch = createMockFetch({ scanned: 0, deleted: 0, deleted_ids: [], dry_run: true });
      client["fetchFn"] = mockFetch;
      await client.forgetLongTermMemories({
        policy: { max_age_days: 30, budget: 100 },
        pinnedIds: ["keep-1", "keep-2"],
      });
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.policy).toEqual({ max_age_days: 30, budget: 100 });
      expect(callBody.pinned_ids).toEqual(["keep-1", "keep-2"]);
    });
  });

  describe("Summary Views", () => {
    describe("listSummaryViews", () => {
      it("should list summary views", async () => {
        const mockResponse = [{ id: "view-1", name: "Test View", source: "long_term" }];
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.listSummaryViews();
        expect(result).toEqual(mockResponse);
      });

      it("should use GET method", async () => {
        mockFetch = createMockFetch([]);
        client["fetchFn"] = mockFetch;
        await client.listSummaryViews();
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/summary-views"),
          expect.objectContaining({ method: "GET" })
        );
      });
    });

    describe("createSummaryView", () => {
      it("should create summary view", async () => {
        const mockResponse = { id: "view-1", name: "Test View", source: "long_term", group_by: ["user_id"] };
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.createSummaryView({
          name: "Test View",
          source: "long_term",
          group_by: ["user_id"],
        });
        expect(result).toEqual(mockResponse);
      });

      it("should use POST method", async () => {
        mockFetch = createMockFetch({ id: "view-1" });
        client["fetchFn"] = mockFetch;
        await client.createSummaryView({ source: "long_term", group_by: ["user_id"] });
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/summary-views"),
          expect.objectContaining({ method: "POST" })
        );
      });

      it("should send request body", async () => {
        mockFetch = createMockFetch({ id: "view-1" });
        client["fetchFn"] = mockFetch;
        await client.createSummaryView({
          name: "My View",
          source: "working_memory",
          group_by: ["namespace", "user_id"],
          filters: { namespace: { eq: "test" } },
          time_window_days: 7,
          continuous: true,
          prompt: "Custom prompt",
          model_name: "gpt-4o",
        });
        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.name).toBe("My View");
        expect(callBody.source).toBe("working_memory");
        expect(callBody.group_by).toEqual(["namespace", "user_id"]);
        expect(callBody.filters).toEqual({ namespace: { eq: "test" } });
        expect(callBody.time_window_days).toBe(7);
        expect(callBody.continuous).toBe(true);
        expect(callBody.prompt).toBe("Custom prompt");
        expect(callBody.model_name).toBe("gpt-4o");
      });
    });

    describe("getSummaryView", () => {
      it("should get summary view by ID", async () => {
        const mockResponse = { id: "view-1", name: "Test View", source: "long_term" };
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.getSummaryView("view-1");
        expect(result).toEqual(mockResponse);
      });

      it("should return null on 404", async () => {
        client["fetchFn"] = createErrorFetch(404, { detail: "Not found" });
        const result = await client.getSummaryView("nonexistent");
        expect(result).toBeNull();
      });

      it("should encode view ID in URL", async () => {
        mockFetch = createMockFetch({ id: "test" });
        client["fetchFn"] = mockFetch;
        await client.getSummaryView("view/with/slashes");
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("view%2Fwith%2Fslashes"),
          expect.any(Object)
        );
      });

      it("should re-throw non-404 errors", async () => {
        client["fetchFn"] = createErrorFetch(500, { detail: "Server error" });
        await expect(client.getSummaryView("test")).rejects.toThrow(MemoryServerError);
      });
    });

    describe("deleteSummaryView", () => {
      it("should delete summary view", async () => {
        const mockResponse = { status: "ok" };
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.deleteSummaryView("view-1");
        expect(result).toEqual(mockResponse);
      });

      it("should use DELETE method", async () => {
        mockFetch = createMockFetch({ status: "ok" });
        client["fetchFn"] = mockFetch;
        await client.deleteSummaryView("view-1");
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/summary-views/view-1"),
          expect.objectContaining({ method: "DELETE" })
        );
      });
    });

    describe("runSummaryViewPartition", () => {
      it("should run summary view partition", async () => {
        const mockResponse = { view_id: "view-1", group: { user_id: "alice" }, summary: "Summary text", memory_count: 10 };
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.runSummaryViewPartition("view-1", { user_id: "alice" });
        expect(result).toEqual(mockResponse);
      });

      it("should use POST method", async () => {
        mockFetch = createMockFetch({ view_id: "view-1", group: {}, summary: "", memory_count: 0 });
        client["fetchFn"] = mockFetch;
        await client.runSummaryViewPartition("view-1", { user_id: "alice" });
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/summary-views/view-1/partitions/run"),
          expect.objectContaining({ method: "POST" })
        );
      });

      it("should send group in request body", async () => {
        mockFetch = createMockFetch({ view_id: "view-1", group: {}, summary: "", memory_count: 0 });
        client["fetchFn"] = mockFetch;
        await client.runSummaryViewPartition("view-1", { user_id: "alice", namespace: "chat" });
        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.group).toEqual({ user_id: "alice", namespace: "chat" });
      });
    });

    describe("listSummaryViewPartitions", () => {
      it("should list summary view partitions", async () => {
        const mockResponse = [{ view_id: "view-1", group: { user_id: "alice" }, summary: "Summary", memory_count: 5 }];
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.listSummaryViewPartitions("view-1");
        expect(result).toEqual(mockResponse);
      });

      it("should use GET method", async () => {
        mockFetch = createMockFetch([]);
        client["fetchFn"] = mockFetch;
        await client.listSummaryViewPartitions("view-1");
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/summary-views/view-1/partitions"),
          expect.objectContaining({ method: "GET" })
        );
      });

      it("should send filter params", async () => {
        mockFetch = createMockFetch([]);
        client["fetchFn"] = mockFetch;
        await client.listSummaryViewPartitions("view-1", {
          namespace: "test-ns",
          userId: "user-1",
          sessionId: "sess-1",
          memoryType: "semantic",
        });
        const calledUrl = mockFetch.mock.calls[0][0];
        expect(calledUrl).toContain("namespace=test-ns");
        expect(calledUrl).toContain("user_id=user-1");
        expect(calledUrl).toContain("session_id=sess-1");
        expect(calledUrl).toContain("memory_type=semantic");
      });
    });

    describe("runSummaryView", () => {
      it("should run full summary view", async () => {
        const mockResponse = { id: "task-1", type: "summary_view_full_run", status: "pending" };
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.runSummaryView("view-1");
        expect(result).toEqual(mockResponse);
      });

      it("should use POST method", async () => {
        mockFetch = createMockFetch({ id: "task-1", type: "summary_view_full_run", status: "pending" });
        client["fetchFn"] = mockFetch;
        await client.runSummaryView("view-1");
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/summary-views/view-1/run"),
          expect.objectContaining({ method: "POST" })
        );
      });

      it("should send options in request body", async () => {
        mockFetch = createMockFetch({ id: "task-1", type: "summary_view_full_run", status: "pending" });
        client["fetchFn"] = mockFetch;
        await client.runSummaryView("view-1", { force: true });
        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.force).toBe(true);
      });
    });
  });

  describe("Tasks", () => {
    describe("getTask", () => {
      it("should get task by ID", async () => {
        const mockResponse = { id: "task-1", type: "summary_view_full_run", status: "completed" };
        client["fetchFn"] = createMockFetch(mockResponse);
        const result = await client.getTask("task-1");
        expect(result).toEqual(mockResponse);
      });

      it("should use GET method", async () => {
        mockFetch = createMockFetch({ id: "task-1", type: "summary_view_full_run", status: "pending" });
        client["fetchFn"] = mockFetch;
        await client.getTask("task-1");
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("/v1/tasks/task-1"),
          expect.objectContaining({ method: "GET" })
        );
      });

      it("should return null on 404", async () => {
        client["fetchFn"] = createErrorFetch(404, { detail: "Not found" });
        const result = await client.getTask("nonexistent");
        expect(result).toBeNull();
      });

      it("should encode task ID in URL", async () => {
        mockFetch = createMockFetch({ id: "test" });
        client["fetchFn"] = mockFetch;
        await client.getTask("task/with/slashes");
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining("task%2Fwith%2Fslashes"),
          expect.any(Object)
        );
      });

      it("should re-throw non-404 errors", async () => {
        client["fetchFn"] = createErrorFetch(500, { detail: "Server error" });
        await expect(client.getTask("test")).rejects.toThrow(MemoryServerError);
      });
    });
  });
});
