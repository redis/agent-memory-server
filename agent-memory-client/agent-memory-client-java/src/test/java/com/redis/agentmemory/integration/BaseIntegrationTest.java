package com.redis.agentmemory.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.redis.agentmemory.MemoryAPIClient;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;

/**
 * Base class for integration tests using Testcontainers.
 * <p>
 * This spins up:
 * 1. Redis container (redis:8)
 * 2. Agent Memory Server container (redislabs/agent-memory-server:latest)
 * <p>
 * Tests extending this class will have access to a real Agent Memory Server
 * backed by a real Redis instance.
 */
@Testcontainers
@Tag("integration")
public abstract class BaseIntegrationTest {

    protected static Network network;
    protected static GenericContainer<?> redisContainer;
    protected static GenericContainer<?> agentMemoryServerContainer;
    protected static MemoryAPIClient client;
    protected static ObjectMapper objectMapper;
    protected static String baseUrl;

    @BeforeAll
    static void setUpContainers() {
        // Create a shared network for containers to communicate
        network = Network.newNetwork();

        // Start Redis container
        redisContainer = new GenericContainer<>(DockerImageName.parse("redis:8"))
                .withNetwork(network)
                .withNetworkAliases("redis")
                .withExposedPorts(6379)
                .waitingFor(Wait.forLogMessage(".*Ready to accept connections.*\\n", 1))
                .withStartupTimeout(Duration.ofMinutes(2));

        redisContainer.start();

        // Get a dummy OpenAI API key from environment or use a placeholder
        String openaiApiKey = System.getenv("OPENAI_API_KEY");
        if (openaiApiKey == null || openaiApiKey.isEmpty()) {
            openaiApiKey = "sk-dummy-key-for-testing-only";
        }

        // Start Agent Memory Server container
        agentMemoryServerContainer = new GenericContainer<>(
                DockerImageName.parse("redislabs/agent-memory-server:latest"))
                .withNetwork(network)
                .withExposedPorts(8000)
                .withEnv("REDIS_URL", "redis://redis:6379")
                .withEnv("OPENAI_API_KEY", openaiApiKey)
                .withEnv("DISABLE_AUTH", "true")  // Disable auth for testing
                .withEnv("LOG_LEVEL", "INFO")
                .waitingFor(Wait.forHttp("/v1/health")
                        .forStatusCode(200)
                        .withStartupTimeout(Duration.ofMinutes(3)))
                .withStartupTimeout(Duration.ofMinutes(3));

        agentMemoryServerContainer.start();

        // Get the mapped port and construct base URL
        Integer mappedPort = agentMemoryServerContainer.getMappedPort(8000);
        baseUrl = String.format("http://%s:%d",
                agentMemoryServerContainer.getHost(),
                mappedPort);

        // Create the client
        client = MemoryAPIClient.builder(baseUrl)
                .timeout(30.0)  // Longer timeout for integration tests
                .build();

        // Create ObjectMapper for test assertions
        objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        objectMapper.disable(com.fasterxml.jackson.databind.SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);

        System.out.println("Integration test environment ready:");
        System.out.println("  Redis: " + redisContainer.getHost() + ":" + redisContainer.getMappedPort(6379));
        System.out.println("  Agent Memory Server: " + baseUrl);
    }

    @AfterAll
    static void tearDownContainers() {
        if (client != null) {
            try {
                client.close();
            } catch (Exception e) {
                System.err.println("Error closing client: " + e.getMessage());
            }
        }

        if (agentMemoryServerContainer != null) {
            agentMemoryServerContainer.stop();
        }

        if (redisContainer != null) {
            redisContainer.stop();
        }

        if (network != null) {
            try {
                network.close();
            } catch (Exception e) {
                System.err.println("Error closing network: " + e.getMessage());
            }
        }
    }

    /**
     * Get the base URL for the Agent Memory Server.
     */
    protected String getBaseUrl() {
        return baseUrl;
    }

    /**
     * Get the MemoryAPIClient instance.
     */
    protected MemoryAPIClient getClient() {
        return client;
    }

    /**
     * Get the ObjectMapper for JSON operations.
     */
    protected ObjectMapper getObjectMapper() {
        return objectMapper;
    }
}

