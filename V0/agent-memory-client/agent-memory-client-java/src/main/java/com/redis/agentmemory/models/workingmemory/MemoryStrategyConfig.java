package com.redis.agentmemory.models.workingmemory;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.HashMap;
import java.util.Map;

/**
 * Configuration for memory extraction strategy.
 */
public class MemoryStrategyConfig {

    @NotNull
    private String strategy;

    @NotNull
    private Map<String, Object> config;

    public MemoryStrategyConfig() {
        this.strategy = "discrete";
        this.config = new HashMap<>();
    }

    public MemoryStrategyConfig(@NotNull String strategy) {
        this.strategy = strategy;
        this.config = new HashMap<>();
    }

    public MemoryStrategyConfig(@NotNull String strategy, @NotNull Map<String, Object> config) {
        this.strategy = strategy;
        this.config = config;
    }

    @NotNull
    public String getStrategy() {
        return strategy;
    }

    public void setStrategy(@NotNull String strategy) {
        this.strategy = strategy;
    }

    @NotNull
    public Map<String, Object> getConfig() {
        return config;
    }

    public void setConfig(@NotNull Map<String, Object> config) {
        this.config = config;
    }

    @Override
    public String toString() {
        return "MemoryStrategyConfig{" +
                "strategy='" + strategy + '\'' +
                ", config=" + config +
                '}';
    }

    /**
     * Creates a new builder for MemoryStrategyConfig.
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for MemoryStrategyConfig.
     */
    public static class Builder {
        private String strategy;
        private Map<String, Object> config;

        private Builder() {
            // Initialize with defaults
            this.strategy = "discrete";
            this.config = new HashMap<>();
        }

        /**
         * Sets the strategy name.
         * @param strategy the strategy name (e.g., "discrete", "continuous")
         * @return this builder
         */
        public Builder strategy(@NotNull String strategy) {
            this.strategy = strategy;
            return this;
        }

        /**
         * Sets the configuration map.
         * @param config the configuration map
         * @return this builder
         */
        public Builder config(@NotNull Map<String, Object> config) {
            this.config = config;
            return this;
        }

        /**
         * Adds a single configuration entry.
         * @param key the configuration key
         * @param value the configuration value
         * @return this builder
         */
        public Builder addConfig(@NotNull String key, @Nullable Object value) {
            this.config.put(key, value);
            return this;
        }

        /**
         * Builds the MemoryStrategyConfig instance.
         * @return a new MemoryStrategyConfig
         */
        public MemoryStrategyConfig build() {
            MemoryStrategyConfig strategyConfig = new MemoryStrategyConfig();
            strategyConfig.strategy = this.strategy;
            strategyConfig.config = this.config;
            return strategyConfig;
        }
    }
}
