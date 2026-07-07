package com.redis.agentmemory.models.common;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.List;

/**
 * Filter for tag-style string fields (user_id, session_id, namespace, topics, entities).
 * Match the server-side TagFilter; supports eq, ne, any, all, and startswith operators.
 *
 * <p>
 * Example — match any of several user IDs in one search:
 * <pre>{@code
 * SearchRequest.builder()
 *     .userId(TagFilter.any("user-123", "__account__"))
 *     .build()
 * }</pre>
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TagFilter {

    @Nullable
    private String eq;

    @Nullable
    private String ne;

    @Nullable
    @JsonProperty("any")
    private List<String> any;

    @Nullable
    @JsonProperty("all")
    private List<String> all;

    @Nullable
    private String startswith;

    private TagFilter() {}

    public static TagFilter eq(@NotNull String value) {
        TagFilter f = new TagFilter();
        f.eq = value;
        return f;
    }

    public static TagFilter ne(@NotNull String value) {
        TagFilter f = new TagFilter();
        f.ne = value;
        return f;
    }

    public static TagFilter any(@NotNull List<String> values) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("any cannot be an empty list");
        }
        TagFilter f = new TagFilter();
        f.any = List.copyOf(values);
        return f;
    }

    public static TagFilter any(@NotNull String... values) {
        return any(Arrays.asList(values));
    }

    public static TagFilter all(@NotNull List<String> values) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("all cannot be an empty list");
        }
        TagFilter f = new TagFilter();
        f.all = List.copyOf(values);
        return f;
    }

    public static TagFilter all(@NotNull String... values) {
        return all(Arrays.asList(values));
    }

    public static TagFilter startsWith(@NotNull String prefix) {
        if (prefix.isEmpty()) {
            throw new IllegalArgumentException("startswith cannot be an empty string");
        }
        TagFilter f = new TagFilter();
        f.startswith = prefix;
        return f;
    }

    @Nullable
    public String getEq() { return eq; }

    @Nullable
    public String getNe() { return ne; }

    @Nullable
    public List<String> getAny() { return any; }

    @Nullable
    public List<String> getAll() { return all; }

    @Nullable
    public String getStartswith() { return startswith; }
}
