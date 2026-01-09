# Query Optimization

The Redis Agent Memory Server includes intelligent query optimization that uses configurable language models to improve search accuracy and retrieval quality. This feature automatically refines user queries to better match stored memories using specialized AI models.

## Overview

Query optimization transforms natural language searches into more effective queries for semantic search, resulting in better memory retrieval. When enabled, the system uses a separate LLM to analyze and optimize the search query before performing vector similarity search.

**Key Benefits:**
- **Improved search accuracy**: Transforms vague queries into precise search terms
- **Better semantic matching**: Optimizes queries to match how memories are stored
- **Configurable models**: Use different models for optimization vs. generation
- **Automatic fallback**: Gracefully handles optimization failures

## How Query Optimization Works

1. **User Query**: Original search query from user or application
2. **Query Analysis**: Optimization model analyzes the query and available context
3. **Query Refinement**: Model generates an improved search query
4. **Vector Search**: Optimized query is used for semantic similarity search
5. **Result Ranking**: Results are ranked and returned with recency boost if enabled

### Example Transformation

**Before Optimization:**
```
User query: "tell me what I like to eat"
```

**After Optimization:**
```
Optimized query: "user food preferences dietary likes dislikes favorite meals cuisine"
```

This optimization helps find relevant memories even when the original query uses different terminology than the stored memories.

## Configuration

Query optimization is controlled by several settings that can be configured via environment variables.

### Basic Configuration

```bash
# Enable/disable query optimization (default based on interface)
# REST API: disabled by default (optimize_query=false)
# MCP Server: disabled by default (optimize_query=false)

# Models for query optimization (can be different from generation model)
QUERY_OPTIMIZATION_MODEL=gpt-4o-mini           # Model used for query optimization
GENERATION_MODEL=gpt-4o-mini                   # Model used for other AI tasks

# Optimization prompt template (advanced)
QUERY_OPTIMIZATION_PROMPT_TEMPLATE="Optimize this search query for better semantic matching: {query}"
```

### Model Selection

You can use different models for query optimization and other AI tasks:

```bash
# Use a fast, efficient model for query optimization
QUERY_OPTIMIZATION_MODEL=gpt-4o-mini

# Use a more powerful model for memory extraction and other tasks
GENERATION_MODEL=gpt-4o

# Supported models include:
# - gpt-5.2-chat-latest, gpt-5.1-chat-latest, gpt-5-mini, gpt-5-nano (OpenAI GPT-5)
# - gpt-4o, gpt-4o-mini (OpenAI)
# - claude-3-5-sonnet-20241022, claude-3-haiku-20240307 (Anthropic)
# - anthropic.claude-sonnet-4-5-20250929-v1:0 (AWS Bedrock)
# - ollama/llama2 (Ollama)
# - Any model supported by LiteLLM (100+ providers)
```

See [LLM Providers](llm-providers.md) for complete configuration options.

## Usage Examples

### REST API

Query optimization can be controlled per request using the `optimize_query` query parameter:

```bash
# Search with optimization (default: false)
curl -X POST "http://localhost:8000/v1/long-term-memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "what do I like to eat",
    "limit": 5
  }'

# Search without optimization
curl -X POST "http://localhost:8000/v1/long-term-memory/search?optimize_query=false" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "what do I like to eat",
    "limit": 5
  }'

# Explicit optimization enabled
curl -X POST "http://localhost:8000/v1/long-term-memory/search?optimize_query=true" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "what do I like to eat",
    "limit": 5
  }'
```

### MCP Server

Query optimization can be controlled in MCP tool calls:

```python
# Search without optimization (MCP default)
await client.call_tool("search_long_term_memory", {
    "text": "tell me about my preferences"
})

# Search with optimization enabled
await client.call_tool("search_long_term_memory", {
    "text": "tell me about my preferences",
    "optimize_query": True
})

# Memory prompt with optimization
await client.call_tool("memory_prompt", {
    "query": "What are my food preferences?",
    "optimize_query": True
})
```

### Python Client

Using the Agent Memory Client library:

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Search with optimization (explicitly enabled)
results = await client.search_long_term_memory(
    text="what do I like to eat",
    limit=5
)

# Search without optimization
results = await client.search_long_term_memory(
    text="what do I like to eat",
    limit=5,
    optimize_query=False  # Override default
)
```

## Interface Defaults

Different interfaces have different default behaviors for query optimization:

| Interface | Default | Rationale |
|-----------|---------|-----------|
| **REST API** | `optimize_query=False` | Consistent behavior across all interfaces |
| **MCP Server** | `optimize_query=False` | AI agents may prefer direct control over queries |
| **Client Library** | Follows API defaults | Inherits from underlying interface |

## Performance Considerations

### Optimization Overhead

Query optimization adds a small latency overhead due to the additional LLM call:

- **Typical overhead**: 100-500ms depending on model
- **Model choice impact**: Faster models (gpt-4o-mini) vs slower (gpt-4o)
- **Concurrent requests**: Optimization calls are made concurrently with other operations where possible

### When to Use Optimization

**Enable optimization when:**
- Users make natural language queries
- Search terms may not match memory storage terminology
- Search accuracy is more important than latency
- Working with diverse memory content

**Disable optimization when:**
- Making programmatic/structured queries
- Latency is critical
- Query terms already match stored content well
- Making high-frequency searches

## Error Handling

Query optimization includes robust error handling to ensure search reliability:

### Automatic Fallback

If query optimization fails, the system automatically falls back to the original query:

```python
# If optimization fails, original query is used
try:
    optimized_query = await optimize_query(original_query)
except Exception as e:
    logger.warning(f"Query optimization failed: {e}")
    optimized_query = original_query  # Fallback to original
```

### Common Error Scenarios

1. **Model API errors**: Network issues, rate limits, authentication
2. **Prompt template errors**: Invalid template formatting
3. **Response parsing errors**: Unexpected model output format
4. **Timeout errors**: Model response takes too long

All errors result in graceful fallback to the original query, ensuring search functionality remains available.

## Monitoring and Debugging

### Logging

Query optimization activities are logged for monitoring and debugging:

```python
# Enable debug logging
LOG_LEVEL=DEBUG

# Optimization logs include:
# - Original query
# - Optimized query
# - Optimization duration
# - Error details (if any)
```

### Example Log Output

```
DEBUG:agent_memory_server.llm:Optimizing query: "what do I like to eat"
DEBUG:agent_memory_server.llm:Optimized query: "user food preferences dietary likes dislikes favorite meals"
DEBUG:agent_memory_server.llm:Query optimization took 245ms
```

## Advanced Configuration

### Custom Optimization Prompts

You can customize the prompt used for query optimization:

```bash
# Custom optimization prompt template
QUERY_OPTIMIZATION_PROMPT_TEMPLATE='
Analyze this search query and optimize it for better semantic search results.
Focus on expanding synonyms and related concepts.
Original query: {query}
Optimized query:'
```

### Model-Specific Configuration

Different models may require different configurations:

```bash
# For Claude models
QUERY_OPTIMIZATION_MODEL=claude-3-haiku-20240307
ANTHROPIC_API_KEY=your_key_here

# For OpenAI models
QUERY_OPTIMIZATION_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
```

## Best Practices

### Model Selection

1. **Use efficient models**: `gpt-4o-mini` or `claude-3-haiku` for query optimization
2. **Match your budget**: Optimization adds extra token usage
3. **Consider latency**: Faster models for real-time applications

### Optimization Strategy

1. **Test with your data**: Measure improvement on your specific memory content
2. **Monitor performance**: Track search accuracy and latency metrics
3. **A/B testing**: Compare optimized vs non-optimized results
4. **Gradual rollout**: Enable optimization for subset of queries first

### Query Design

1. **Provide context**: Better queries lead to better optimization
2. **Use consistent terminology**: Helps optimization learn patterns
3. **Monitor edge cases**: Very short or very long queries may need special handling

## Troubleshooting

### Common Issues

**Query optimization not working:**
- Check model API keys and configuration
- Verify prompt template syntax
- Review debug logs for errors

**Poor optimization results:**
- Try different optimization models
- Customize the optimization prompt template
- Compare results with optimization disabled

**High latency:**
- Switch to faster optimization models
- Consider disabling for latency-critical applications
- Monitor concurrent request limits

### Debug Steps

1. **Enable debug logging**: `LOG_LEVEL=DEBUG`
2. **Check API connectivity**: Test model APIs directly
3. **Validate configuration**: Verify environment variables
4. **Test fallback behavior**: Ensure search works when optimization fails

## Integration Examples

### FastAPI Application

```python
from agent_memory_server.api import router
from fastapi import FastAPI, Query

app = FastAPI()
app.include_router(router)

@app.post("/custom-search")
async def custom_search(
    query: str,
    optimize: bool = Query(False, alias="optimize_query")
):
    # Custom search with configurable optimization
    results = await search_long_term_memories(
        SearchRequest(text=query),
        optimize_query=optimize
    )
    return results
```

### MCP Client

```python
import asyncio
from mcp.client.session import ClientSession

async def search_with_optimization():
    async with ClientSession() as session:
        # Search with optimization
        result = await session.call_tool(
            "search_long_term_memory",
            {
                "text": "my travel preferences",
                "optimize_query": True,
                "limit": 10
            }
        )
        return result
```

This query optimization feature significantly improves search quality by intelligently refining user queries while maintaining reliable fallback behavior and configurable performance characteristics.
