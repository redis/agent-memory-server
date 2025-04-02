import os

import uvicorn
from fastapi import BackgroundTasks, FastAPI

from redis_memory_server import utils
from redis_memory_server.api import router as memory_router
from redis_memory_server.config import settings
from redis_memory_server.healthcheck import router as health_router
from redis_memory_server.llms import MODEL_CONFIGS, ModelProvider
from redis_memory_server.logging import configure_logging, get_logger
from redis_memory_server.mcp import mcp_app
from redis_memory_server.utils import ensure_redisearch_index, get_redis_conn


configure_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="Redis Agentic Memory Server")


async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Redis Agentic Memory Server 🤘")

    # Check for required API keys
    available_providers = []

    if settings.openai_api_key:
        available_providers.append(ModelProvider.OPENAI)
    else:
        logger.warning("OpenAI API key not set, OpenAI models will not be available")

    if settings.anthropic_api_key:
        available_providers.append(ModelProvider.ANTHROPIC)
    else:
        logger.warning(
            "Anthropic API key not set, Anthropic models will not be available"
        )

    # Check if the configured models are available
    generation_model_config = MODEL_CONFIGS.get(settings.generation_model)
    embedding_model_config = MODEL_CONFIGS.get(settings.embedding_model)

    if (
        generation_model_config
        and generation_model_config.provider not in available_providers
    ):
        logger.warning(
            f"Selected generation model {settings.generation_model} requires {generation_model_config.provider} API key"
        )

    if (
        embedding_model_config
        and embedding_model_config.provider not in available_providers
    ):
        logger.warning(
            f"Selected embedding model {settings.embedding_model} requires {embedding_model_config.provider} API key"
        )

    # If long-term memory is enabled but OpenAI isn't available, warn user
    if settings.long_term_memory and ModelProvider.OPENAI not in available_providers:
        logger.warning(
            "Long-term memory requires OpenAI for embeddings, but OpenAI API key is not set"
        )

    # Set up RediSearch index if long-term memory is enabled
    if settings.long_term_memory:
        redis = get_redis_conn()

        # Get embedding dimensions from model config
        embedding_model_config = MODEL_CONFIGS.get(settings.embedding_model)
        vector_dimensions = (
            embedding_model_config.embedding_dimensions
            if embedding_model_config
            else 1536
        )
        distance_metric = "COSINE"

        try:
            await ensure_redisearch_index(redis, vector_dimensions, distance_metric)
        except Exception as e:
            logger.error(f"Failed to ensure RediSearch index: {e}")
            raise

    # Show available models
    openai_models = [
        model
        for model, config in MODEL_CONFIGS.items()
        if config.provider == ModelProvider.OPENAI
        and ModelProvider.OPENAI in available_providers
    ]
    anthropic_models = [
        model
        for model, config in MODEL_CONFIGS.items()
        if config.provider == ModelProvider.ANTHROPIC
        and ModelProvider.ANTHROPIC in available_providers
    ]

    if openai_models:
        logger.info(f"Available OpenAI models: {', '.join(openai_models)}")
    if anthropic_models:
        logger.info(f"Available Anthropic models: {', '.join(anthropic_models)}")

    logger.info(
        "Redis Agentic Memory Server initialized",
        window_size=settings.window_size,
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        long_term_memory=settings.long_term_memory,
    )


async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Redis Agentic Memory Server")
    if utils._redis_pool:
        await utils._redis_pool.aclose()


app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

app.include_router(health_router)
app.include_router(memory_router)


# Set up MCP routes
@app.middleware("http")
async def mcp_middleware(request, call_next):
    """Middleware to inject BackgroundTasks into MCP handler"""
    background_tasks = BackgroundTasks()
    request.state.background_tasks = background_tasks
    return await call_next(request)


# Mount MCP server
app.mount("/mcp", mcp_app.sse_app())


def on_start_logger(port: int):
    """Log startup information"""
    print("\n-----------------------------------")
    print(f"🧠 Redis Agentic Memory Server running on port: {port}")
    print("-----------------------------------\n")


# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    on_start_logger(port)
    uvicorn.run("redis_memory_server.main:app", host="0.0.0.0", port=port, reload=False)
