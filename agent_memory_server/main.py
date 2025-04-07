import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from agent_memory_server import utils
from agent_memory_server.api import router as memory_router
from agent_memory_server.config import settings
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.llms import MODEL_CONFIGS, ModelProvider
from agent_memory_server.logging import configure_logging, get_logger
from agent_memory_server.utils import ensure_redisearch_index, get_redis_conn


configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the application on startup"""
    logger.info("Starting Redis Agent Memory Server 🤘")

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
        "Redis Agent Memory Server initialized",
        window_size=settings.window_size,
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        long_term_memory=settings.long_term_memory,
    )

    yield

    logger.info("Shutting down Redis Agent Memory Server")
    if utils._redis_pool:
        await utils._redis_pool.aclose()


# Create FastAPI app
app = FastAPI(title="Redis Agent Memory Server", lifespan=lifespan)


app.include_router(health_router)
app.include_router(memory_router)


def on_start_logger(port: int):
    """Log startup information"""
    print("\n-----------------------------------")
    print(f"🧠 Redis Agent Memory Server running on port: {port}")
    print("-----------------------------------\n")


# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    on_start_logger(port)
    uvicorn.run("agent_memory_server.main:app", host="0.0.0.0", port=port, reload=False)
