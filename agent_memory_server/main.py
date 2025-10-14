import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from agent_memory_server import __version__
from agent_memory_server.api import router as memory_router
from agent_memory_server.auth import verify_auth_config
from agent_memory_server.config import MODEL_CONFIGS, ModelProvider, settings
from agent_memory_server.docket_tasks import register_tasks
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.logging import get_logger
from agent_memory_server.utils.redis import (
    _redis_pool as connection_pool,
    get_redis_conn,
)


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the application on startup"""
    logger.info("Starting Redis Agent Memory Server 🤘")

    # Verify OAuth2/JWT authentication configuration
    try:
        verify_auth_config()
    except Exception as e:
        logger.error(f"Authentication configuration error: {e}")
        raise

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

    # Set up Redis connection if long-term memory is enabled
    if settings.long_term_memory:
        await get_redis_conn()

    # Initialize Docket for background tasks if enabled
    if settings.use_docket:
        try:
            await register_tasks()
            logger.info("Initialized Docket for background tasks")
            logger.info("To run the worker, use one of these methods:")
            logger.info(
                "1. CLI: docket worker --tasks agent_memory_server.docket_tasks:task_collection"
            )
            logger.info("2. Python: python -m agent_memory_server.worker")
        except Exception as e:
            logger.error(f"Failed to initialize Docket: {e}")
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
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        long_term_memory=settings.long_term_memory,
    )

    yield

    logger.info("Shutting down Redis Agent Memory Server")
    if connection_pool is not None:
        await connection_pool.aclose()


# Create FastAPI app
app = FastAPI(
    title="Redis Agent Memory Server",
    lifespan=lifespan,
    version=__version__,
)


app.include_router(health_router)
app.include_router(memory_router)


def on_start_logger(port: int):
    """Log startup information"""
    print("\n-----------------------------------")
    print(f"🧠 Redis Agent Memory Server running on port: {port}")
    print("-----------------------------------\n")


# Run the application
if __name__ == "__main__":
    # Parse command line arguments for port
    port = settings.port

    # Check if --port argument is provided
    if "--port" in sys.argv:
        try:
            port_index = sys.argv.index("--port") + 1
            if port_index < len(sys.argv):
                port = int(sys.argv[port_index])
                print(f"Using port from command line: {port}")
        except (ValueError, IndexError):
            # If conversion fails or index out of bounds, use default
            print(f"Invalid port argument, using default: {port}")
    else:
        print(f"No port argument provided, using default: {port}")

    # Explicitly unset the PORT environment variable if it exists
    if "PORT" in os.environ:
        port_val = os.environ.pop("PORT")
        print(f"Removed environment variable PORT={port_val}")

    on_start_logger(port)
    uvicorn.run(
        app,  # Using the app instance directly
        host="0.0.0.0",
        port=port,
        reload=False,
    )
