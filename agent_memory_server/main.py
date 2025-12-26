import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from agent_memory_server import __version__
from agent_memory_server.api import router as memory_router
from agent_memory_server.auth import verify_auth_config
from agent_memory_server.config import ModelProvider, settings
from agent_memory_server.docket_tasks import register_tasks
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.llm import (
    APIKeyMissingError,
    LLMClient,
    ModelValidationError,
)
from agent_memory_server.logging import get_logger
from agent_memory_server.utils.redis import (
    _redis_pool as connection_pool,
    get_redis_conn,
)


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the application on startup"""
    logger.info("Starting Redis Agent Memory Server.")

    # Verify OAuth2/JWT authentication configuration
    try:
        verify_auth_config()
    except Exception:
        logger.exception("Authentication configuration error.")
        raise

    # Validate configured models using LLMClient
    try:
        generation_model_config = LLMClient.get_model_config(settings.generation_model)
        logger.info(
            f"Generation model: {settings.generation_model} "
            f"(provider: {generation_model_config.provider.value}, "
            f"max_tokens: {generation_model_config.max_tokens})"
        )
    except Exception as e:
        err_msg = (
            f"Failed to resolve generation model '{settings.generation_model}': {e}. "
            "Ensure the model name is valid or add it to MODEL_CONFIGS. "
            "Note: We support most models supported by LiteLLM."
        )
        logger.error(err_msg)
        raise ModelValidationError(err_msg) from e

    try:
        embedding_model_config = LLMClient.get_model_config(settings.embedding_model)
        logger.info(
            f"Embedding model: {settings.embedding_model} "
            f"(provider: {embedding_model_config.provider.value}, "
            f"dimensions: {embedding_model_config.embedding_dimensions})"
        )
    except Exception as e:
        err_msg = (
            f"Failed to resolve embedding model '{settings.embedding_model}': {e}. "
            "Ensure the model name is valid or add it to MODEL_CONFIGS."
            "Note: We support most models supported by LiteLLM."
        )
        logger.error(err_msg)
        raise ModelValidationError(err_msg) from e

    # Validate API keys for resolved providers
    for model_config in [generation_model_config, embedding_model_config]:
        match model_config.provider:
            case ModelProvider.OPENAI:
                if not settings.openai_api_key:
                    error = APIKeyMissingError("OpenAI", "OPENAI_API_KEY")
                    logger.error(str(error))
                    raise error
            case ModelProvider.ANTHROPIC:
                if not settings.anthropic_api_key:
                    error = APIKeyMissingError("Anthropic", "ANTHROPIC_API_KEY")
                    logger.error(str(error))
                    raise error
            case ModelProvider.AWS_BEDROCK:
                # The access key ID and secret access key are mandatory.
                # The session token is optional (only for STS).
                has_access_key = (
                    settings.aws_access_key_id and settings.aws_secret_access_key
                )
                if not has_access_key:
                    error = APIKeyMissingError(
                        "AWS Bedrock", "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                    )
                    logger.error(str(error))
                    raise error

    # Set up Redis connection and check working memory migration status
    redis_conn = await get_redis_conn()

    # Check if any working memory keys need migration from string to JSON format
    from agent_memory_server.working_memory import check_and_set_migration_status

    await check_and_set_migration_status(redis_conn)

    # Initialize Docket for background tasks if enabled
    if settings.use_docket:
        logger.info("Attempting to initialize Docket for background tasks.")
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

    logger.info(
        "Redis Agent Memory Server initialized.",
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
