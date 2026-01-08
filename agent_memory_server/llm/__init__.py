"""
LLM client package for unified LLM operations.

This package provides a single entry point for all LLM interactions,
abstracting away the underlying provider (OpenAI, Anthropic, Bedrock, etc.).

Usage:
    from agent_memory_server.llm import LLMClient, ChatCompletionResponse

    response = await LLMClient.create_chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from agent_memory_server.llm.client import (
    LLMClient,
    get_model_config,
    optimize_query_for_vector_search,
)
from agent_memory_server.llm.exceptions import (
    APIKeyMissingError,
    LLMClientError,
    ModelValidationError,
)
from agent_memory_server.llm.types import (
    ChatCompletionResponse,
    EmbeddingResponse,
)


__all__ = [
    # Client
    "LLMClient",
    # Convenience functions
    "get_model_config",
    "optimize_query_for_vector_search",
    # Exceptions
    "LLMClientError",
    "ModelValidationError",
    "APIKeyMissingError",
    # Types
    "ChatCompletionResponse",
    "EmbeddingResponse",
]
