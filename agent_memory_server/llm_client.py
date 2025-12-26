"""
Unified LLM client for all LLM operations.

This module provides a single entry point for all LLM interactions,
abstracting away the underlying provider (OpenAI, Anthropic, Bedrock, etc.).

This abstraction layer allows the internal implementation to change without
affecting any code that uses this class. The backend implementation details
(currently LiteLLM) are hidden from consumers of this API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from litellm import acompletion, aembedding


if TYPE_CHECKING:
    from agent_memory_server.config import ModelConfig, ModelProvider


logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================


class LLMClientError(Exception):
    """Base exception for all LLMClient errors."""

    pass


class ModelValidationError(LLMClientError):
    """
    Raised when model validation fails during startup.

    This occurs when:
    - The configured model cannot be resolved
    - The model configuration is invalid for the intended use
    - Required model capabilities are missing
    """

    pass


class APIKeyMissingError(LLMClientError):
    """
    Raised when a required API key is not configured.

    This occurs when:
    - The model's provider requires an API key that is not set
    - Environment variables for the provider are missing
    """

    def __init__(self, provider: str, env_var: str | None = None):
        self.provider = provider
        self.env_var = env_var
        if env_var:
            message = f"{provider} API key is not set. Set the {env_var} environment variable."
        else:
            message = f"{provider} API key is not set."
        super().__init__(message)


@dataclass(frozen=True)
class ChatCompletionResponse:
    """Standardized response from chat completion APIs."""

    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    raw_response: Any = None  # Original response for debugging


@dataclass(frozen=True)
class EmbeddingResponse:
    """Standardized response from embedding APIs."""

    embeddings: list[list[float]]
    total_tokens: int
    model: str


class LLMBackend(Protocol):
    """Protocol for testing - allows injecting mock backends."""

    async def create_chat_completion(self, **kwargs: Any) -> ChatCompletionResponse: ...

    async def create_embedding(self, **kwargs: Any) -> EmbeddingResponse: ...


class LLMClient:
    """
    Unified LLM client for all LLM operations.

    This is the ONLY class call sites should use. Internal implementation
    can change (LiteLLM today, something else tomorrow) without affecting
    any code that uses this class.

    Supports:
    - Multiple providers (OpenAI, Anthropic, Bedrock, Vertex AI, Azure)
    - Custom/fine-tuned models (via api_base parameter)
    - Local models (Ollama, vLLM, LM Studio via api_base)
    - Provider-specific features (via **kwargs passthrough)

    Usage:
        response = await LLMClient.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

    Testing:
        LLMClient.set_backend(MockBackend())
        # ... run tests ...
        LLMClient.reset()
    """

    _backend: LLMBackend | None = None  # For testing injection

    # -------------------------------------------------------------------------
    # Gateway/Proxy Configuration
    # NOTE: LiteLLM has built-in proxy support. You can configure it globally:
    #   import litellm
    #   litellm.api_base = "https://llm-gateway.company.com/v1"
    #   litellm.api_key = "gateway-api-key"
    #
    # Or per-request via the api_base and api_key parameters in
    # create_chat_completion() and create_embedding().
    #
    # See: https://docs.litellm.ai/docs/simple_proxy
    # -------------------------------------------------------------------------

    # TODO: Model alias support is available in LiteLLM via litellm.model_alias_map
    # Example:
    #   litellm.model_alias_map = {"fast": "gpt-4o-mini", "smart": "gpt-4o"}
    # This could be used for settings.fast_model and settings.generation_model
    # but is not implemented yet.

    @classmethod
    async def create_chat_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """
        Create a chat completion.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-sonnet")
            messages: List of message dicts with 'role' and 'content' keys
            api_base: Custom API endpoint (for Ollama, vLLM, Azure, or gateways)
            api_key: Override API key (for custom endpoints or gateways)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            response_format: Response format (e.g., {"type": "json_object"})
            **kwargs: Provider-specific parameters (tools, tool_choice, etc.)

        Returns:
            ChatCompletionResponse with normalized content and usage
        """
        # Allow test backend injection
        if cls._backend is not None:
            return await cls._backend.create_chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                api_base=api_base,
                api_key=api_key,
                **kwargs,
            )

        # LiteLLM automatically detects the provider from the model name.
        # No need for explicit provider prefixes (e.g., "openai/gpt-4o").
        # See: https://docs.litellm.ai/docs/providers

        # Build kwargs for LiteLLM call
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if api_base is not None:
            call_kwargs["api_base"] = api_base
        if api_key is not None:
            call_kwargs["api_key"] = api_key

        # Merge provider-specific kwargs
        call_kwargs.update(kwargs)

        response = await acompletion(**call_kwargs)

        return ChatCompletionResponse(
            content=response.choices[0].message.content or "",
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=response.model,
            raw_response=response,
        )

    @classmethod
    async def create_embedding(
        cls,
        model: str,
        input_texts: list[str],
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Create embeddings for input texts.

        Args:
            model: Embedding model identifier
            input_texts: List of texts to embed
            api_base: Custom API endpoint (for local or gateway endpoints)
            api_key: Override API key
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingResponse with embedding vectors
        """
        # Allow test backend injection
        if cls._backend is not None:
            return await cls._backend.create_embedding(
                model=model,
                input_texts=input_texts,
                api_base=api_base,
                api_key=api_key,
                **kwargs,
            )

        # LiteLLM automatically detects the provider from the model name.
        # No need for explicit provider prefixes (e.g., "openai/text-embedding-3-small").

        # Build kwargs for LiteLLM call
        call_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_texts,
        }
        if api_base is not None:
            call_kwargs["api_base"] = api_base
        if api_key is not None:
            call_kwargs["api_key"] = api_key

        # Merge provider-specific kwargs
        call_kwargs.update(kwargs)

        response = await aembedding(**call_kwargs)

        embeddings = [item["embedding"] for item in response.data]

        return EmbeddingResponse(
            embeddings=embeddings,
            total_tokens=response.usage.total_tokens,
            model=response.model,
        )

    @classmethod
    def _map_provider(cls, litellm_provider: str) -> ModelProvider:
        """Map LiteLLM provider string to ModelProvider enum."""
        from agent_memory_server.config import ModelProvider

        provider_map = {
            "openai": ModelProvider.OPENAI,
            "anthropic": ModelProvider.ANTHROPIC,
            "bedrock": ModelProvider.AWS_BEDROCK,
            "azure": ModelProvider.OPENAI,  # Azure OpenAI maps to OpenAI
            "vertex_ai": ModelProvider.OPENAI,  # Default fallback
            "gemini": ModelProvider.OPENAI,  # Default fallback
        }
        return provider_map.get(litellm_provider, ModelProvider.OPENAI)

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """
        Get configuration for a model.

        Resolution order:
        1. MODEL_CONFIGS (custom overrides)
        2. LLMClient's internal model database
        3. Fallback to gpt-4o-mini defaults with warning

        Args:
            model_name: Name of the model (e.g., "gpt-4o", "claude-3-sonnet-20240229")

        Returns:
            ModelConfig with provider, name, max_tokens, and embedding_dimensions
        """
        from agent_memory_server.config import MODEL_CONFIGS, ModelConfig

        # First check for custom overrides in MODEL_CONFIGS
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name]

        # Try to resolve model from LLMClient's internal database
        try:
            from litellm import get_model_info

            info = get_model_info(model_name)

            # Extract relevant fields from model metadata
            max_tokens = info.get("max_input_tokens") or info.get("max_tokens") or 4096
            embedding_dims = info.get("output_vector_size") or 1536
            provider_str = info.get("litellm_provider", "openai")

            return ModelConfig(
                provider=cls._map_provider(provider_str),
                name=model_name,
                max_tokens=max_tokens,
                embedding_dimensions=embedding_dims,
            )
        except Exception as e:
            logger.debug(f"Failed to resolve model configuration for {model_name}: {e}")

        # Final fallback to gpt-4o-mini defaults
        logger.warning(
            f"Model '{model_name}' not found in LLMClient model database or MODEL_CONFIGS. "
            "Using gpt-4o-mini defaults."
        )
        return MODEL_CONFIGS.get(
            "gpt-4o-mini",
            ModelConfig(
                provider=cls._map_provider("openai"),
                name=model_name,
                max_tokens=128000,
                embedding_dimensions=1536,
            ),
        )

    @classmethod
    async def optimize_query(
        cls,
        query: str,
        model_name: str | None = None,
    ) -> str:
        """
        Optimize a user query for vector search using a fast model.

        This method takes a natural language query and rewrites it to be more
        effective for semantic similarity search. It uses a fast, small model
        to improve search performance while maintaining query intent.

        Args:
            query: The original user query to optimize
            model_name: Model to use for optimization (defaults to settings.fast_model)

        Returns:
            Optimized query string better suited for vector search
        """
        from agent_memory_server.config import settings

        if not query or not query.strip():
            return query

        # Use fast model from settings if not specified
        effective_model = model_name or settings.fast_model

        # Create optimization prompt from config template
        optimization_prompt = settings.query_optimization_prompt_template.format(
            query=query
        )

        try:
            response = await cls.create_chat_completion(
                model=effective_model,
                messages=[{"role": "user", "content": optimization_prompt}],
            )

            # Get optimized query from response
            optimized = response.content.strip() if response.content else ""

            # Fallback to original if optimization failed
            if not optimized or len(optimized) < settings.min_optimized_query_length:
                logger.warning(f"Query optimization failed for: {query}")
                return query

            logger.debug(f"Optimized query: '{query}' -> '{optimized}'")
            return optimized

        except Exception as e:
            logger.warning(f"Failed to optimize query '{query}': {e}")
            # Return original query if optimization fails
            return query

    @classmethod
    def set_backend(cls, backend: LLMBackend) -> None:
        """Set a custom backend (useful for testing)."""
        cls._backend = backend

    @classmethod
    def reset(cls) -> None:
        """Reset the backend to default (useful for testing)."""
        cls._backend = None
        # TODO: When gateway is enabled, also reset:
        # cls._gateway_base_url = None
        # cls._gateway_api_key = None


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a model.

    This is a convenience function that delegates to LLMClient.get_model_config().
    Prefer using LLMClient.get_model_config() directly in new code.
    """
    return LLMClient.get_model_config(model_name)


async def optimize_query_for_vector_search(
    query: str,
    model_name: str | None = None,
) -> str:
    """
    Optimize a user query for vector search.

    This is a convenience function that delegates to LLMClient.optimize_query().
    Prefer using LLMClient.optimize_query() directly in new code.
    """
    return await LLMClient.optimize_query(query, model_name)
