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
from typing import TYPE_CHECKING, Any

from litellm import acompletion, aembedding

from agent_memory_server.llm.exceptions import ModelValidationError
from agent_memory_server.llm.types import (
    ChatCompletionResponse,
    EmbeddingResponse,
)


if TYPE_CHECKING:
    from agent_memory_server.config import ModelConfig, ModelProvider
    from agent_memory_server.llm.embeddings import LiteLLMEmbeddings


logger = logging.getLogger(__name__)


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
            model="gpt-5",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

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
    #   litellm.model_alias_map = {"fast": "gpt-5-mini", "smart": "gpt-5"}
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
            model: Model identifier (e.g., "gpt-5", "claude-3-sonnet")
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
        # LiteLLM automatically detects the provider from the model name.
        # No need for explicit provider prefixes (e.g., "openai/gpt-5").
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
    def create_embeddings(cls) -> LiteLLMEmbeddings:
        """Create an embeddings instance based on configuration.

        This method returns a LiteLLMEmbeddings object that uses LiteLLM
        internally, enabling support for any embedding provider:
        - OpenAI (text-embedding-3-small, text-embedding-3-large)
        - AWS Bedrock (bedrock/amazon.titan-embed-text-v2:0)
        - Ollama (ollama/nomic-embed-text)
        - HuggingFace (huggingface/BAAI/bge-large-en)
        - Cohere (cohere/embed-english-v3.0)
        - And many more...

        Returns:
            A LiteLLMEmbeddings instance.

        Raises:
            ModelValidationError: If Anthropic is configured (no embedding models).

        Example:
            >>> from agent_memory_server.llm import LLMClient
            >>> embeddings = LLMClient.create_embeddings()
        """
        import warnings

        from agent_memory_server.config import ModelProvider, settings
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        model = settings.embedding_model
        embedding_config = settings.embedding_model_config

        # Check for Anthropic - they don't have embedding models
        if embedding_config and embedding_config.provider == ModelProvider.ANTHROPIC:
            raise ModelValidationError(
                "Anthropic does not provide embedding models. "
                "Please configure a different embedding provider (e.g., OpenAI) "
                "by setting EMBEDDING_MODEL to an OpenAI model like 'text-embedding-3-small'."
            )

        # Handle Bedrock models without prefix - add prefix and warn
        if (
            embedding_config
            and embedding_config.provider == ModelProvider.AWS_BEDROCK
            and not model.startswith("bedrock/")
        ):
            warnings.warn(
                f"Bedrock embedding model '{model}' should use 'bedrock/' prefix "
                f"(e.g., 'bedrock/{model}'). Unprefixed Bedrock models are deprecated "
                "and will require the prefix in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            model = f"bedrock/{model}"

        # Get dimensions from config or let LiteLLMEmbeddings auto-detect
        dimensions = None
        if embedding_config:
            dimensions = embedding_config.embedding_dimensions
        elif settings.redisvl_vector_dimensions:
            # Fallback to explicit REDISVL_VECTOR_DIMENSIONS setting
            dimensions = settings.redisvl_vector_dimensions

        # Build kwargs for the embeddings instance
        kwargs: dict = {}

        # Pass API key if available (for OpenAI)
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key

        # Pass API base if configured
        if settings.openai_api_base:
            kwargs["api_base"] = settings.openai_api_base

        return LiteLLMEmbeddings(model=model, dimensions=dimensions, **kwargs)

    @classmethod
    def _map_provider(cls, litellm_provider: str) -> ModelProvider:
        """Map LiteLLM provider string to ModelProvider enum.

        Unknown providers are mapped to OTHER, allowing the server to support
        all 100+ LiteLLM providers without maintaining a hardcoded list.
        """
        from agent_memory_server.config import ModelProvider

        provider_map = {
            "openai": ModelProvider.OPENAI,
            "anthropic": ModelProvider.ANTHROPIC,
            "bedrock": ModelProvider.AWS_BEDROCK,
            "azure": ModelProvider.OPENAI,  # Azure OpenAI uses OpenAI-compatible API
        }
        # Return OTHER for unknown providers (Gemini, Ollama, Cohere, etc.)
        # This allows LiteLLM to handle them natively without our intervention
        return provider_map.get(litellm_provider, ModelProvider.OTHER)

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """
        Get configuration for a model.

        Resolution order:
        1. MODEL_CONFIGS (custom overrides)
        2. LLMClient's internal model database
        3. Fallback to gpt-5-mini defaults with warning

        Args:
            model_name: Name of the model (e.g., "gpt-5", "claude-3-sonnet-20240229")

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

        # Final fallback to gpt-5-mini defaults
        logger.warning(
            f"Model {model_name!r} not found in LLMClient model database or MODEL_CONFIGS. "
            "Using gpt-5-mini defaults."
        )
        return MODEL_CONFIGS.get(
            "gpt-5-mini",
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

            logger.debug(f"Optimized query: {query!r} -> {optimized!r}")
            return optimized

        except Exception as e:
            logger.warning(f"Failed to optimize query {query!r}: {e}")
            # Return original query if optimization fails
            return query


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
