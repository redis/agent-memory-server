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
    from langchain_core.embeddings import Embeddings

    from agent_memory_server.config import ModelConfig, ModelProvider


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
            model="gpt-4o",
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
    def create_embeddings(cls) -> Embeddings:
        """Create a LangChain Embeddings instance based on configuration.

        This method returns a LangChain-compatible Embeddings object that can be
        passed to vector stores. It centralizes all embedding provider configuration
        in LLMClient, making it the single entry point for embedding instances.

        The provider is determined from `settings.embedding_model_config`. If not
        configured, defaults to OpenAI.

        Supported providers:
            - OpenAI: Uses langchain_openai.OpenAIEmbeddings
            - Anthropic: Falls back to OpenAI (Anthropic has no embedding models)
            - AWS Bedrock: Uses langchain_aws.BedrockEmbeddings

        Returns:
            A LangChain Embeddings instance compatible with vector stores.

        Raises:
            ImportError: If required provider package is not installed.
            APIKeyMissingError: If required API key is not configured.
            ValueError: If provider is unsupported or Bedrock model doesn't exist.

        Example:
            >>> from agent_memory_server.llm import LLMClient
            >>> embeddings = LLMClient.create_embeddings()
            >>> # Use with a vector store
            >>> from langchain_redis import RedisVectorStore
            >>> vectorstore = RedisVectorStore(embeddings=embeddings, ...)
        """
        from agent_memory_server.config import ModelProvider, settings

        embedding_config = settings.embedding_model_config
        if embedding_config is None:
            raise ModelValidationError(
                f"Unknown embedding model: {settings.embedding_model!r}. "
                "Please configure a supported embedding model in EMBEDDING_MODEL."
            )
        provider = embedding_config.provider

        if provider == ModelProvider.OPENAI:
            return cls._create_openai_embeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )

        if provider == ModelProvider.ANTHROPIC:
            raise ModelValidationError(
                "Anthropic does not provide embedding models. "
                "Please configure a different embedding provider (e.g., OpenAI) "
                "by setting EMBEDDING_MODEL to an OpenAI model like 'text-embedding-3-small'."
            )

        if provider == ModelProvider.AWS_BEDROCK:
            return cls._create_bedrock_embeddings(
                model=settings.embedding_model,
                region=settings.aws_region,
            )

        raise ModelValidationError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: openai, aws-bedrock. "
            f"Note: Anthropic does not provide embedding models."
        )

    @classmethod
    def _create_openai_embeddings(
        cls,
        model: str,
        api_key: str | None = None,
    ) -> Embeddings:
        """Create OpenAI embeddings instance.

        Args:
            model: OpenAI embedding model name (e.g., "text-embedding-3-small")
            api_key: Optional API key. If None, uses OPENAI_API_KEY env var.

        Returns:
            OpenAIEmbeddings instance.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            logger.error(
                "langchain-openai not installed. Install with: pip install langchain-openai"
            )
            raise

        if api_key is not None:
            from pydantic.types import SecretStr

            return OpenAIEmbeddings(model=model, api_key=SecretStr(api_key))

        # Let OpenAIEmbeddings read from OPENAI_API_KEY env var
        return OpenAIEmbeddings(model=model)

    @classmethod
    def _create_bedrock_embeddings(
        cls,
        model: str,
        region: str | None = None,
    ) -> Embeddings:
        """Create AWS Bedrock embeddings instance.

        Args:
            model: Bedrock model ID (e.g., "amazon.titan-embed-text-v2:0")
            region: AWS region. If None, uses default from AWS config.

        Returns:
            BedrockEmbeddings instance.

        Raises:
            ImportError: If langchain-aws or AWS dependencies are not installed.
            ValueError: If the model doesn't exist in the specified region.
        """
        try:
            from langchain_aws import BedrockEmbeddings

            from agent_memory_server._aws.clients import create_bedrock_runtime_client
            from agent_memory_server._aws.utils import bedrock_embedding_model_exists
        except ImportError:
            err_msg = (
                "AWS-related dependencies might be missing. "
                "Try to install with: pip install agent-memory-server[aws]."
            )
            logger.exception(err_msg)
            raise

        if not bedrock_embedding_model_exists(model, region_name=region):
            err_msg = (
                f"Bedrock embedding model {model} not found in region {region}. "
                "Please ensure that the model ID is valid, that the model is "
                "available in the given AWS region, and that your AWS role has "
                "the correct permissions to invoke it."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        bedrock_runtime_client = create_bedrock_runtime_client()
        return BedrockEmbeddings(model_id=model, client=bedrock_runtime_client)

    @classmethod
    def _map_provider(cls, litellm_provider: str) -> ModelProvider:
        """Map LiteLLM provider string to ModelProvider enum."""
        from agent_memory_server.config import ModelProvider

        provider_map = {
            "openai": ModelProvider.OPENAI,
            "anthropic": ModelProvider.ANTHROPIC,
            "bedrock": ModelProvider.AWS_BEDROCK,
            "azure": ModelProvider.OPENAI,  # Azure OpenAI uses OpenAI-compatible API
        }
        if litellm_provider not in provider_map:
            raise ModelValidationError(
                f"Unsupported LiteLLM provider: {litellm_provider!r}. "
                f"Supported providers: {', '.join(provider_map.keys())}"
            )
        return provider_map[litellm_provider]

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
            f"Model {model_name!r} not found in LLMClient model database or MODEL_CONFIGS. "
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
