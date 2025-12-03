import json
import logging
import os
from typing import Any

import anthropic
import numpy as np
from openai import AsyncOpenAI

from agent_memory_server.config import (
    MODEL_CONFIGS,
    ModelConfig,
    ModelProvider,
    settings,
)


logger = logging.getLogger(__name__)


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Default to GPT-4o-mini if model not found
    logger.warning(f"Model {model_name} not found in configuration, using gpt-4o-mini")
    return MODEL_CONFIGS["gpt-4o-mini"]


class ChatResponse:
    """Unified wrapper for chat responses from different providers"""

    def __init__(self, choices: list[Any], usage: dict[str, int]):
        self.choices = choices or []
        self.usage = usage or {"total_tokens": 0}

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class AnthropicClientWrapper:
    """Wrapper for Anthropic client"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the Anthropic client"""
        anthropic_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        anthropic_api_base = base_url or os.environ.get("ANTHROPIC_API_BASE")

        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required")

        if anthropic_api_base:
            self.client = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
                base_url=anthropic_api_base,
            )
        else:
            self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Create a chat completion using the Anthropic API"""
        try:
            # For Anthropic, we need to handle structured output differently
            if response_format and response_format.get("type") == "json_object":
                prompt = f"{prompt}\n\nYou must respond with a valid JSON object."

            if functions and function_call:
                # Add function schema to prompt
                schema = functions[0]["parameters"]
                prompt = f"{prompt}\n\nYou must respond with a JSON object matching this schema:\n{json.dumps(schema, indent=2)}"

            response = await self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            # Convert to a unified format - safely extract content
            content = ""
            if (
                hasattr(response, "content")
                and response.content
                and len(response.content) > 0
                and hasattr(response.content[0], "text")
            ):
                content = response.content[0].text

            choices = [{"message": {"content": content}}]

            # Handle both object and dictionary usage formats from API responses
            input_tokens = output_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    input_tokens = response.usage.get("input_tokens", 0)
                    output_tokens = response.usage.get("output_tokens", 0)
                else:
                    input_tokens = getattr(response.usage, "input_tokens", 0)
                    output_tokens = getattr(response.usage, "output_tokens", 0)

            usage = {"total_tokens": input_tokens + output_tokens}

            return ChatResponse(choices=choices, usage=usage)
        except Exception as e:
            logger.error(f"Error creating chat completion with Anthropic: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """
        Create embeddings for the given texts
        Note: Anthropic doesn't offer an embedding API, so we'll use OpenAI's
        embeddings or raise an error if needed
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Please use OpenAI for embeddings."
        )


class BedrockClientWrapper:
    """Wrapper for AWS Bedrock client for LLM generation using ChatBedrockConverse."""

    def __init__(
        self,
        region_name: str | None = None,
        credentials: dict[str, str] | None = None,
    ):
        """Initialize the Bedrock client.

        Args:
            region_name (str | None): AWS region name. If not provided, it will be picked up from the environment.
            credentials (dict[str, str] | None): AWS credentials. If not provided, it will be picked up from the environment.
        """
        self.region_name = region_name or settings.region_name
        self.credentials = credentials or settings.aws_credentials
        # Cache for ChatBedrockConverse instances per model
        self._chat_models: dict = {}

    def _get_chat_model(self, model_id: str):
        """Get or create a ChatBedrockConverse instance for a model.

        Args:
            model_id: The Bedrock model ID

        Returns:
            A ChatBedrockConverse instance
        """
        if model_id in self._chat_models:
            return self._chat_models[model_id]

        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as err:
            raise ImportError(
                "Missing AWS-related dependencies. "
                "Try to install with: pip install agent-memory-server[aws]."
            ) from err

        chat_model = ChatBedrockConverse(
            model=model_id,
            region_name=self.region_name,
            **self.credentials,
        )
        self._chat_models[model_id] = chat_model
        return chat_model

    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Create a chat completion using AWS Bedrock via ChatBedrockConverse.

        Args:
            model (str): The Bedrock model ID to use
            prompt (str): The prompt to send to the model
            response_format (dict[str, str] | None): Optional response format (e.g., {"type": "json_object"})
            functions (list[dict[str, Any]] | None): Optional function definitions (handled via prompt injection)
            function_call (dict[str, str] | None): Optional function call specification

        Returns:
            ChatResponse with the model's response
        """
        from langchain_core.messages import HumanMessage

        try:
            chat_model = self._get_chat_model(model)

            # Handle JSON response format by adding instruction to prompt
            effective_prompt = prompt
            if response_format and response_format.get("type") == "json_object":
                effective_prompt = (
                    f"{prompt}\n\nYou must respond with a valid JSON object."
                )

            if functions and function_call:
                # Add function schema to prompt for structured output
                schema = functions[0]["parameters"]
                effective_prompt = f"{effective_prompt}\n\nYou must respond with a JSON object matching this schema:\n{json.dumps(schema, indent=2)}"

            # Use ainvoke for async operation
            message = HumanMessage(content=effective_prompt)
            response = await chat_model.ainvoke([message])

            # Extract content from the response
            content = response.content if hasattr(response, "content") else ""

            # Extract token usage from response metadata
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = response.usage_metadata.get("input_tokens", 0)
                output_tokens = response.usage_metadata.get("output_tokens", 0)

            choices = [{"message": {"content": content}}]
            usage = {"total_tokens": input_tokens + output_tokens}

            return ChatResponse(choices=choices, usage=usage)

        except Exception:
            logger.exception("Error creating chat completion with Bedrock.")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """
        Create embeddings for the given texts.
        Note: For embeddings, use BedrockEmbeddings from langchain-aws instead.
        """
        raise NotImplementedError(
            "For Bedrock embeddings, configure EMBEDDING_MODEL with a Bedrock "
            "embedding model ID (e.g., amazon.titan-embed-text-v2:0). "
            "See documentation for details."
        )


class OpenAIClientWrapper:
    """Wrapper for OpenAI client"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the OpenAI client based on environment variables"""

        # Regular OpenAI setup
        openai_api_base = base_url or os.environ.get("OPENAI_API_BASE")
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required")

        if openai_api_base:
            self.completion_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            self.embedding_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            self.completion_client = AsyncOpenAI(api_key=openai_api_key)
            self.embedding_client = AsyncOpenAI(api_key=openai_api_key)

    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Create a chat completion using the OpenAI API"""
        try:
            # Build the request parameters
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add optional parameters if provided
            if response_format:
                request_params["response_format"] = response_format
            if functions:
                request_params["functions"] = functions
            if function_call:
                request_params["function_call"] = function_call

            response = await self.completion_client.chat.completions.create(
                **request_params
            )

            # Convert to unified format
            # Handle both object and dictionary usage formats from API responses
            total_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    total_tokens = response.usage.get("total_tokens", 0)
                else:
                    total_tokens = getattr(response.usage, "total_tokens", 0)

            return ChatResponse(
                choices=response.choices,
                usage={"total_tokens": total_tokens},
            )
        except Exception as e:
            logger.error(f"Error creating chat completion with OpenAI: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """Create embeddings for the given texts"""
        try:
            embeddings = []
            embedding_model = "text-embedding-ada-002"

            # Process in batches of 20 to avoid rate limits
            batch_size = 20
            for i in range(0, len(query_vec), batch_size):
                batch = query_vec[i : i + batch_size]
                response = await self.embedding_client.embeddings.create(
                    model=embedding_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise


# Global LLM client cache
_model_clients = {}


# TODO: This should take a provider as input, not model name, and cache on provider
async def get_model_client(
    model_name: str,
) -> OpenAIClientWrapper | AnthropicClientWrapper | BedrockClientWrapper:
    """Get the appropriate client for a model using the factory.

    This is a module-level function that caches clients for reuse.

    Args:
        model_name: Name of the model to get a client for

    Returns:
        An appropriate client wrapper for the model
    """
    global _model_clients
    model = None

    if model_name not in _model_clients:
        model_config = get_model_config(model_name)

        if model_config.provider == ModelProvider.OPENAI:
            model = OpenAIClientWrapper(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
            )
        elif model_config.provider == ModelProvider.ANTHROPIC:
            model = AnthropicClientWrapper(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_api_base,
            )
        elif model_config.provider == ModelProvider.AWS_BEDROCK:
            model = BedrockClientWrapper(
                region_name=settings.region_name,
                credentials=settings.aws_credentials,
            )

        if model:
            _model_clients[model_name] = model
            return model

        raise ValueError(f"Unsupported model provider: {model_config.provider}")

    return _model_clients[model_name]


async def optimize_query_for_vector_search(
    query: str,
    model_name: str | None = None,
) -> str:
    """
    Optimize a user query for vector search using a fast model.

    This function takes a natural language query and rewrites it to be more effective
    for semantic similarity search. It uses a fast, small model to improve search
    performance while maintaining query intent.

    Args:
        query: The original user query to optimize
        model_name: Model to use for optimization (defaults to settings.fast_model)

    Returns:
        Optimized query string better suited for vector search
    """
    if not query or not query.strip():
        return query

    # Use fast model from settings if not specified
    effective_model = model_name or settings.fast_model

    # Create optimization prompt from config template
    optimization_prompt = settings.query_optimization_prompt_template.format(
        query=query
    )

    try:
        client = await get_model_client(effective_model)

        response = await client.create_chat_completion(
            model=effective_model,
            prompt=optimization_prompt,
        )

        if (
            hasattr(response, "choices")
            and response.choices
            and len(response.choices) > 0
        ):
            optimized = ""
            if hasattr(response.choices[0], "message"):
                optimized = response.choices[0].message.content
            elif hasattr(response.choices[0], "text"):
                optimized = response.choices[0].text
            else:
                optimized = str(response.choices[0])

            # Clean up the response
            optimized = optimized.strip()

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

    return query
