"""
LiteLLM-based embeddings implementation.

This module provides a LangChain-compatible Embeddings class that uses LiteLLM
internally, enabling support for any embedding provider LiteLLM supports.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from litellm import aembedding, embedding, get_model_info


logger = logging.getLogger(__name__)


class LiteLLMEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings using LiteLLM.

    This class implements the LangChain Embeddings interface while using LiteLLM
    internally, enabling support for any embedding provider LiteLLM supports:
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - AWS Bedrock (amazon.titan-embed-text-v2:0, cohere.embed-english-v3)
    - Ollama (ollama/nomic-embed-text, ollama/mxbai-embed-large)
    - HuggingFace (huggingface/BAAI/bge-large-en)
    - Cohere (cohere/embed-english-v3.0)
    - Gemini (gemini/text-embedding-004)
    - Mistral (mistral/mistral-embed)
    - And many more...

    Usage:
        >>> embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")
        >>> vectors = embeddings.embed_documents(["Hello", "World"])
        >>> query_vector = embeddings.embed_query("Hello")

        # With Ollama
        >>> embeddings = LiteLLMEmbeddings(
        ...     model="ollama/nomic-embed-text",
        ...     api_base="http://localhost:11434"
        ... )
    """

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LiteLLM embeddings.

        Args:
            model: Model identifier. Use LiteLLM format:
                - OpenAI: "text-embedding-3-small" or "openai/text-embedding-3-small"
                - Bedrock: "bedrock/amazon.titan-embed-text-v2:0"
                - Ollama: "ollama/nomic-embed-text"
                - HuggingFace: "huggingface/BAAI/bge-large-en"
                - Cohere: "cohere/embed-english-v3.0"
            api_base: Custom API endpoint (for Ollama, proxies, etc.)
            api_key: API key override. If None, uses provider's env var.
            dimensions: Output embedding dimensions. If None, auto-detected.
            **kwargs: Additional provider-specific parameters passed to LiteLLM.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self._dimensions = dimensions
        self._extra_kwargs = kwargs

    @property
    def dimensions(self) -> int | None:
        """Get embedding dimensions, auto-detecting if not set."""
        if self._dimensions is not None:
            return self._dimensions

        # Try to get from LiteLLM model info
        try:
            info = get_model_info(self.model)
            dims = info.get("output_vector_size")
            if dims is not None:
                self._dimensions = dims
                return dims
        except Exception:
            pass

        return None

    def _build_call_kwargs(self, input_texts: list[str]) -> dict[str, Any]:
        """Build kwargs for LiteLLM embedding call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_texts,
        }
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        kwargs.update(self._extra_kwargs)
        return kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        kwargs = self._build_call_kwargs(texts)
        response = embedding(**kwargs)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Async embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        kwargs = self._build_call_kwargs(texts)
        response = await aembedding(**kwargs)
        return [item["embedding"] for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        """
        Async embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        result = await self.aembed_documents([text])
        return result[0]
