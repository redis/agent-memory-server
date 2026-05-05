"""
LiteLLM-based embeddings implementation.

This module provides a standalone Embeddings class that uses LiteLLM
internally, enabling support for any embedding provider LiteLLM supports.
"""

from __future__ import annotations

import logging
from typing import Any

from litellm import aembedding, embedding
from litellm.exceptions import NotFoundError as LiteLLMNotFoundError
from litellm.utils import get_model_info


logger = logging.getLogger(__name__)


# Defensive char-level truncation cap. Embedding providers have hard token
# caps (e.g. Ollama nomic-embed-text caps at 2048 tokens architecturally:
# `nomic-bert.context_length: 2048`, even with `options.num_ctx` set higher).
# Token/char ratio for English prose is roughly 0.25-0.40 tokens/char, so
# 6000 chars ~ 1500-2400 tokens — slightly conservative buffer under 2048.
# Primary protection is the pre-merge size gate in deduplicate_by_semantic_search;
# this is a safety net catching any path that slips through with oversized text.
EMBEDDING_TRUNCATE_CHARS = 6000


class LiteLLMEmbeddings:
    """
    Embeddings using LiteLLM.

    This class provides a standard embeddings interface while using LiteLLM
    internally, enabling support for any embedding provider LiteLLM supports:
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - AWS Bedrock (amazon.titan-embed-text-v2:0, cohere.embed-english-v3)
    - Ollama (ollama/nomic-embed-text, ollama/mxbai-embed-large)
    - HuggingFace (huggingface/BAAI/bge-large-en)
    - Cohere (cohere/embed-english-v3.0)
    - Vertex AI (vertex_ai/text-embedding-004, text-embedding-005)
    - Mistral (mistral/mistral-embed)
    - And many more...

    Note: Google's embedding models are available via Vertex AI, not the Gemini API.
    Use `vertex_ai/text-embedding-004` or just `text-embedding-004` (no prefix).

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
        self._extra_kwargs = kwargs

        # Detect dimensions at construction time
        self._dimensions = dimensions or self._detect_dimensions()

        if self._dimensions is None:
            logger.warning(
                "Could not determine embedding dimensions for model %r. "
                "Databases requiring explicit dimensions may fail. "
                "Consider specifying dimensions explicitly.",
                self.model,
            )

    def _detect_dimensions(self) -> int | None:
        """
        Detect embedding dimensions using fallback chain.

        Order of precedence:
        1. Our MODEL_CONFIGS (most reliable, we control it)
        2. LiteLLM's model registry (fallback for unknown models)

        Returns:
            Detected dimensions or None if unknown.
        """
        # Import here to avoid circular dependency
        from agent_memory_server.config import MODEL_CONFIGS

        # 1. Check our known config first (most reliable)
        if self.model in MODEL_CONFIGS:
            dims = MODEL_CONFIGS[self.model].embedding_dimensions
            logger.debug(
                "Detected dimensions=%d for model %r from MODEL_CONFIGS",
                dims,
                self.model,
            )
            return dims

        # 2. Try LiteLLM's registry as fallback
        try:
            info = get_model_info(self.model)
            dims = info.get("output_vector_size")
            if dims is not None:
                logger.debug(
                    "Detected dimensions=%d for model %r from LiteLLM registry",
                    dims,
                    self.model,
                )
                return dims
        except LiteLLMNotFoundError:
            logger.debug(
                "Model %r not found in LiteLLM registry",
                self.model,
            )

        return None

    @property
    def dimensions(self) -> int | None:
        """Get embedding dimensions."""
        return self._dimensions

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

    def _truncate_for_embedding(self, texts: list[str]) -> list[str]:
        """
        Truncate texts that exceed the safety cap.

        Defense-in-depth: the primary protection against oversized texts is
        the pre-merge size gate in deduplicate_by_semantic_search. This is
        a safety net for any path that slips through (e.g. an oversized
        memory ingested directly via index_long_term_memories without going
        through compaction). Without this, Ollama nomic-embed-text returns
        HTTP 400 "input length exceeds context length" for texts > ~2048
        tokens, which (with the v6 index-before-delete reorder) aborts the
        merge and preserves originals — but ALSO blocks the legitimate
        index of an oversized memory. Truncation lets that ingest succeed
        with degraded recall on the tail.

        Logs every truncation with logger.warning so operators can detect
        if this safety net is firing in production.
        """
        result: list[str] = []
        for i, text in enumerate(texts):
            if isinstance(text, str) and len(text) > EMBEDDING_TRUNCATE_CHARS:
                logger.warning(
                    "Truncating text at index %d for embedding: model=%s "
                    "original_len=%d truncated_len=%d (cap=%d). This safety "
                    "net should rarely fire if pre-merge size gates are in "
                    "place; investigate the calling path if frequent.",
                    i,
                    self.model,
                    len(text),
                    EMBEDDING_TRUNCATE_CHARS,
                    EMBEDDING_TRUNCATE_CHARS,
                )
                result.append(text[:EMBEDDING_TRUNCATE_CHARS])
            else:
                result.append(text)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If any text in the list is empty (OpenAI rejects empty strings).
        """
        if not texts:
            return []

        # Validate that no texts are empty - OpenAI rejects empty strings with
        # "'$.input' is invalid" error
        for i, text in enumerate(texts):
            if not text:
                raise ValueError(
                    f"Cannot embed empty string at index {i}. "
                    "OpenAI's embedding API rejects empty strings."
                )

        # Safety-net truncation BEFORE provider call so oversized texts
        # don't crash the embedding (model token caps).
        texts = self._truncate_for_embedding(texts)

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

        Raises:
            ValueError: If any text in the list is empty (OpenAI rejects empty strings).
        """
        if not texts:
            return []

        # Validate that no texts are empty - OpenAI rejects empty strings with
        # "'$.input' is invalid" error
        for i, text in enumerate(texts):
            if not text:
                raise ValueError(
                    f"Cannot embed empty string at index {i}. "
                    "OpenAI's embedding API rejects empty strings."
                )

        # Safety-net truncation BEFORE provider call so oversized texts
        # don't crash the embedding (model token caps).
        texts = self._truncate_for_embedding(texts)

        kwargs = self._build_call_kwargs(texts)
        try:
            response = await aembedding(**kwargs)
        except Exception as e:
            # Capture input shape on embedding failures so we can diagnose
            # provider-specific rejections (e.g. Ollama's 400 "invalid input
            # type" on non-string list elements). Logs types, lengths, and a
            # truncated sample without leaking full memory content.
            sample = texts[0] if texts else None
            sample_repr = repr(sample)
            if len(sample_repr) > 200:
                sample_repr = sample_repr[:200] + "...<truncated>"
            logger.exception(
                "Embedding call failed: model=%s n_texts=%d types=%s "
                "lengths=%s sample[0]=%s err=%s",
                kwargs.get("model"),
                len(texts),
                [type(t).__name__ for t in texts[:5]],
                [len(t) if isinstance(t, str) else None for t in texts[:5]],
                sample_repr,
                e,
            )
            raise
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
