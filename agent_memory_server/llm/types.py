"""
Type definitions for LLM operations.

This module defines the data structures and protocols used by the LLM client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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
