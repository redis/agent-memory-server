"""
Type definitions for LLM operations.

This module defines the data structures and protocols used by the LLM client.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatCompletionResponse(BaseModel):
    """Standardized response from chat completion APIs."""

    model_config = ConfigDict(frozen=True)

    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    raw_response: Any = None  # Original response for debugging


class EmbeddingResponse(BaseModel):
    """Standardized response from embedding APIs."""

    model_config = ConfigDict(frozen=True)

    embeddings: list[list[float]]
    total_tokens: int
    model: str
