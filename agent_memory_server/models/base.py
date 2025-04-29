"""Base client for language models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseClient(ABC):
    """Base class for language model clients."""

    @abstractmethod
    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> Any:
        """Create a chat completion using the model."""
        pass

    @abstractmethod
    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """Create embeddings for a list of text strings."""
        pass
