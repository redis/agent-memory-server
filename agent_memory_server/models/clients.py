"""Client factory for language models."""

import logging
from typing import Any

from agent_memory_server.llms import get_model_client


logger = logging.getLogger(__name__)


async def get_llm_client(model_name: str | None = None) -> Any:
    """Get a language model client for the specified model.

    Args:
        model_name: Name of the model to get a client for, or None for default

    Returns:
        A language model client instance
    """
    if model_name is None:
        model_name = "gpt-4o-mini"  # Default model

    return await get_model_client(model_name)
