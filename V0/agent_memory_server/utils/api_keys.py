"""API key management utilities."""

import logging
import os


logger = logging.getLogger(__name__)


def load_api_key(service: str) -> str | None:
    """Load an API key from the environment.

    Args:
        service: The service name, e.g., 'openai', 'anthropic'

    Returns:
        The API key if found, or None
    """
    env_var = f"{service.upper()}_API_KEY"
    api_key = os.environ.get(env_var)

    if not api_key:
        logger.warning(f"No API key found for {service} (${env_var})")
        return None

    return api_key
