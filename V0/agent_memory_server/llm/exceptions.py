"""
Custom exceptions for LLM operations.
"""


class LLMClientError(Exception):
    """Base exception for all LLMClient errors."""

    pass


class ModelValidationError(LLMClientError):
    """
    Raised when model validation fails during startup.

    This occurs when:
    - The configured model cannot be resolved
    - The model configuration is invalid for the intended use
    - Required model capabilities are missing
    """

    pass


class APIKeyMissingError(LLMClientError):
    """
    Raised when a required API key is not configured.

    This occurs when:
    - The model's provider requires an API key that is not set
    - Environment variables for the provider are missing
    """

    def __init__(self, provider: str, env_var: str | None = None):
        self.provider = provider
        self.env_var = env_var
        if env_var:
            message = f"{provider} API key is not set. Set the {env_var} environment variable."
        else:
            message = f"{provider} API key is not set."
        super().__init__(message)
