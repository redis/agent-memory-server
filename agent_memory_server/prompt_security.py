"""
Security utilities for prompt validation and sanitization.

Provides defenses against prompt injection, template injection, and other
adversarial attacks when using user-provided prompts with LLMs.
"""

import re
import string


class PromptSecurityError(Exception):
    """Raised when a security issue is detected in a prompt."""

    pass


class PromptValidator:
    """Validates and sanitizes user-provided prompts for security."""

    # Dangerous patterns that could indicate prompt injection
    DANGEROUS_PATTERNS = [
        # Direct instruction overrides
        r"ignore\s+(previous|all|above)\s+instructions?",
        r"forget\s+(everything|all|previous)",
        r"new\s+instructions?:",
        r"system\s*[:=]\s*",
        r"override\s+(system|instructions?)",
        # Jailbreaking attempts
        r"act\s+as\s+(?:dan|developer\s+mode|unrestricted)",
        r"pretend\s+(?:you\s+are|to\s+be)",
        r"roleplay\s+as",
        r"simulate\s+(?:a|being)",
        # Information extraction attempts
        r"reveal\s+(?:your|the)\s+(?:system|instructions?|prompt)",
        r"show\s+me\s+(?:your|the)\s+(?:system|instructions?|prompt)",
        r"what\s+(?:are\s+)?your\s+instructions?",
        r"print\s+(?:your|the)\s+(?:system|instructions?|prompt)",
        # Code execution attempts
        r"execute\s+(?:code|command|script)",
        r"run\s+(?:code|command|script)",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
        # Template injection patterns
        r"\{[^}]*__[^}]*\}",  # Dunder methods in templates
        r"\{[^}]*\.__[^}]*\}",  # Attribute access to dunder methods
        r"\{[^}]*\.globals\b[^}]*\}",  # Access to globals
        r"\{[^}]*\.locals\b[^}]*\}",  # Access to locals
        r"\{[^}]*\.builtins\b[^}]*\}",  # Access to builtins
    ]

    # Allowed template variables (whitelist approach)
    ALLOWED_TEMPLATE_VARS = {
        "message",
        "current_datetime",
        "session_id",
        "namespace",
        "user_id",
        "model_name",
        "context",
        "topics",
        "entities",
    }

    # Maximum prompt length to prevent resource exhaustion
    MAX_PROMPT_LENGTH = 10000

    def __init__(self, strict_mode: bool = True):
        """
        Initialize prompt validator.

        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
        self.dangerous_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.DANGEROUS_PATTERNS
        ]

    def validate_prompt(self, prompt: str) -> None:
        """
        Validate a user-provided prompt for security issues.

        Args:
            prompt: The prompt to validate

        Raises:
            PromptSecurityError: If security issues are found
        """
        if not isinstance(prompt, str):
            raise PromptSecurityError("Prompt must be a string")

        # Check length
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise PromptSecurityError(
                f"Prompt too long: {len(prompt)} > {self.MAX_PROMPT_LENGTH}"
            )

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(prompt):
                raise PromptSecurityError(
                    f"Potentially malicious pattern detected: {pattern.pattern}"
                )

        # Validate template variables
        self._validate_template_variables(prompt)

    def _validate_template_variables(self, prompt: str) -> None:
        """Validate template variables in the prompt."""
        # Find all template variables
        template_vars = re.findall(r"\{([^}]+)\}", prompt)

        for var in template_vars:
            # Check for complex expressions (potential injection)
            if any(
                dangerous in var.lower()
                for dangerous in [
                    "__",
                    "import",
                    "eval",
                    "exec",
                    "globals",
                    "locals",
                    "builtins",
                ]
            ):
                raise PromptSecurityError(f"Dangerous template variable: {var}")

            # In strict mode, only allow whitelisted variables
            if self.strict_mode:
                var_name = var.split(".")[0].split("[")[0]  # Get base variable name
                if var_name not in self.ALLOWED_TEMPLATE_VARS:
                    raise PromptSecurityError(
                        f"Template variable not allowed: {var_name}"
                    )

    def sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize a prompt by removing potentially dangerous content.

        Args:
            prompt: The prompt to sanitize

        Returns:
            Sanitized prompt
        """
        # Validate first
        self.validate_prompt(prompt)

        # Remove excessive whitespace
        sanitized = re.sub(r"\s+", " ", prompt.strip())

        # Escape any remaining problematic characters
        # This is conservative but safe
        if self.strict_mode:
            # Only allow printable ASCII plus common punctuation
            allowed_chars = set(
                string.ascii_letters + string.digits + string.punctuation + " \n\t"
            )
            sanitized = "".join(c for c in sanitized if c in allowed_chars)

        return sanitized


class SecureFormatter:
    """Safe string formatter that prevents template injection."""

    def __init__(self, allowed_keys: set[str] | None = None):
        """
        Initialize secure formatter.

        Args:
            allowed_keys: Set of allowed template variable names
        """
        self.allowed_keys = allowed_keys or set()

    def safe_format(self, template: str, **kwargs) -> str:
        """
        Safely format a template string with restricted variable access.

        Args:
            template: Template string to format
            **kwargs: Variables to substitute

        Returns:
            Formatted string

        Raises:
            PromptSecurityError: If unsafe operations detected
        """
        # Filter kwargs to only allowed keys if specified
        if self.allowed_keys:
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in self.allowed_keys
            }
        else:
            # Sanitize all values
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str):
                    # Escape potentially dangerous strings
                    filtered_kwargs[k] = self._sanitize_value(v)
                elif isinstance(v, int | float | bool):
                    filtered_kwargs[k] = v
                else:
                    # Convert other types to safe string representation
                    filtered_kwargs[k] = str(v)

        try:
            return template.format(**filtered_kwargs)
        except (KeyError, ValueError) as e:
            raise PromptSecurityError(f"Template formatting error: {e}") from e

    def _sanitize_value(self, value: str) -> str:
        """Sanitize a string value to prevent injection."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r"[{}\\]", "", str(value))
        return sanitized[:1000]  # Limit length


# Global instances for common use
default_validator = PromptValidator(strict_mode=True)
lenient_validator = PromptValidator(strict_mode=False)
secure_formatter = SecureFormatter()


def validate_custom_prompt(prompt: str, strict: bool = True) -> None:
    """
    Convenience function to validate a custom prompt.

    Args:
        prompt: The prompt to validate
        strict: Whether to use strict validation rules

    Raises:
        PromptSecurityError: If security issues found
    """
    validator = default_validator if strict else lenient_validator
    validator.validate_prompt(prompt)


def secure_format_prompt(
    template: str, allowed_vars: set[str] | None = None, **kwargs
) -> str:
    """
    Securely format a prompt template.

    Args:
        template: Template string
        allowed_vars: Set of allowed variable names
        **kwargs: Template variables

    Returns:
        Safely formatted prompt
    """
    formatter = SecureFormatter(allowed_vars)
    return formatter.safe_format(template, **kwargs)
