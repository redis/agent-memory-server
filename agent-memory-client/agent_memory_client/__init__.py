"""
Agent Memory Client

A Python client for the Agent Memory Server REST API providing comprehensive
memory management capabilities for AI agents and applications.
"""

__version__ = "0.14.0"

from .client import MemoryAPIClient, MemoryClientConfig, create_memory_client
from .exceptions import (
    MemoryClientError,
    MemoryNotFoundError,
    MemoryServerError,
    MemoryValidationError,
)
from .models import (
    # Re-export essential models for convenience
    CreateSummaryViewRequest,
    ForgetPolicy,
    ForgetResponse,
    ModelNameLiteral,
    SummaryView,
    SummaryViewPartitionResult,
    Task,
)
from .tool_schema import ToolSchema, ToolSchemaCollection

__all__ = [
    # Client classes
    "MemoryAPIClient",
    "MemoryClientConfig",
    "create_memory_client",
    # Tool schema classes
    "ToolSchema",
    "ToolSchemaCollection",
    # Exceptions
    "MemoryClientError",
    "MemoryValidationError",
    "MemoryNotFoundError",
    "MemoryServerError",
    # Types
    "ModelNameLiteral",
    # Forget
    "ForgetPolicy",
    "ForgetResponse",
    # Summary Views
    "SummaryView",
    "CreateSummaryViewRequest",
    "SummaryViewPartitionResult",
    # Tasks
    "Task",
]
