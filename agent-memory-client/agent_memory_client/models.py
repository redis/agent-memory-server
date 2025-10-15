"""
Data models for the Agent Memory Client.

This module contains essential data models needed by the client.
For full model definitions, see the main agent_memory_server package.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from ulid import ULID

# Model name literals for model-specific window sizes
ModelNameLiteral = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3-mini",
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
]


class MemoryTypeEnum(str, Enum):
    """Enum for memory types with string values"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    MESSAGE = "message"


class MemoryStrategyConfig(BaseModel):
    """Configuration for memory extraction strategy."""

    strategy: Literal["discrete", "summary", "preferences", "custom"] = Field(
        default="discrete", description="Type of memory extraction strategy to use"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific configuration options"
    )


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str
    id: str = Field(
        default_factory=lambda: str(ULID()),
        description="Unique identifier for the message (auto-generated)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was created",
    )
    persisted_at: datetime | None = Field(
        default=None,
        description="Server-assigned timestamp when message was persisted to long-term storage",
    )
    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="f",
        description="Whether memory extraction has run for this message",
    )


class MemoryRecord(BaseModel):
    """A memory record"""

    id: str = Field(description="Client-provided ID for deduplication and overwrites")
    text: str
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for the memory record",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the memory record",
    )
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the memory record",
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Datetime when the memory was last accessed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Datetime when the memory was created",
    )
    updated_at: datetime = Field(
        description="Datetime when the memory was last updated",
        default_factory=lambda: datetime.now(timezone.utc),
    )
    topics: list[str] | None = Field(
        default=None,
        description="Optional topics for the memory record",
    )
    entities: list[str] | None = Field(
        default=None,
        description="Optional entities for the memory record",
    )
    memory_hash: str | None = Field(
        default=None,
        description="Hash representation of the memory for deduplication",
    )
    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="f",
        description="Whether memory extraction has run for this memory",
    )
    memory_type: MemoryTypeEnum = Field(
        default=MemoryTypeEnum.MESSAGE,
        description="Type of memory",
    )
    persisted_at: datetime | None = Field(
        default=None,
        description="Server-assigned timestamp when memory was persisted to long-term storage",
    )
    extracted_from: list[str] | None = Field(
        default=None,
        description="List of message IDs that this memory was extracted from",
    )
    event_date: datetime | None = Field(
        default=None,
        description="Date/time when the event described in this memory occurred (primarily for episodic memories)",
    )


class ExtractedMemoryRecord(MemoryRecord):
    """A memory record that has already been extracted (e.g., explicit memories from API/MCP)"""

    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="t",
        description="Whether memory extraction has run for this memory",
    )
    memory_type: MemoryTypeEnum = Field(
        default=MemoryTypeEnum.SEMANTIC,
        description="Type of memory",
    )


class ClientMemoryRecord(MemoryRecord):
    """A memory record with a client-provided ID"""

    id: str = Field(
        default_factory=lambda: str(ULID()),
        description="Client-provided ID generated by the client (ULID)",
    )


JSONTypes = str | float | int | bool | list[Any] | dict[str, Any]


class WorkingMemory(BaseModel):
    """Working memory for a session - contains both messages and structured memory records"""

    # Support both message-based memory (conversation) and structured memory records
    messages: list[MemoryMessage] = Field(
        default_factory=list,
        description="Conversation messages with tracking fields",
    )
    memories: list[MemoryRecord | ClientMemoryRecord] = Field(
        default_factory=list,
        description="Structured memory records for promotion to long-term storage",
    )

    # Arbitrary JSON data storage (separate from memories)
    data: dict[str, JSONTypes] | None = Field(
        default=None,
        description="Arbitrary JSON data storage (key-value pairs)",
    )

    # Session context and metadata
    context: str | None = Field(
        default=None,
        description="Optional summary of past session messages",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the working memory",
    )
    tokens: int = Field(
        default=0,
        description="Optional number of tokens in the working memory",
    )

    # Required session scoping
    session_id: str
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the working memory",
    )
    long_term_memory_strategy: MemoryStrategyConfig = Field(
        default_factory=MemoryStrategyConfig,
        description="Configuration for memory extraction strategy when promoting to long-term memory",
    )

    # TTL and timestamps
    ttl_seconds: int | None = Field(
        default=None,  # Persistent by default
        description="TTL for the working memory in seconds",
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Datetime when the working memory was last accessed",
    )


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class HealthCheckResponse(BaseModel):
    """Health check response"""

    now: float


class SessionListResponse(BaseModel):
    """Response containing a list of sessions"""

    sessions: list[str]
    total: int


class WorkingMemoryResponse(WorkingMemory):
    """Response from working memory operations"""

    context_percentage_total_used: float | None = Field(
        default=None,
        description="Percentage of total context window currently used (0-100)",
    )
    context_percentage_until_summarization: float | None = Field(
        default=None,
        description="Percentage until auto-summarization triggers (0-100, reaches 100% at summarization threshold)",
    )
    new_session: bool | None = Field(
        default=None,
        description="True if session was created, False if existing session was found, None if not applicable",
    )
    unsaved: bool | None = Field(
        default=None,
        description="True if this session data has not been persisted to Redis yet (deprecated behavior for old clients)",
    )


class MemoryRecordResult(MemoryRecord):
    """Result from a memory search"""

    dist: float


class RecencyConfig(BaseModel):
    """Client-side configuration for recency-aware ranking options."""

    recency_boost: bool | None = Field(
        default=None, description="Enable recency-aware re-ranking"
    )
    semantic_weight: float | None = Field(
        default=None, description="Weight for semantic similarity"
    )
    recency_weight: float | None = Field(
        default=None, description="Weight for recency score"
    )
    freshness_weight: float | None = Field(
        default=None, description="Weight for freshness component"
    )
    novelty_weight: float | None = Field(
        default=None, description="Weight for novelty/age component"
    )

    half_life_last_access_days: float | None = Field(
        default=None, description="Half-life (days) for last_accessed decay"
    )
    half_life_created_days: float | None = Field(
        default=None, description="Half-life (days) for created_at decay"
    )
    server_side_recency: bool | None = Field(
        default=None,
        description="If true, attempt server-side recency ranking (Redis-only)",
    )


class MemoryRecordResults(BaseModel):
    """Results from memory search operations"""

    memories: list[MemoryRecordResult]
    total: int
    next_offset: int | None = None


class MemoryPromptResponse(BaseModel):
    """Response from memory prompt endpoint"""

    messages: list[dict[str, Any]]  # Simplified to avoid MCP dependencies
