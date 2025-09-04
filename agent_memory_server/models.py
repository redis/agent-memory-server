import logging
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from mcp.server.fastmcp.prompts import base
from mcp.types import AudioContent, EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel, Field
from ulid import ULID

from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    EventDate,
    LastAccessed,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)


logger = logging.getLogger(__name__)

JSONTypes = str | float | int | bool | list | dict


class MemoryTypeEnum(str, Enum):
    """Enum for memory types with string values"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    MESSAGE = "message"


# These should match the keys in MODEL_CONFIGS
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


class MemoryStrategyConfig(BaseModel):
    """Configuration for memory extraction strategy."""

    strategy: Literal["discrete", "summary", "preferences", "custom"] = Field(
        default="discrete", description="Type of memory extraction strategy to use"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific configuration options"
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override to ensure JSON serialization works properly."""
        return super().model_dump(mode="json", **kwargs)


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str
    id: str = Field(
        default_factory=lambda: str(ULID()),
        description="Unique identifier for the message (auto-generated if not provided)",
    )
    persisted_at: datetime | None = Field(
        default=None,
        description="Server-assigned timestamp when message was persisted to long-term storage",
    )
    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="f",
        description="Whether memory extraction has run for this message",
    )


class SessionListResponse(BaseModel):
    """Response containing a list of sessions"""

    sessions: list[str]
    total: int


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
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the memory was last accessed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the memory was created",
    )
    updated_at: datetime = Field(
        description="Datetime when the memory was last updated",
        default_factory=lambda: datetime.now(UTC),
    )
    pinned: bool = Field(
        default=False,
        description="Whether this memory is pinned and should not be auto-deleted",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been accessed (best-effort, rate-limited)",
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
    extraction_strategy: str = Field(
        default="discrete",
        description="Memory extraction strategy used when this was promoted from working memory",
    )
    extraction_strategy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the extraction strategy used",
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
        description="Client-provided ID for deduplication and overwrites",
    )


class WorkingMemory(BaseModel):
    """Working memory for a session - contains both messages and structured memory records"""

    messages: list[MemoryMessage] = Field(
        default_factory=list,
        description="Conversation messages (role/content pairs)",
    )
    memories: list[MemoryRecord | ClientMemoryRecord] = Field(
        default_factory=list,
        description="Structured memory records for promotion to long-term storage",
    )
    data: dict[str, JSONTypes] | None = Field(
        default=None,
        description="Arbitrary JSON data storage (key-value pairs)",
    )
    context: str | None = Field(
        default=None,
        description="Summary of past session messages if server has auto-summarized",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the working memory",
    )
    tokens: int = Field(
        default=0,
        description="Optional number of tokens in the working memory",
    )
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
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the working memory was last accessed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the working memory was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the working memory was last updated",
    )

    def get_create_long_term_memory_tool_description(self) -> str:
        """
        Generate a strategy-aware description for the create_long_term_memory MCP tool.

        Returns:
            Description string that includes strategy-specific extraction behavior
        """
        from agent_memory_server.memory_strategies import get_memory_strategy

        # Get the configured strategy
        strategy = get_memory_strategy(
            self.long_term_memory_strategy.strategy,
            **self.long_term_memory_strategy.config,
        )

        base_description = """Create long-term memories that can be searched later.

This tool creates persistent memories that are stored for future retrieval. Use this
when you want to remember information that would be useful in future conversations.

MEMORY EXTRACTION BEHAVIOR:
The memory extraction for this session is configured with: {}

MEMORY TYPES:
1. **SEMANTIC MEMORIES** (memory_type="semantic"):
   - User preferences and general knowledge
   - Facts, rules, and persistent information
   - Examples:
     * "User prefers dark mode in all applications"
     * "User is a data scientist working with Python"
     * "User dislikes spicy food"
     * "The company's API rate limit is 1000 requests per hour"

2. **EPISODIC MEMORIES** (memory_type="episodic"):
   - Specific events, experiences, or time-bound information
   - Things that happened at a particular time or in a specific context
   - MUST have a time dimension to be truly episodic
   - Should include an event_date when the event occurred
   - Examples:
     * "User visited Paris last month and had trouble with the metro"
     * "User reported a login bug on January 15th, 2024"
     * "User completed the onboarding process yesterday"
     * "User mentioned they're traveling to Tokyo next week"

IMPORTANT NOTES ON SESSION IDs:
- When including a session_id, use the EXACT session identifier from the current conversation
- NEVER invent or guess a session ID - if you don't know it, omit the field
- If you want memories accessible across all sessions, omit the session_id field

Args:
    memories: A list of MemoryRecord objects to create

Returns:
    An acknowledgement response indicating success"""

        return base_description.format(strategy.get_extraction_description())

    def create_long_term_memory_tool(self) -> Callable:
        """
        Create a strategy-aware MCP tool function for creating long-term memories.

        This method generates a tool function that uses the working memory's
        configured strategy for memory extraction guidance.

        Returns:
            A callable MCP tool function with strategy-aware description
        """
        description = self.get_create_long_term_memory_tool_description()

        async def create_long_term_memories_with_strategy(memories: list[dict]) -> dict:
            """
            Create long-term memories using the configured extraction strategy.

            This tool is generated dynamically based on the working memory session's
            configured memory extraction strategy.
            """
            # Import here to avoid circular imports
            from agent_memory_server.api import (
                create_long_term_memory as core_create_long_term_memory,
            )
            from agent_memory_server.config import settings
            from agent_memory_server.dependencies import get_background_tasks
            from agent_memory_server.models import (
                CreateMemoryRecordRequest,
                LenientMemoryRecord,
            )

            # Apply default namespace for STDIO if not provided in memory entries
            processed_memories = []
            for mem_data in memories:
                if isinstance(mem_data, dict):
                    mem = LenientMemoryRecord(**mem_data)
                else:
                    mem = mem_data

                if mem.namespace is None and settings.default_mcp_namespace:
                    mem.namespace = settings.default_mcp_namespace
                if mem.user_id is None and settings.default_mcp_user_id:
                    mem.user_id = settings.default_mcp_user_id

                processed_memories.append(mem)

            payload = CreateMemoryRecordRequest(memories=processed_memories)
            result = await core_create_long_term_memory(
                payload, background_tasks=get_background_tasks()
            )
            return result.model_dump() if hasattr(result, "model_dump") else result

        # Set the function's metadata
        create_long_term_memories_with_strategy.__doc__ = description
        create_long_term_memories_with_strategy.__name__ = (
            f"create_long_term_memories_{self.long_term_memory_strategy.strategy}"
        )

        return create_long_term_memories_with_strategy


class WorkingMemoryResponse(WorkingMemory):
    """Response containing working memory"""

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


class WorkingMemoryRequest(BaseModel):
    """Request parameters for working memory operations"""

    session_id: str
    namespace: str | None = None
    user_id: str | None = None
    model_name: ModelNameLiteral | None = None
    context_window_max: int | None = None
    long_term_memory_strategy: MemoryStrategyConfig | None = Field(
        default=None,
        description="Configuration for memory extraction strategy when promoting to long-term memory",
    )


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class MemoryRecordResult(MemoryRecord):
    """Result from a memory search"""

    dist: float


class MemoryRecordResults(BaseModel):
    """Results from a memory search"""

    memories: list[MemoryRecordResult]
    total: int
    next_offset: int | None = None


class MemoryRecordResultsResponse(MemoryRecordResults):
    """Response containing memory search results"""


class CreateMemoryRecordRequest(BaseModel):
    """Payload for creating memory records"""

    memories: list[ExtractedMemoryRecord]


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    namespace: str | None = None
    user_id: str | None = None


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


class SearchRequest(BaseModel):
    """Payload for long-term memory search"""

    text: str | None = Field(
        default=None,
        description="Optional text to use for a semantic search",
    )
    session_id: SessionId | None = Field(
        default=None,
        description="Optional session ID to filter by",
    )
    namespace: Namespace | None = Field(
        default=None,
        description="Optional namespace to filter by",
    )
    topics: Topics | None = Field(
        default=None,
        description="Optional topics to filter by",
    )
    entities: Entities | None = Field(
        default=None,
        description="Optional entities to filter by",
    )
    created_at: CreatedAt | None = Field(
        default=None,
        description="Optional created at timestamp to filter by",
    )
    last_accessed: LastAccessed | None = Field(
        default=None,
        description="Optional last accessed timestamp to filter by",
    )
    user_id: UserId | None = Field(
        default=None,
        description="Optional user ID to filter by",
    )
    distance_threshold: float | None = Field(
        default=None,
        description="Optional distance threshold to filter by",
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description="Optional memory type to filter by",
    )
    event_date: EventDate | None = Field(
        default=None,
        description="Optional event date to filter by (for episodic memories)",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Optional limit on the number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Optional offset",
    )

    # Recency re-ranking controls (optional)
    recency_boost: bool | None = Field(
        default=None,
        description="Enable recency-aware re-ranking (defaults to enabled if None)",
    )
    recency_semantic_weight: float | None = Field(
        default=None,
        description="Weight for semantic similarity",
    )
    recency_recency_weight: float | None = Field(
        default=None,
        description="Weight for recency score",
    )
    recency_freshness_weight: float | None = Field(
        default=None,
        description="Weight for freshness component",
    )
    recency_novelty_weight: float | None = Field(
        default=None,
        description="Weight for novelty (age) component",
    )
    recency_half_life_last_access_days: float | None = Field(
        default=None, description="Half-life (days) for last_accessed decay"
    )
    recency_half_life_created_days: float | None = Field(
        default=None, description="Half-life (days) for created_at decay"
    )

    # Server-side recency rerank (Redis-only path) toggle
    server_side_recency: bool | None = Field(
        default=None,
        description="If true, attempt server-side recency-aware re-ranking when supported by backend",
    )

    def get_filters(self):
        """Get all filter objects as a dictionary"""
        filters = {}

        if self.session_id is not None:
            filters["session_id"] = self.session_id

        if self.namespace is not None:
            filters["namespace"] = self.namespace

        if self.topics is not None:
            filters["topics"] = self.topics

        if self.entities is not None:
            filters["entities"] = self.entities

        if self.user_id is not None:
            filters["user_id"] = self.user_id

        if self.created_at is not None:
            filters["created_at"] = self.created_at

        if self.last_accessed is not None:
            filters["last_accessed"] = self.last_accessed

        if self.memory_type is not None:
            filters["memory_type"] = self.memory_type

        if self.event_date is not None:
            filters["event_date"] = self.event_date

        return filters


class MemoryPromptRequest(BaseModel):
    query: str
    session: WorkingMemoryRequest | None = None
    long_term_search: SearchRequest | bool | None = None


class SystemMessage(BaseModel):
    """A system message"""

    role: Literal["system"] = "system"
    content: str | TextContent | ImageContent | AudioContent | EmbeddedResource


class UserMessage(base.Message):
    """A user message"""

    role: Literal["user"] = "user"


class MemoryPromptResponse(BaseModel):
    messages: list[base.Message | SystemMessage]


class LenientMemoryRecord(ExtractedMemoryRecord):
    """
    A memory record that can be created without an ID.

    Useful for the MCP server, where we would otherwise have to expect
    an agent or LLM to provide a memory ID.
    """

    id: str = Field(default_factory=lambda: str(ULID()))


class DeleteMemoryRecordRequest(BaseModel):
    """Payload for deleting memory records"""

    ids: list[str]


class EditMemoryRecordRequest(BaseModel):
    """Payload for editing a memory record"""

    text: str | None = Field(
        default=None, description="Updated text content for the memory"
    )
    topics: list[str] | None = Field(
        default=None, description="Updated topics for the memory"
    )
    entities: list[str] | None = Field(
        default=None, description="Updated entities for the memory"
    )
    memory_type: MemoryTypeEnum | None = Field(
        default=None, description="Updated memory type (semantic, episodic, message)"
    )
    namespace: str | None = Field(
        default=None, description="Updated namespace for the memory"
    )
    user_id: str | None = Field(
        default=None, description="Updated user ID for the memory"
    )
    session_id: str | None = Field(
        default=None, description="Updated session ID for the memory"
    )
    event_date: datetime | None = Field(
        default=None, description="Updated event date for episodic memories"
    )
