import logging
import time
from typing import Literal

from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field

from agent_memory_server.config import settings
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)


logger = logging.getLogger(__name__)

JSONTypes = str | float | int | bool | list | dict

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


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str


class SessionMemory(BaseModel):
    """A session's memory"""

    messages: list[MemoryMessage]
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for the session memory",
    )
    context: str | None = Field(
        default=None,
        description="Optional summary of past session messages",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the session memory",
    )
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the session memory",
    )
    tokens: int = Field(
        default=0,
        description="Optional number of tokens in the session memory",
    )
    last_accessed: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when the session memory was last accessed",
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when the session memory was created",
    )
    updated_at: int = Field(
        description="Timestamp when the session memory was last updated",
        default_factory=lambda: int(time.time()),
    )


class SessionMemoryRequest(BaseModel):
    session_id: str
    namespace: str | None = None
    window_size: int = settings.window_size
    model_name: ModelNameLiteral | None = None
    context_window_max: int | None = None


class SessionMemoryResponse(SessionMemory):
    """Response containing a session's memory"""


class SessionListResponse(BaseModel):
    """Response containing a list of sessions"""

    sessions: list[str]
    total: int


class LongTermMemory(BaseModel):
    """A long-term memory"""

    text: str
    id_: str | None = Field(
        default=None,
        description="Optional ID for the long-term memory",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for the long-term memory",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the long-term memory",
    )
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the long-term memory",
    )
    last_accessed: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when the memory was last accessed",
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when the memory was created",
    )
    updated_at: int = Field(
        description="Timestamp when the memory was last updated",
        default_factory=lambda: int(time.time()),
    )
    topics: list[str] | None = Field(
        default=None,
        description="Optional topics for the long-term memory",
    )
    entities: list[str] | None = Field(
        default=None,
        description="Optional entities for the long-term memory",
    )
    memory_hash: str | None = Field(
        default=None,
        description="Hash representation of the memory for deduplication",
    )
    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="f",
        description="Whether memory extraction has run for this memory (only messages)",
    )
    memory_type: Literal["episodic", "semantic", "message"] = Field(
        default="message",
        description="Type of memory",
    )


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class LongTermMemoryResult(LongTermMemory):
    """Result from a long-term memory search"""

    dist: float


class LongTermMemoryResults(BaseModel):
    """Results from a long-term memory search"""

    memories: list[LongTermMemoryResult]
    total: int
    next_offset: int | None = None


class LongTermMemoryResultsResponse(LongTermMemoryResults):
    """Response containing long-term memory search results"""


class CreateLongTermMemoryRequest(BaseModel):
    """Payload for creating a long-term memory"""

    memories: list[LongTermMemory]


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    namespace: str | None = None


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

        return filters


class MemoryPromptRequest(BaseModel):
    query: str
    session: SessionMemoryRequest | None = None
    long_term_search: SearchRequest | None = None


class MemoryPromptResponse(BaseModel):
    messages: list[base.Message]


class SystemMessage(base.Message):
    """A system message"""

    role: Literal["system"] = "system"


class UserMessage(base.Message):
    """A user message"""

    role: Literal["user"] = "user"
