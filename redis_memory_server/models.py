import logging

from pydantic import BaseModel, Field

from redis_memory_server.utils import (
    TokenEscaper,
)


logger = logging.getLogger(__name__)
escaper = TokenEscaper()

JSONTypes = str | float | int | bool | list | dict


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str


class SessionMemory(BaseModel):
    """A session's memory"""

    messages: list[MemoryMessage]
    session_id: str | None = None
    context: str | None = None
    user_id: str | None = None
    last_accessed: int | None = None
    created_at: int | None = None
    tokens: int | None = 0
    namespace: str | None = None


class LongTermMemory(BaseModel):
    """A long-term memory"""

    text: str
    id_: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    last_accessed: int | None = None
    created_at: int | None = None
    namespace: str | None = None
    topics: list[str] | None = None
    entities: list[str] | None = None


class SessionMemoryResponse(SessionMemory):
    """Response containing a session's memory"""


class SessionListResponse(BaseModel):
    """Response containing a list of sessions"""

    sessions: list[str]
    total: int


class SearchPayload(BaseModel):
    """Payload for long-term memory search"""

    text: str
    session_id: str | None = None
    namespace: str | None = None
    topics: list[str] | None = None
    entities: list[str] | None = None
    distance_threshold: float | None = None
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


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


class LongTermMemoryResultsResponse(LongTermMemoryResults):
    """Response containing long-term memory search results"""


class CreateLongTermMemoryPayload(BaseModel):
    """Payload for creating a long-term memory"""

    memories: list[LongTermMemory]


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    namespace: str | None = None
