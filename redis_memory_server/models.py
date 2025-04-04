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
    context: str | None = None
    user_id: str | None = None
    last_accessed: int | None = None
    created_at: int | None = None
    tokens: int | None = 0
    namespace: str | None = None


class StoredSessionMemory(SessionMemory):
    """Stored session memory"""

    session_id: str
    created_at: int | None = None


class SessionMemoryResponse(SessionMemory):
    """Response containing a session's memory"""


class SearchPayload(BaseModel):
    """Payload for semantic search"""

    text: str
    session_id: str | None = None
    namespace: str | None = None
    topics: list[str] | None = None
    entities: list[str] | None = None
    distance_threshold: float | None = None
    limit: int = Field(default=10, ge=1)
    offset: int = Field(default=0, ge=0)


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class RedisearchResult(BaseModel):
    """Result from a redisearch query"""

    role: str
    content: str
    dist: float


class SearchResults(BaseModel):
    """Results from a redisearch query"""

    docs: list[RedisearchResult]
    total: int


class NamespaceQuery(BaseModel):
    """Query parameters for namespace"""

    namespace: str | None = None


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1)
    namespace: str | None = None
