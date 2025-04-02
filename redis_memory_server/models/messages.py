import logging

import nanoid
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.commands.search.query import Query

from redis_memory_server.llms import OpenAIClientWrapper
from redis_memory_server.utils import REDIS_INDEX_NAME, Keys, TokenEscaper


logger = logging.getLogger(__name__)
escaper = TokenEscaper()


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str
    topics: list[str] = Field(
        default_factory=list, description="List of topics associated with this message"
    )
    entities: list[str] = Field(
        default_factory=list, description="List of entities mentioned in this message"
    )


class MemoryMessagesAndContext(BaseModel):
    """Request payload for adding messages to memory"""

    messages: list[MemoryMessage]
    context: str | None = None


class MemoryResponse(BaseModel):
    """Response containing messages and context"""

    messages: list[MemoryMessage]
    context: str | None = None
    tokens: int | None = None


class SearchPayload(BaseModel):
    """Payload for semantic search"""

    text: str


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


async def index_messages(
    messages: list[MemoryMessage],
    session_id: str,
    client: OpenAIClientWrapper,  # Only OpenAI supports embeddings currently
    redis_conn: Redis,
    namespace: str | None = None,
) -> None:
    """Index messages in Redis for vector search"""
    try:
        # Extract contents for embedding
        contents = [msg.content for msg in messages]

        # Get embeddings from OpenAI
        embeddings = await client.create_embedding(contents)

        # Index each message with its embedding
        for index, embedding in enumerate(embeddings):
            # Generate unique ID for the message
            id = nanoid.generate()
            key = Keys.memory_key(id, namespace)

            # Encode the embedding vector as bytes
            vector = embedding.tobytes()

            # Store in Redis with HSET
            await redis_conn.hset(  # type: ignore
                key,
                mapping={
                    "session": session_id or "",
                    "namespace": namespace or "",
                    "vector": vector,
                    "content": contents[index],
                    "role": messages[index].role,
                },
            )

        logger.info(f"Indexed {len(messages)} messages for session {session_id}")
        return
    except Exception as e:
        logger.error(f"Error indexing messages: {e}")
        raise


class Unset:
    pass


async def search_messages(
    query: str,
    client: OpenAIClientWrapper,  # Only OpenAI supports embeddings currently
    redis_conn: Redis,
    session_id: str | None = None,
    namespace: str | None = None,
    distance_threshold: float | type[Unset] = Unset,
    limit: int = 10,
) -> SearchResults:
    """Search for messages using vector similarity"""
    try:
        query = escaper.escape(query)
        if session_id:
            session_id = escaper.escape(session_id)
        if namespace:
            namespace = escaper.escape(namespace)

        # Get embedding for query
        query_embedding = await client.create_embedding([query])
        vector = query_embedding.tobytes()

        # Set up query parameters
        params = {"vec": vector}
        namespace_filter = f"@namespace:{{{namespace}}}" if namespace else ""
        session_filter = f"@session:{{{session_id}}}" if session_id else ""

        if distance_threshold and distance_threshold is not Unset:
            base_query = Query(
                f"{session_filter} {namespace_filter} @vector:[VECTOR_RANGE $radius $vec]=>{{$YIELD_DISTANCE_AS: dist}}"
            )
            params = {"vec": vector, "radius": distance_threshold}
        else:
            base_query = Query(
                f"{session_filter} {namespace_filter}=>[KNN {limit} @vector $vec AS dist]"
            )

        q = (
            base_query.return_fields("role", "content", "dist")
            .sort_by("dist", asc=True)
            .paging(0, limit)
            .dialect(2)
        )

        # Execute search
        raw_results = await redis_conn.ft(REDIS_INDEX_NAME).search(
            q,
            query_params=params,  # type: ignore
        )

        # Parse results safely
        results = []
        total_results = 0

        # Check if raw_results has the expected attributes
        if hasattr(raw_results, "docs") and isinstance(raw_results.docs, list):
            for doc in raw_results.docs:
                if (
                    hasattr(doc, "role")
                    and hasattr(doc, "content")
                    and hasattr(doc, "dist")
                ):
                    results.append(
                        RedisearchResult(
                            role=doc.role, content=doc.content, dist=float(doc.dist)
                        )
                    )

            total_results = getattr(raw_results, "total", len(results))
        else:
            # Handle the case where raw_results doesn't have the expected structure
            logger.warning("Unexpected search result format")
            total_results = 0

        logger.info(f"Found {len(results)} results for query in session {session_id}")
        return SearchResults(total=total_results, docs=results)
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise
