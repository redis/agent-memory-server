from typing import List, Type
import nanoid
import numpy as np
from redis.asyncio import Redis
from redis.commands.search.query import Query
from models import (
    MemoryMessage,
    OpenAIClientWrapper,
    RedisearchResult,
    SearchResults,
)
import logging

from utils import REDIS_INDEX_NAME, Keys

logger = logging.getLogger(__name__)


async def index_messages(
    messages: List[MemoryMessage],
    session_id: str,
    openai_client: OpenAIClientWrapper,
    redis_conn: Redis,
) -> None:
    """Index messages in Redis for vector search"""
    try:
        # Extract contents for embedding
        contents = [msg.content for msg in messages]

        # Get embeddings from OpenAI
        embeddings = await openai_client.create_embedding(contents)

        # Index each message with its embedding
        for index, embedding in enumerate(embeddings):
            # Generate unique ID for the message
            id = nanoid.generate()
            key = Keys.memory_key(id)

            # Encode the embedding vector as bytes
            vector = embedding.tobytes()

            # Store in Redis with HSET
            await redis_conn.hset(  # type: ignore
                key,
                mapping={
                    "session": session_id,
                    "vector": vector,
                    "content": contents[index],
                    "role": messages[index].role,
                },
            )

        logger.info(f"Indexed {len(messages)} messages for session {session_id}")
        return None
    except Exception as e:
        logger.error(f"Error indexing messages: {e}")
        raise


class Unset:
    pass


async def search_messages(
    query: str,
    session_id: str,
    openai_client: OpenAIClientWrapper,
    redis_conn: Redis,
    distance_threshold: float | Type[Unset] = Unset,
    limit: int = 10,
) -> SearchResults:
    """Search for messages using vector similarity"""
    try:
        # Get embedding for query
        query_embedding = await openai_client.create_embedding([query])
        vector = query_embedding.tobytes()
        params = {"vec": vector}

        if distance_threshold and distance_threshold is not Unset:
            base_query = Query(
                f"@session:{{{session_id}}} @vector:[VECTOR_RANGE $radius $vec]=>{{$YIELD_DISTANCE_AS: dist}}"
            )
            params = {"vec": vector, "radius": distance_threshold}
        else:
            base_query = Query(
                f"@session:{{{session_id}}}=>[KNN {limit} @vector $vec AS dist]"
            )
        q = (
            base_query.return_fields("role", "content", "dist")
            .sort_by("dist", asc=True)
            .paging(0, limit)
            .dialect(2)
        )

        raw_results = await redis_conn.ft(REDIS_INDEX_NAME).search(
            q,
            query_params=params,  # type: ignore
        )

        # Parse results
        results = [
            RedisearchResult(role=doc.role, content=doc.content, dist=doc.dist)
            for doc in raw_results.docs
        ]

        logger.info(f"Found {len(results)} results for query in session {session_id}")
        return SearchResults(total=raw_results.total, docs=results)
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise
