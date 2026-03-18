"""
This module provides an abstraction layer for memory vector database operations,
with a RedisVL-based implementation for Redis backends.
"""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import reduce
from typing import Any

import numpy as np
from redisvl.index import AsyncSearchIndex
from redisvl.query import (
    AggregateHybridQuery,
    FilterQuery,
    RangeQuery,
    TextQuery,
    VectorQuery,
)

from agent_memory_server.filters import (
    CreatedAt,
    DiscreteMemoryExtracted,
    Entities,
    EventDate,
    Id,
    LastAccessed,
    MemoryHash,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    SearchModeEnum,
    SearchScoreTypeEnum,
)
from agent_memory_server.utils.recency import generate_memory_hash, rerank_with_recency
from agent_memory_server.utils.redis_query import RecencyAggregationQuery
from agent_memory_server.utils.tag_codec import decode_tag_values, encode_tag_values


logger = logging.getLogger(__name__)


class MemoryVectorDatabase(ABC):
    """Abstract base class for memory vector database implementations.

    Defines the pure interface that all memory vector database backends
    must implement. No database-specific dependencies in the base class.
    """

    @abstractmethod
    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memory records to the database.

        Args:
            memories: List of MemoryRecord objects to add

        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        search_mode: SearchModeEnum = SearchModeEnum.SEMANTIC,
        hybrid_alpha: float = 0.7,
        text_scorer: str = "BM25STD",
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        distance_threshold: float | None = None,
        server_side_recency: bool | None = None,
        recency_params: dict | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories in the database.

        Args:
            query: Text query for semantic, keyword, or hybrid search
            search_mode: Which search strategy to use
            hybrid_alpha: Weight assigned to vector similarity in hybrid search
            text_scorer: Redis full-text scorer to use for keyword or hybrid search
            session_id: Optional session ID filter
            user_id: Optional user ID filter
            namespace: Optional namespace filter
            created_at: Optional created at filter
            last_accessed: Optional last accessed filter
            topics: Optional topics filter
            entities: Optional entities filter
            memory_type: Optional memory type filter
            event_date: Optional event date filter
            memory_hash: Optional memory hash filter
            id: Optional memory ID filter
            discrete_memory_extracted: Optional discrete memory extracted filter
            distance_threshold: Optional similarity threshold
            server_side_recency: Whether to use server-side recency scoring
            recency_params: Parameters for recency scoring
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            MemoryRecordResults containing matching memories
        """
        pass

    @abstractmethod
    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs.

        Args:
            memory_ids: List of memory IDs to delete

        Returns:
            Number of memories deleted
        """
        pass

    @abstractmethod
    async def update_memories(self, memories: list[MemoryRecord]) -> int:
        """Update memory records in the database.

        Args:
            memories: List of MemoryRecord objects to update

        Returns:
            Number of memories updated
        """
        pass

    @abstractmethod
    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories matching the given filters.

        Args:
            namespace: Optional namespace filter
            user_id: Optional user ID filter
            session_id: Optional session ID filter

        Returns:
            Number of matching memories
        """
        pass

    @abstractmethod
    async def list_memories(
        self,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """List memories matching the given filters without semantic search.

        This method retrieves memories based solely on metadata filters,
        without requiring a query for embedding. Use this when you need to
        filter by hash, ID, or other metadata fields without semantic similarity.

        Args:
            session_id: Optional session ID filter
            user_id: Optional user ID filter
            namespace: Optional namespace filter
            created_at: Optional created at filter
            last_accessed: Optional last accessed filter
            topics: Optional topics filter
            entities: Optional entities filter
            memory_type: Optional memory type filter
            event_date: Optional event date filter
            memory_hash: Optional memory hash filter
            id: Optional memory ID filter
            discrete_memory_extracted: Optional discrete memory extracted filter
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            MemoryRecordResults containing matching memories
        """
        pass

    def _parse_list_field(self, field_value: Any) -> list[str]:
        """Parse a field that might be a list, serialized TAG string, or None.

        Args:
            field_value: Value that may be a list, string, or None

        Returns:
            List of strings, empty list if field_value is falsy
        """
        return decode_tag_values(field_value)

    def generate_memory_hash(self, memory: MemoryRecord) -> str:
        """Generate a stable hash for a memory based on text, user_id, and session_id.

        Args:
            memory: MemoryRecord to hash

        Returns:
            A stable hash string
        """
        return generate_memory_hash(memory)

    def _apply_client_side_recency_reranking(
        self, memory_results: list[MemoryRecordResult], recency_params: dict | None
    ) -> list[MemoryRecordResult]:
        """Apply client-side recency reranking as a fallback when server-side is not available.

        Args:
            memory_results: List of memory results to rerank
            recency_params: Parameters for recency scoring

        Returns:
            Reranked list of memory results
        """
        if not memory_results:
            return memory_results

        try:
            now = datetime.now(UTC)
            params = {
                "semantic_weight": float(recency_params.get("semantic_weight", 0.8))
                if recency_params
                else 0.8,
                "recency_weight": float(recency_params.get("recency_weight", 0.2))
                if recency_params
                else 0.2,
                "freshness_weight": float(recency_params.get("freshness_weight", 0.6))
                if recency_params
                else 0.6,
                "novelty_weight": float(recency_params.get("novelty_weight", 0.4))
                if recency_params
                else 0.4,
                "half_life_last_access_days": float(
                    recency_params.get("half_life_last_access_days", 7.0)
                )
                if recency_params
                else 7.0,
                "half_life_created_days": float(
                    recency_params.get("half_life_created_days", 30.0)
                )
                if recency_params
                else 30.0,
            }
            return rerank_with_recency(memory_results, now=now, params=params)
        except Exception as e:
            logger.warning(f"Client-side recency reranking failed: {e}")
            return memory_results


class RedisVLMemoryVectorDatabase(MemoryVectorDatabase):
    """Redis memory vector database implementation using RedisVL directly.

    Performs indexing and search operations using RedisVL's AsyncSearchIndex,
    VectorQuery, FilterQuery, and RangeQuery without any LangChain dependencies.
    """

    RETURN_FIELDS = [
        "id_",
        "text",
        "session_id",
        "user_id",
        "namespace",
        "created_at",
        "last_accessed",
        "updated_at",
        "pinned",
        "access_count",
        "topics",
        "entities",
        "memory_hash",
        "discrete_memory_extracted",
        "memory_type",
        "persisted_at",
        "extracted_from",
        "event_date",
    ]

    def __init__(self, index: AsyncSearchIndex, embeddings: Any):
        """Initialize the RedisVL memory vector database.

        Args:
            index: An AsyncSearchIndex instance (connected but not necessarily created)
            embeddings: An embeddings instance with embed_query/aembed_query and
                embed_documents/aembed_documents methods
        """
        self._index = index
        self.embeddings = embeddings
        self._index_created = False

    @property
    def index(self) -> AsyncSearchIndex:
        """The underlying RedisVL AsyncSearchIndex."""
        return self._index

    async def _ensure_index(self) -> None:
        """Ensure the search index exists in Redis, creating it if needed."""
        if not self._index_created:
            if not await self._index.exists():
                await self._index.create(overwrite=False)
            self._index_created = True

    def _memory_to_data(self, memory: MemoryRecord) -> dict[str, Any]:
        """Convert a MemoryRecord to a data dict for RedisVL storage.

        Uses Unix timestamps for datetime fields and comma-separated strings
        for list fields (topics, entities, extracted_from).

        Args:
            memory: MemoryRecord to convert

        Returns:
            Dictionary suitable for RedisVL index.load()
        """
        created_at_val = memory.created_at.timestamp() if memory.created_at else None
        last_accessed_val = (
            memory.last_accessed.timestamp() if memory.last_accessed else None
        )
        updated_at_val = memory.updated_at.timestamp() if memory.updated_at else None
        persisted_at_val = (
            memory.persisted_at.timestamp() if memory.persisted_at else None
        )
        event_date_val = memory.event_date.timestamp() if memory.event_date else None

        pinned_int = 1 if getattr(memory, "pinned", False) else 0
        access_count_int = int(getattr(memory, "access_count", 0) or 0)

        # Normalize list fields to AMS's canonical comma-separated tag format.
        topics_str = encode_tag_values(memory.topics)
        entities_str = encode_tag_values(memory.entities)
        extracted_from_str = encode_tag_values(memory.extracted_from)

        memory_type_val = (
            memory.memory_type.value
            if hasattr(memory.memory_type, "value")
            else str(memory.memory_type)
        )

        data: dict[str, Any] = {
            "text": memory.text,
            "id_": memory.id,
            "session_id": memory.session_id or "",
            "user_id": memory.user_id or "",
            "namespace": memory.namespace or "",
            "memory_type": memory_type_val,
            "topics": topics_str,
            "entities": entities_str,
            "memory_hash": memory.memory_hash or "",
            "discrete_memory_extracted": memory.discrete_memory_extracted or "f",
            "pinned": pinned_int,
            "access_count": access_count_int,
            "extracted_from": extracted_from_str,
        }

        # Add numeric datetime fields (only if not None, to keep data clean)
        if created_at_val is not None:
            data["created_at"] = created_at_val
        if last_accessed_val is not None:
            data["last_accessed"] = last_accessed_val
        if updated_at_val is not None:
            data["updated_at"] = updated_at_val
        if persisted_at_val is not None:
            data["persisted_at"] = persisted_at_val
        if event_date_val is not None:
            data["event_date"] = event_date_val

        return data

    def _data_to_memory_result(
        self,
        fields: dict[str, Any],
        *,
        dist: float = 0.0,
        score: float | None = None,
        score_type: SearchScoreTypeEnum | None = None,
    ) -> MemoryRecordResult:
        """Convert a search result dict to a MemoryRecordResult.

        Handles parsing of Unix timestamps back to datetime objects,
        comma-separated strings back to lists, and type coercions.

        Args:
            fields: Dictionary of field values from search results
            dist: Legacy distance-like value (0 = best match)
            score: Normalized relevance score for the selected search mode
            score_type: How the normalized score was produced

        Returns:
            MemoryRecordResult with converted data
        """
        if score is not None and dist == 0.0 and score_type is None:
            dist = score

        def parse_timestamp(val: Any) -> datetime | None:
            if val is None:
                return None
            try:
                ts = float(val)
                return datetime.fromtimestamp(ts, tz=UTC)
            except (ValueError, TypeError, OSError):
                if isinstance(val, str):
                    try:
                        return datetime.fromisoformat(val)
                    except ValueError:
                        pass
            return None

        created_at = parse_timestamp(fields.get("created_at"))
        last_accessed = parse_timestamp(fields.get("last_accessed"))
        updated_at = parse_timestamp(fields.get("updated_at"))
        persisted_at = parse_timestamp(fields.get("persisted_at"))
        event_date = parse_timestamp(fields.get("event_date"))

        # Provide defaults for required fields
        if not created_at:
            created_at = datetime.now(UTC)
        if not last_accessed:
            last_accessed = datetime.now(UTC)
        if not updated_at:
            updated_at = datetime.now(UTC)

        # Normalize pinned/access_count from metadata
        pinned_meta = fields.get("pinned", 0)
        try:
            pinned_bool = bool(int(pinned_meta))
        except Exception:
            pinned_bool = bool(pinned_meta)

        access_count_meta = fields.get("access_count", 0)
        try:
            access_count_val = int(access_count_meta or 0)
        except Exception:
            access_count_val = 0

        # Convert empty strings back to None for optional fields
        session_id = fields.get("session_id") or None
        user_id = fields.get("user_id") or None
        namespace = fields.get("namespace") or None

        return MemoryRecordResult(
            text=fields.get("text", ""),
            id=fields.get("id_", ""),
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            created_at=created_at,
            last_accessed=last_accessed,
            updated_at=updated_at,
            pinned=pinned_bool,
            access_count=access_count_val,
            topics=self._parse_list_field(fields.get("topics")),
            entities=self._parse_list_field(fields.get("entities")),
            memory_hash=fields.get("memory_hash", ""),
            discrete_memory_extracted=fields.get("discrete_memory_extracted", "f"),
            memory_type=fields.get("memory_type", "message"),
            persisted_at=persisted_at,
            extracted_from=self._parse_list_field(fields.get("extracted_from")),
            event_date=event_date,
            dist=dist,
            score=score,
            score_type=score_type,
        )

    def _normalize_rank_scores(self, raw_scores: list[float]) -> list[float]:
        """Normalize arbitrary descending rank scores into the 0-1 range."""
        if not raw_scores:
            return []

        max_score = max(raw_scores)
        if max_score <= 0:
            return [1.0 for _ in raw_scores]

        return [min(max(score / max_score, 0.0), 1.0) for score in raw_scores]

    def _build_filter_expression(self, **filter_kwargs: Any) -> Any | None:
        """Build a combined RedisVL filter expression from filter objects.

        Each filter object (e.g., SessionId, Namespace) has a .to_filter()
        method that returns a RedisVL FilterExpression. This method combines
        them with AND logic.

        Args:
            **filter_kwargs: Named filter objects (may be None)

        Returns:
            Combined FilterExpression or None if no filters
        """
        filters = []
        for filter_obj in filter_kwargs.values():
            if filter_obj is not None:
                filters.append(filter_obj.to_filter())

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return reduce(lambda x, y: x & y, filters)

    async def _search_with_recency_aggregation(
        self,
        query: str,
        redis_filter: Any | None,
        limit: int,
        offset: int,
        distance_threshold: float | None,
        recency_params: dict | None,
    ) -> MemoryRecordResults:
        """Perform server-side Redis aggregation search with recency scoring.

        Args:
            query: Search query text
            redis_filter: Redis filter expression
            limit: Maximum number of results
            offset: Offset for pagination
            distance_threshold: Distance threshold for range queries
            recency_params: Parameters for recency scoring

        Returns:
            MemoryRecordResults with server-side scored results

        Raises:
            Exception: If Redis aggregation fails (caller should handle fallback)
        """
        # Embed the query text to vector
        embedding_vector = await self.embeddings.aembed_query(query)

        # Build base KNN or range query
        if distance_threshold is not None:
            knn = RangeQuery(
                vector=embedding_vector,
                vector_field_name="vector",
                filter_expression=redis_filter,
                distance_threshold=float(distance_threshold),
                num_results=limit,
            )
        else:
            knn = VectorQuery(
                vector=embedding_vector,
                vector_field_name="vector",
                filter_expression=redis_filter,
                num_results=limit,
            )

        # Aggregate with APPLY/SORTBY boosted score
        now_ts = int(datetime.now(UTC).timestamp())
        agg = (
            RecencyAggregationQuery.from_vector_query(
                knn, filter_expression=redis_filter
            )
            .load_default_fields()
            .apply_recency(now_ts=now_ts, params=recency_params or {})
            .sort_by_boosted_desc()
            .paginate(offset, limit)
        )

        raw = await self._index.aggregate(agg)

        rows = getattr(raw, "rows", raw) or []
        memory_results: list[MemoryRecordResult] = []
        for row in rows:
            fields = getattr(row, "__dict__", None) or row
            if isinstance(fields, dict):
                pass
            else:
                fields = dict(fields) if fields else {}

            dist = float(fields.get("__vector_score", 1.0) or 1.0)
            memory_results.append(
                self._data_to_memory_result(
                    fields,
                    dist=dist,
                    score=max(0.0, 1.0 - dist),
                    score_type=SearchScoreTypeEnum.SEMANTIC,
                )
            )

        next_offset = offset + limit if len(memory_results) == limit else None
        return MemoryRecordResults(
            memories=memory_results[:limit],
            total=offset + len(memory_results),
            next_offset=next_offset,
        )

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memories using RedisVL's index.load()."""
        if not memories:
            return []

        await self._ensure_index()

        try:
            # Prepare memories with defaults
            for memory in memories:
                if not memory.memory_hash:
                    memory.memory_hash = self.generate_memory_hash(memory)
                now = datetime.now(UTC)
                if not memory.created_at:
                    memory.created_at = now
                if not memory.last_accessed:
                    memory.last_accessed = now
                if not memory.updated_at:
                    memory.updated_at = now

            # Generate embeddings for all texts
            texts = [memory.text for memory in memories]
            embeddings = await self.embeddings.aembed_documents(texts)

            # Build data dicts with embeddings
            data_list = []
            memory_ids = []
            for memory, embedding in zip(memories, embeddings, strict=False):
                data = self._memory_to_data(memory)
                data["vector"] = np.array(embedding, dtype=np.float32).tobytes()
                data_list.append(data)
                memory_ids.append(memory.id)

            # Load into Redis via RedisVL -- use id_field so keys are
            # auto-generated with the index prefix (e.g. "memory_idx:<id>").
            # Do NOT pass explicit keys, as that bypasses the prefix.
            await self._index.load(data_list, id_field="id_")
            return memory_ids

        except Exception as e:
            logger.error(f"Error adding memories to Redis: {e}")
            raise

    async def search_memories(
        self,
        query: str,
        search_mode: SearchModeEnum = SearchModeEnum.SEMANTIC,
        hybrid_alpha: float = 0.7,
        text_scorer: str = "BM25STD",
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        distance_threshold: float | None = None,
        server_side_recency: bool | None = None,
        recency_params: dict | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories using RedisVL semantic, keyword, or hybrid search."""
        await self._ensure_index()

        # Build combined filter expression
        redis_filter = self._build_filter_expression(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            memory_type=memory_type,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            event_date=event_date,
            memory_hash=memory_hash,
            id=id,
            discrete_memory_extracted=discrete_memory_extracted,
        )

        # If server-side recency is requested, attempt aggregation path first.
        # This path currently supports semantic vector search only.
        if server_side_recency and search_mode == SearchModeEnum.SEMANTIC:
            try:
                return await self._search_with_recency_aggregation(
                    query=query,
                    redis_filter=redis_filter,
                    limit=limit,
                    offset=offset,
                    distance_threshold=distance_threshold,
                    recency_params=recency_params,
                )
            except Exception as e:
                logger.warning(
                    f"RedisVL DB-level recency search failed; falling back to standard path: {e}"
                )

        raw_results: list[dict[str, Any]]
        memory_results: list[MemoryRecordResult] = []

        if search_mode == SearchModeEnum.KEYWORD:
            text_query = TextQuery(
                text=query,
                text_field_name="text",
                text_scorer=text_scorer,
                filter_expression=redis_filter,
                return_fields=self.RETURN_FIELDS,
                num_results=limit + offset,
                stopwords=None,
            )
            raw_results = await self._index.query(text_query)
            normalized_scores = self._normalize_rank_scores(
                [float(fields.get("score", 0.0) or 0.0) for fields in raw_results]
            )
            for i, fields in enumerate(raw_results):
                if i < offset:
                    continue

                score = normalized_scores[i]
                memory_results.append(
                    self._data_to_memory_result(
                        fields,
                        dist=max(0.0, 1.0 - score),
                        score=score,
                        score_type=SearchScoreTypeEnum.KEYWORD,
                    )
                )
                if len(memory_results) >= limit:
                    break
        elif search_mode == SearchModeEnum.HYBRID:
            embedding_vector = await self.embeddings.aembed_query(query)
            hybrid_query = AggregateHybridQuery(
                text=query,
                text_field_name="text",
                vector=embedding_vector,
                vector_field_name="vector",
                text_scorer=text_scorer,
                filter_expression=redis_filter,
                alpha=hybrid_alpha,
                num_results=limit + offset,
                return_fields=self.RETURN_FIELDS,
                stopwords=None,
            )
            raw = await self._index.aggregate(hybrid_query, hybrid_query.params)
            raw_results = getattr(raw, "rows", raw) or []

            parsed_rows: list[dict[str, Any]] = []
            for row in raw_results:
                fields = getattr(row, "__dict__", None) or row
                if not isinstance(fields, dict):
                    fields = dict(fields) if fields else {}
                parsed_rows.append(fields)

            normalized_scores = self._normalize_rank_scores(
                [
                    float(fields.get("hybrid_score", 0.0) or 0.0)
                    for fields in parsed_rows
                ]
            )
            for i, fields in enumerate(parsed_rows):
                if i < offset:
                    continue

                score = normalized_scores[i]
                memory_results.append(
                    self._data_to_memory_result(
                        fields,
                        dist=max(0.0, 1.0 - score),
                        score=score,
                        score_type=SearchScoreTypeEnum.HYBRID,
                    )
                )
                if len(memory_results) >= limit:
                    break
        else:
            embedding_vector = await self.embeddings.aembed_query(query)
            if distance_threshold is not None:
                vector_query = RangeQuery(
                    vector=embedding_vector,
                    vector_field_name="vector",
                    filter_expression=redis_filter,
                    distance_threshold=float(distance_threshold),
                    num_results=limit + offset,
                    return_fields=self.RETURN_FIELDS,
                )
            else:
                vector_query = VectorQuery(
                    vector=embedding_vector,
                    vector_field_name="vector",
                    filter_expression=redis_filter,
                    num_results=limit + offset,
                    return_fields=self.RETURN_FIELDS,
                )

            raw_results = await self._index.query(vector_query)
            for i, fields in enumerate(raw_results):
                if i < offset:
                    continue

                dist = float(
                    fields.get("vector_distance", fields.get("__vector_score", 0.0))
                    or 0.0
                )
                memory_results.append(
                    self._data_to_memory_result(
                        fields,
                        dist=dist,
                        score=max(0.0, 1.0 - dist),
                        score_type=SearchScoreTypeEnum.SEMANTIC,
                    )
                )
                if len(memory_results) >= limit:
                    break

        # Client-side recency fallback if server-side was requested
        if server_side_recency:
            memory_results = self._apply_client_side_recency_reranking(
                memory_results, recency_params
            )

        next_offset = offset + limit if len(raw_results) > offset + limit else None

        return MemoryRecordResults(
            memories=memory_results[:limit],
            total=len(raw_results),
            next_offset=next_offset,
        )

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs using RedisVL."""
        if not memory_ids:
            return 0

        await self._ensure_index()

        try:
            return await self._index.drop_documents(memory_ids)
        except Exception as e:
            logger.error(f"Error deleting memories from Redis: {e}")
            raise

    async def update_memories(self, memories: list[MemoryRecord]) -> int:
        """Update memory records by re-adding them (HSET overwrites in Redis)."""
        if not memories:
            return 0

        added = await self.add_memories(memories)
        return len(added)

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories using a filter-only query."""
        try:
            results = await self.list_memories(
                namespace=Namespace(eq=namespace) if namespace else None,
                user_id=UserId(eq=user_id) if user_id else None,
                session_id=SessionId(eq=session_id) if session_id else None,
                limit=10000,  # Large number to get all results
            )
            return results.total
        except Exception as e:
            logger.error(f"Error counting memories: {e}", exc_info=True)
            return 0

    async def list_memories(
        self,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """List memories using filters without semantic search.

        Uses RedisVL FilterQuery for metadata-only filtering without requiring
        an embedding.
        """
        await self._ensure_index()

        try:
            # Build filter expression
            redis_filter = self._build_filter_expression(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                memory_type=memory_type,
                topics=topics,
                entities=entities,
                created_at=created_at,
                last_accessed=last_accessed,
                event_date=event_date,
                memory_hash=memory_hash,
                id=id,
                discrete_memory_extracted=discrete_memory_extracted,
            )

            # Create FilterQuery for non-vector search
            filter_query = FilterQuery(
                filter_expression=redis_filter,
                return_fields=self.RETURN_FIELDS,
                num_results=limit + offset,
            )

            # Execute query using index.query() which properly handles params
            results = await self._index.query(filter_query)

            # Parse results (query() returns List[Dict[str, Any]])
            memory_results: list[MemoryRecordResult] = []
            for fields in results[offset:]:
                memory_result = self._data_to_memory_result(
                    fields,
                    dist=0.0,
                    score=1.0,
                    score_type=None,
                )
                memory_results.append(memory_result)

                if len(memory_results) >= limit:
                    break

            next_offset = offset + limit if len(results) > offset + limit else None

            return MemoryRecordResults(
                memories=memory_results[:limit],
                total=len(results),
                next_offset=next_offset,
            )

        except Exception as e:
            logger.error(
                f"Error listing memories with filter-only query: {e}", exc_info=True
            )
            raise
