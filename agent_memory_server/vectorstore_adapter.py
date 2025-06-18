"""VectorStore adapter for agent memory server.

This module provides an abstraction layer between the agent memory server
and LangChain VectorStore implementations, allowing for pluggable backends.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, TypeVar

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

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
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
)
from agent_memory_server.utils.redis import (
    get_redis_conn,
    get_search_index,
)


logger = logging.getLogger(__name__)

# Type variable for VectorStore implementations
VectorStoreType = TypeVar("VectorStoreType", bound=VectorStore)


class VectorStoreAdapter(ABC):
    """Abstract base class for VectorStore adapters."""

    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    @abstractmethod
    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memory records to the vector store.

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
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        distance_threshold: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories in the vector store.

        Args:
            query: Text query for semantic search
            session_id: Optional session ID filter
            user_id: Optional user ID filter
            namespace: Optional namespace filter
            created_at: Optional created at filter
            last_accessed: Optional last accessed filter
            topics: Optional topics filter
            entities: Optional entities filter
            memory_type: Optional memory type filter
            event_date: Optional event date filter
            distance_threshold: Optional similarity threshold
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

    def memory_to_document(self, memory: MemoryRecord) -> Document:
        """Convert a MemoryRecord to a LangChain Document.

        Args:
            memory: MemoryRecord to convert

        Returns:
            LangChain Document with metadata
        """
        # Use ISO strings for datetime fields (standard format for most backends)
        created_at_val = memory.created_at.isoformat() if memory.created_at else None
        last_accessed_val = (
            memory.last_accessed.isoformat() if memory.last_accessed else None
        )
        updated_at_val = memory.updated_at.isoformat() if memory.updated_at else None
        persisted_at_val = (
            memory.persisted_at.isoformat() if memory.persisted_at else None
        )
        event_date_val = memory.event_date.isoformat() if memory.event_date else None

        metadata = {
            "id_": memory.id,
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "namespace": memory.namespace,
            "created_at": created_at_val,
            "last_accessed": last_accessed_val,
            "updated_at": updated_at_val,
            "topics": memory.topics,
            "entities": memory.entities,
            "memory_hash": memory.memory_hash,
            "discrete_memory_extracted": memory.discrete_memory_extracted,
            "memory_type": memory.memory_type.value,
            "id": memory.id,
            "persisted_at": persisted_at_val,
            "extracted_from": memory.extracted_from,
            "event_date": event_date_val,
        }

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(
            page_content=memory.text,
            metadata=metadata,
        )

    def document_to_memory(
        self, doc: Document, score: float = 0.0
    ) -> MemoryRecordResult:
        """Convert a LangChain Document to a MemoryRecordResult.

        Args:
            doc: LangChain Document to convert
            score: Similarity score for the document

        Returns:
            MemoryRecordResult with converted data
        """
        metadata = doc.metadata

        # Parse datetime values back to datetime objects (handle both timestamp and ISO string formats)
        def parse_datetime(dt_val: str | float | None) -> datetime | None:
            if dt_val is None:
                return None
            if isinstance(dt_val, int | float):
                # Unix timestamp from Redis
                return datetime.fromtimestamp(dt_val, tz=UTC)
            if isinstance(dt_val, str):
                # ISO string from other backends
                return datetime.fromisoformat(dt_val)
            return None

        created_at = parse_datetime(metadata.get("created_at"))
        last_accessed = parse_datetime(metadata.get("last_accessed"))
        updated_at = parse_datetime(metadata.get("updated_at"))
        persisted_at = parse_datetime(metadata.get("persisted_at"))
        event_date = parse_datetime(metadata.get("event_date"))

        # Provide defaults for required fields
        if not created_at:
            created_at = datetime.now(UTC)
        if not last_accessed:
            last_accessed = datetime.now(UTC)
        if not updated_at:
            updated_at = datetime.now(UTC)

        return MemoryRecordResult(
            text=doc.page_content,
            id=metadata.get("id") or metadata.get("id_") or "",
            session_id=metadata.get("session_id"),
            user_id=metadata.get("user_id"),
            namespace=metadata.get("namespace"),
            created_at=created_at,
            last_accessed=last_accessed,
            updated_at=updated_at,
            topics=metadata.get("topics"),
            entities=metadata.get("entities"),
            memory_hash=metadata.get("memory_hash"),
            discrete_memory_extracted=metadata.get("discrete_memory_extracted", "f"),
            memory_type=metadata.get("memory_type", "message"),
            persisted_at=persisted_at,
            extracted_from=metadata.get("extracted_from"),
            event_date=event_date,
            dist=score,
        )

    def generate_memory_hash(self, memory: MemoryRecord) -> str:
        """Generate a stable hash for a memory based on text, user_id, and session_id.

        Args:
            memory: MemoryRecord to hash

        Returns:
            A stable hash string
        """
        text = memory.text
        user_id = memory.user_id or ""
        session_id = memory.session_id or ""

        # Combine the fields in a predictable order
        hash_content = f"{text}|{user_id}|{session_id}"

        # Create a stable hash
        return hashlib.sha256(hash_content.encode()).hexdigest()

    def _convert_filters_to_backend_format(
        self,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        event_date: EventDate | None = None,
    ) -> dict[str, Any] | None:
        """Convert filter objects to standard LangChain dictionary format.

        Uses the PGVector/Pinecone style dictionary format with operators like $eq, $in, etc.
        This works with most standard LangChain VectorStore implementations.

        Backend-specific datetime handling:
        - Pinecone: Uses Unix timestamps (numbers)
        - Others: Use ISO strings

        Args:
            Filter objects from filters.py

        Returns:
            Dictionary filter in format: {"field": {"$eq": "value"}} or None
        """
        filter_dict = {}

        # Determine datetime format based on backend type
        def format_datetime(dt: datetime) -> str | float:
            """Format datetime for the specific backend."""
            vectorstore_type = str(type(self.vectorstore)).lower()

            # Pinecone requires Unix timestamps for datetime comparisons
            if "pinecone" in vectorstore_type:
                logger.info(f"Using Unix timestamp for Pinecone: {dt.timestamp()}")
                return dt.timestamp()
            # Redis might also need timestamps - let's test this
            if "redis" in vectorstore_type:
                logger.info(f"Testing Redis with ISO format: {dt.isoformat()}")
                return dt.isoformat()  # Start with ISO, we'll see if this works
            # Most other backends use ISO strings
            logger.info(f"Using ISO format for {vectorstore_type}: {dt.isoformat()}")
            return dt.isoformat()

        # Simple equality filters
        if session_id and session_id.eq:
            filter_dict["session_id"] = {"$eq": session_id.eq}
        elif session_id and session_id.ne:
            filter_dict["session_id"] = {"$ne": session_id.ne}
        elif session_id and session_id.any:
            filter_dict["session_id"] = {"$in": session_id.any}

        if user_id and user_id.eq:
            filter_dict["user_id"] = {"$eq": user_id.eq}
        elif user_id and user_id.ne:
            filter_dict["user_id"] = {"$ne": user_id.ne}
        elif user_id and user_id.any:
            filter_dict["user_id"] = {"$in": user_id.any}

        if namespace and namespace.eq:
            filter_dict["namespace"] = {"$eq": namespace.eq}
        elif namespace and namespace.ne:
            filter_dict["namespace"] = {"$ne": namespace.ne}
        elif namespace and namespace.any:
            filter_dict["namespace"] = {"$in": namespace.any}

        if memory_type and memory_type.eq:
            filter_dict["memory_type"] = {"$eq": memory_type.eq}
        elif memory_type and memory_type.ne:
            filter_dict["memory_type"] = {"$ne": memory_type.ne}
        elif memory_type and memory_type.any:
            filter_dict["memory_type"] = {"$in": memory_type.any}

        # List filters (topics/entities) - use $in for "any" matches
        if topics and topics.any:
            filter_dict["topics"] = {"$in": topics.any}
        elif topics and topics.eq:
            filter_dict["topics"] = {"$eq": topics.eq}

        if entities and entities.any:
            filter_dict["entities"] = {"$in": entities.any}
        elif entities and entities.eq:
            filter_dict["entities"] = {"$eq": entities.eq}

        # Datetime range filters
        if created_at:
            created_filter = {}
            if created_at.eq:
                created_filter["$eq"] = format_datetime(created_at.eq)
            elif created_at.ne:
                created_filter["$ne"] = format_datetime(created_at.ne)
            elif created_at.gt:
                created_filter["$gt"] = format_datetime(created_at.gt)
            elif created_at.gte:
                created_filter["$gte"] = format_datetime(created_at.gte)
            elif created_at.lt:
                created_filter["$lt"] = format_datetime(created_at.lt)
            elif created_at.lte:
                created_filter["$lte"] = format_datetime(created_at.lte)
            elif created_at.between:
                created_filter["$between"] = [
                    format_datetime(dt) for dt in created_at.between
                ]

            if created_filter:
                filter_dict["created_at"] = created_filter

        if last_accessed:
            last_accessed_filter = {}
            if last_accessed.eq:
                last_accessed_filter["$eq"] = format_datetime(last_accessed.eq)
            elif last_accessed.ne:
                last_accessed_filter["$ne"] = format_datetime(last_accessed.ne)
            elif last_accessed.gt:
                last_accessed_filter["$gt"] = format_datetime(last_accessed.gt)
            elif last_accessed.gte:
                last_accessed_filter["$gte"] = format_datetime(last_accessed.gte)
            elif last_accessed.lt:
                last_accessed_filter["$lt"] = format_datetime(last_accessed.lt)
            elif last_accessed.lte:
                last_accessed_filter["$lte"] = format_datetime(last_accessed.lte)
            elif last_accessed.between:
                last_accessed_filter["$between"] = [
                    format_datetime(dt) for dt in last_accessed.between
                ]

            if last_accessed_filter:
                filter_dict["last_accessed"] = last_accessed_filter

        if event_date:
            event_date_filter = {}
            if event_date.eq:
                event_date_filter["$eq"] = format_datetime(event_date.eq)
            elif event_date.ne:
                event_date_filter["$ne"] = format_datetime(event_date.ne)
            elif event_date.gt:
                event_date_filter["$gt"] = format_datetime(event_date.gt)
            elif event_date.gte:
                event_date_filter["$gte"] = format_datetime(event_date.gte)
            elif event_date.lt:
                event_date_filter["$lt"] = format_datetime(event_date.lt)
            elif event_date.lte:
                event_date_filter["$lte"] = format_datetime(event_date.lte)
            elif event_date.between:
                event_date_filter["$between"] = [
                    format_datetime(dt) for dt in event_date.between
                ]

            if event_date_filter:
                filter_dict["event_date"] = event_date_filter

        logger.debug(f"Converted to LangChain filter format: {filter_dict}")
        return filter_dict if filter_dict else None


class LangChainVectorStoreAdapter(VectorStoreAdapter):
    """Generic adapter for any LangChain VectorStore implementation."""

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memory records to the vector store."""
        if not memories:
            return []

        # Convert MemoryRecords to Documents
        documents = []
        for memory in memories:
            # Generate hash if not provided
            if not memory.memory_hash:
                memory.memory_hash = self.generate_memory_hash(memory)

            doc = self.memory_to_document(memory)
            logger.info(
                f"Converting memory to document: {memory.id} -> metadata: {doc.metadata}"
            )
            documents.append(doc)

        # Add documents to the vector store
        try:
            # Extract IDs from memory records to prevent ULID generation
            memory_ids = [memory.id for memory in memories]

            # Standard LangChain VectorStore implementation
            if hasattr(self.vectorstore, "aadd_documents"):
                ids = await self.vectorstore.aadd_documents(documents, ids=memory_ids)
            elif hasattr(self.vectorstore, "add_documents"):
                ids = self.vectorstore.add_documents(documents, ids=memory_ids)
            else:
                # Fallback to add_texts
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                if hasattr(self.vectorstore, "aadd_texts"):
                    ids = await self.vectorstore.aadd_texts(
                        texts, metadatas=metadatas, ids=memory_ids
                    )
                else:
                    ids = self.vectorstore.add_texts(
                        texts, metadatas=metadatas, ids=memory_ids
                    )

            return ids or memory_ids
        except Exception as e:
            logger.error(f"Error adding memories to vector store: {e}")
            raise

    async def search_memories(
        self,
        query: str,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        distance_threshold: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories in the vector store."""
        try:
            # Convert filter objects to standard LangChain dictionary format
            backend_filter = self._convert_filters_to_backend_format(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                topics=topics,
                entities=entities,
                memory_type=memory_type,
                created_at=created_at,
                last_accessed=last_accessed,
                event_date=event_date,
            )

            # Prepare search arguments
            search_kwargs: dict[str, Any] = {
                "k": limit + offset
            }  # Get more results for offset handling

            if backend_filter:
                search_kwargs["filter"] = backend_filter
                logger.info(f"Applied LangChain filter: {backend_filter}")
            else:
                logger.info("No filters to apply")

            if hasattr(self.vectorstore, "asimilarity_search_with_score"):
                docs_with_scores = await self.vectorstore.asimilarity_search_with_score(
                    query, **search_kwargs
                )
            elif hasattr(self.vectorstore, "similarity_search_with_score"):
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query, **search_kwargs
                )
            else:
                # Fallback without scores
                docs = (
                    await self.vectorstore.asimilarity_search(query, **search_kwargs)
                    if hasattr(self.vectorstore, "asimilarity_search")
                    else self.vectorstore.similarity_search(query, **search_kwargs)
                )
                docs_with_scores = [(doc, 0.0) for doc in docs]

            # Apply additional filters that couldn't be handled by the vectorstore
            filtered_results = []

            for doc, score in docs_with_scores:
                # Apply distance threshold
                if distance_threshold is not None and score > distance_threshold:
                    continue

                # Apply complex filters
                if not self._matches_filters(
                    doc,
                    session_id,
                    user_id,
                    namespace,
                    topics,
                    entities,
                    memory_type,
                    created_at,
                    last_accessed,
                    event_date,
                ):
                    continue

                filtered_results.append((doc, score))

            # Apply offset and limit
            start_idx = offset
            end_idx = offset + limit
            paginated_results = filtered_results[start_idx:end_idx]

            # Convert to MemoryRecordResults
            memory_results = []
            for doc, score in paginated_results:
                memory_result = self.document_to_memory(doc, score)
                memory_results.append(memory_result)

            next_offset = offset + limit if len(filtered_results) > end_idx else None

            return MemoryRecordResults(
                memories=memory_results,
                total=len(filtered_results),
                next_offset=next_offset,
            )

        except Exception as e:
            logger.error(f"Error searching memories in vector store: {e}")
            raise

    def _matches_filters(
        self,
        doc: Document,
        session_id: SessionId | None,
        user_id: UserId | None,
        namespace: Namespace | None,
        topics: Topics | None,
        entities: Entities | None,
        memory_type: MemoryType | None,
        created_at: CreatedAt | None,
        last_accessed: LastAccessed | None,
        event_date: EventDate | None,
    ) -> bool:
        """Check if a document matches the given filters."""
        metadata = doc.metadata

        # Check session_id filter
        if session_id and session_id.eq:
            doc_session_id = metadata.get("session_id")
            if doc_session_id != session_id.eq:
                return False

        # Check user_id filter
        if user_id and user_id.eq:
            doc_user_id = metadata.get("user_id")
            if doc_user_id != user_id.eq:
                return False

        # Check namespace filter
        if namespace and namespace.eq:
            doc_namespace = metadata.get("namespace")
            if doc_namespace != namespace.eq:
                return False

        # Check memory_type filter
        if memory_type and memory_type.eq:
            doc_memory_type = metadata.get("memory_type")
            if doc_memory_type != memory_type.eq:
                return False

        # Check topics filter
        if topics and topics.any:
            doc_topics = metadata.get("topics", [])
            if isinstance(doc_topics, str):
                doc_topics = doc_topics.split(",") if doc_topics else []
            if not any(topic in doc_topics for topic in topics.any):
                return False

        # Check entities filter
        if entities and entities.any:
            doc_entities = metadata.get("entities", [])
            if isinstance(doc_entities, str):
                doc_entities = doc_entities.split(",") if doc_entities else []
            if not any(entity in doc_entities for entity in entities.any):
                return False

        # TODO: Add datetime range filters for created_at, last_accessed, event_date
        # This would require parsing the datetime strings in metadata and comparing

        return True

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs."""
        if not memory_ids:
            return 0

        try:
            if hasattr(self.vectorstore, "adelete"):
                deleted = await self.vectorstore.adelete(memory_ids)
            elif hasattr(self.vectorstore, "delete"):
                deleted = self.vectorstore.delete(memory_ids)
            else:
                logger.warning("Vector store does not support delete operation")
                return 0

            return len(memory_ids) if deleted else 0

        except Exception as e:
            logger.error(f"Error deleting memories from vector store: {e}")
            raise

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories in the vector store using LangChain."""
        try:
            # Convert basic filters to our filter objects, then to backend format
            from agent_memory_server.filters import Namespace, SessionId, UserId

            namespace_filter = Namespace(eq=namespace) if namespace else None
            user_id_filter = UserId(eq=user_id) if user_id else None
            session_id_filter = SessionId(eq=session_id) if session_id else None

            # Most vector stores don't have a direct count method
            # We'll use a large similarity search and count results
            # This is not optimal but works as a fallback
            search_kwargs: dict[str, Any] = {
                "k": 10000
            }  # Large number to get all results

            # Apply filters using the proper method signature
            backend_filter = self._convert_filters_to_backend_format(
                namespace=namespace_filter,
                user_id=user_id_filter,
                session_id=session_id_filter,
            )
            if backend_filter:
                search_kwargs["filter"] = backend_filter

            if hasattr(self.vectorstore, "asimilarity_search"):
                docs = await self.vectorstore.asimilarity_search("", **search_kwargs)
            elif hasattr(self.vectorstore, "similarity_search"):
                docs = self.vectorstore.similarity_search("", **search_kwargs)
            else:
                logger.warning("Vector store does not support similarity_search")
                return 0

            # Apply post-processing filters
            if namespace or user_id or session_id:
                filtered_docs = []
                for doc in docs:
                    metadata = doc.metadata
                    matches = True

                    if namespace and metadata.get("namespace") != namespace:
                        matches = False
                    if user_id and metadata.get("user_id") != user_id:
                        matches = False
                    if session_id and metadata.get("session_id") != session_id:
                        matches = False

                    if matches:
                        filtered_docs.append(doc)

                return len(filtered_docs)
            return len(docs)

        except Exception as e:
            logger.error(f"Error counting memories in vector store: {e}")
            return 0


class RedisVectorStoreAdapter(VectorStoreAdapter):
    """Redis adapter that uses LangChain's RedisVectorStore with Redis-specific optimizations."""

    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings):
        """Initialize Redis adapter.

        Args:
            vectorstore: VectorStore instance (not used, only for interface compatibility)
            embeddings: Embeddings instance
        """
        super().__init__(vectorstore, embeddings)

        # Note: We don't use the vectorstore parameter since we use pure RedisVL
        # The vectorstore is only kept for interface compatibility

    def memory_to_document(self, memory: MemoryRecord) -> Document:
        """Convert a MemoryRecord to a LangChain Document with Redis timestamp format.

        Args:
            memory: MemoryRecord to convert

        Returns:
            LangChain Document with metadata optimized for Redis
        """
        # For Redis backends, use Unix timestamps for NUMERIC fields
        created_at_val = memory.created_at.timestamp() if memory.created_at else None
        last_accessed_val = (
            memory.last_accessed.timestamp() if memory.last_accessed else None
        )
        updated_at_val = memory.updated_at.timestamp() if memory.updated_at else None
        persisted_at_val = (
            memory.persisted_at.timestamp() if memory.persisted_at else None
        )
        event_date_val = memory.event_date.timestamp() if memory.event_date else None

        metadata = {
            "id_": memory.id,
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "namespace": memory.namespace,
            "created_at": created_at_val,
            "last_accessed": last_accessed_val,
            "updated_at": updated_at_val,
            "topics": memory.topics,
            "entities": memory.entities,
            "memory_hash": memory.memory_hash,
            "discrete_memory_extracted": memory.discrete_memory_extracted,
            "memory_type": memory.memory_type.value,
            "id": memory.id,
            "persisted_at": persisted_at_val,
            "extracted_from": memory.extracted_from,
            "event_date": event_date_val,
        }

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(
            page_content=memory.text,
            metadata=metadata,
        )

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memories using pure RedisVL to ensure proper data format."""
        if not memories:
            return []

        try:
            # Get Redis connection and search index
            redis_client = await get_redis_conn()
            index = get_search_index(redis_client)

            # Convert memories to RedisVL format
            data = []
            memory_ids = []

            for memory in memories:
                # Generate embeddings for the memory text
                if hasattr(self.embeddings, "aembed_documents"):
                    embeddings_result = await self.embeddings.aembed_documents(
                        [memory.text]
                    )
                    vector = embeddings_result[0]
                else:
                    vector = await self.embeddings.aembed_query(memory.text)

                # Set memory hash if not provided
                if not memory.memory_hash:
                    memory.memory_hash = self.generate_memory_hash(memory)

                # Ensure timestamps are set - create datetime objects if they don't exist
                now_timestamp = datetime.now(UTC)
                if not memory.created_at:
                    memory.created_at = now_timestamp
                if not memory.last_accessed:
                    memory.last_accessed = now_timestamp
                if not memory.updated_at:
                    memory.updated_at = now_timestamp

                # Helper function to convert datetime to timestamp (returns None for None input)
                def to_timestamp(dt_value):
                    if dt_value is None:
                        return None
                    if isinstance(dt_value, datetime):
                        return dt_value.timestamp()
                    if isinstance(dt_value, int | float):
                        return dt_value
                    return None

                # Create memory data dict for RedisVL
                memory_data = {
                    "text": memory.text,
                    "id_": memory.id or "",
                    "id": memory.id or "",  # Keep both for compatibility
                    "session_id": memory.session_id or "",
                    "user_id": memory.user_id or "",
                    "namespace": memory.namespace or "",
                    "topics": ",".join(memory.topics) if memory.topics else "",
                    "entities": ",".join(memory.entities) if memory.entities else "",
                    "memory_type": memory.memory_type.value
                    if memory.memory_type
                    else "message",
                    "created_at": to_timestamp(memory.created_at),
                    "last_accessed": to_timestamp(memory.last_accessed),
                    "updated_at": to_timestamp(memory.updated_at),
                    "memory_hash": memory.memory_hash,
                    "extracted_from": ",".join(memory.extracted_from)
                    if memory.extracted_from
                    else "",
                    "discrete_memory_extracted": memory.discrete_memory_extracted
                    or "f",
                    "vector": np.array(vector, dtype=np.float32).tobytes(),
                }

                # Add optional datetime fields only if they have values (avoid RedisSearch NUMERIC field errors)
                if memory.persisted_at is not None:
                    memory_data["persisted_at"] = to_timestamp(memory.persisted_at)
                if memory.event_date is not None:
                    memory_data["event_date"] = to_timestamp(memory.event_date)

                # Use memory.id as the key, or generate a new one if not provided
                memory_key = memory.id or f"memory:{memory.memory_hash}"
                memory_ids.append(memory_key)

                # RedisVL expects a dictionary with the key included, not a tuple
                memory_data["key"] = memory_key
                data.append(memory_data)

            # Load data into RedisVL index with manual keys
            # Remove the 'key' field we added earlier since we're using the keys parameter
            for memory_data in data:
                if "key" in memory_data:
                    del memory_data["key"]

            # Add the index prefix to keys to match the schema expectation
            # RedisVL expects keys to have the prefix that matches the schema
            prefixed_keys = [f"{index.schema.index.prefix}{key}" for key in memory_ids]

            await index.load(data, keys=prefixed_keys)

            return memory_ids

        except Exception as e:
            logger.error(f"Error adding memories to Redis vectorstore: {e}")
            raise

    async def search_memories(
        self,
        query: str,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        distance_threshold: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories using pure RedisVL instead of LangChain Redis to avoid field conflicts."""
        try:
            from redisvl.query import VectorQuery

            # Build RedisVL FilterExpression using existing filter classes
            filters = []

            # Add individual filters using the .to_filter() methods from filters.py
            if session_id:
                filters.append(session_id.to_filter())
            if user_id:
                filters.append(user_id.to_filter())
            if namespace:
                filters.append(namespace.to_filter())
            if memory_type:
                filters.append(memory_type.to_filter())
            if topics:
                filters.append(topics.to_filter())
            if entities:
                filters.append(entities.to_filter())
            if created_at:
                filters.append(created_at.to_filter())
            if last_accessed:
                filters.append(last_accessed.to_filter())
            if event_date:
                filters.append(event_date.to_filter())

            # Combine filters with AND logic
            redis_filter = None
            if filters:
                if len(filters) == 1:
                    redis_filter = filters[0]
                else:
                    from functools import reduce

                    redis_filter = reduce(lambda x, y: x & y, filters)

            # Get Redis connection and search index
            redis_client = await get_redis_conn()
            index = get_search_index(redis_client)

            # Generate query vector using embeddings
            query_vector = await self.embeddings.aembed_query(query)

            # Create RedisVL vector query
            vector_query = VectorQuery(
                vector=query_vector,
                vector_field_name="vector",
                return_fields=[
                    "id_",
                    "text",
                    "session_id",
                    "user_id",
                    "namespace",
                    "topics",
                    "entities",
                    "memory_type",
                    "created_at",
                    "last_accessed",
                    "updated_at",
                    "persisted_at",
                    "event_date",
                    "memory_hash",
                    "extracted_from",
                    "discrete_memory_extracted",
                    "id",
                ],
                num_results=limit + offset,
            )

            if redis_filter:
                vector_query.set_filter(redis_filter)

            # Execute the query
            search_results = await index.query(vector_query)

            # Convert results to MemoryRecordResult objects
            memory_results = []
            for i, result in enumerate(search_results):
                # Apply offset
                if i < offset:
                    continue

                # Extract fields from RedisVL result
                result_dict = result.__dict__ if hasattr(result, "__dict__") else result

                # Calculate distance score
                score = float(result_dict.get("vector_score", 0.0))

                # Apply distance threshold
                if distance_threshold is not None and score > distance_threshold:
                    continue

                # Helper function to parse timestamp to datetime
                def parse_timestamp_to_datetime(timestamp_val):
                    if not timestamp_val:
                        return datetime.now(UTC)
                    if isinstance(timestamp_val, int | float):
                        return datetime.fromtimestamp(timestamp_val, tz=UTC)
                    return datetime.now(UTC)

                # Extract memory data
                memory_result = MemoryRecordResult(
                    id=result_dict.get("id_", ""),
                    text=result_dict.get("text", ""),
                    dist=score,
                    created_at=parse_timestamp_to_datetime(
                        result_dict.get("created_at")
                    ),
                    updated_at=parse_timestamp_to_datetime(
                        result_dict.get("updated_at")
                    ),
                    last_accessed=parse_timestamp_to_datetime(
                        result_dict.get("last_accessed")
                    ),
                    user_id=result_dict.get("user_id"),
                    session_id=result_dict.get("session_id"),
                    namespace=result_dict.get("namespace"),
                    topics=self._parse_list_field(result_dict.get("topics")),
                    entities=self._parse_list_field(result_dict.get("entities")),
                    memory_hash=result_dict.get("memory_hash", ""),
                    memory_type=result_dict.get("memory_type", "message"),
                    persisted_at=result_dict.get("persisted_at"),
                    extracted_from=self._parse_list_field(
                        result_dict.get("extracted_from")
                    ),
                    event_date=result_dict.get("event_date"),
                )

                memory_results.append(memory_result)

                # Stop if we have enough results
                if len(memory_results) >= limit:
                    break

            next_offset = (
                offset + limit if len(search_results) > offset + limit else None
            )

            return MemoryRecordResults(
                memories=memory_results,
                total=len(search_results),
                next_offset=next_offset,
            )

        except Exception as e:
            logger.error(f"Error searching memories in Redis vectorstore: {e}")
            raise

    def _parse_list_field(self, field_value):
        """Parse a field that might be a list, comma-separated string, or None."""
        if not field_value:
            return []
        if isinstance(field_value, list):
            return field_value
        if isinstance(field_value, str):
            return field_value.split(",") if field_value else []
        return []

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs using LangChain's RedisVectorStore."""
        if not memory_ids:
            return 0

        try:
            if hasattr(self.vectorstore, "adelete"):
                deleted = await self.vectorstore.adelete(memory_ids)
            elif hasattr(self.vectorstore, "delete"):
                deleted = self.vectorstore.delete(memory_ids)
            else:
                logger.warning("Redis vectorstore does not support delete operation")
                return 0

            return len(memory_ids) if deleted else 0

        except Exception as e:
            logger.error(f"Error deleting memories from Redis vectorstore: {e}")
            raise

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories using pure RedisVL instead of LangChain Redis to avoid field conflicts."""
        try:
            from redisvl.query import CountQuery

            # Build RedisVL filter for counting using filter objects
            filters = []

            if namespace:
                from agent_memory_server.filters import Namespace

                namespace_filter = Namespace(eq=namespace).to_filter()
                filters.append(namespace_filter)
                logger.info(
                    f"Added namespace filter: {namespace_filter} for value: {namespace}"
                )
            if user_id:
                from agent_memory_server.filters import UserId

                user_filter = UserId(eq=user_id).to_filter()
                filters.append(user_filter)
                logger.info(f"Added user_id filter: {user_filter} for value: {user_id}")
            if session_id:
                from agent_memory_server.filters import SessionId

                session_filter = SessionId(eq=session_id).to_filter()
                filters.append(session_filter)
                logger.info(
                    f"Added session_id filter: {session_filter} for value: {session_id}"
                )

            # Combine filters
            redis_filter = None
            if filters:
                if len(filters) == 1:
                    redis_filter = filters[0]
                else:
                    from functools import reduce

                    redis_filter = reduce(lambda x, y: x & y, filters)
                logger.info(f"Combined RedisVL filter: {redis_filter}")

            # Get Redis connection and search index
            redis_client = await get_redis_conn()
            index = get_search_index(redis_client)

            # Create RedisVL count query
            count_query = CountQuery()
            if redis_filter:
                count_query.set_filter(redis_filter)

            # Execute the count query
            result = await index.query(count_query)
            logger.info(f"CountQuery result: {result}, type: {type(result)}")

            # Also try without filters to see if data is indexed at all
            if redis_filter:
                unfiltered_query = CountQuery()
                unfiltered_result = await index.query(unfiltered_query)
                logger.info(f"Unfiltered CountQuery result: {unfiltered_result}")

            # CountQuery returns an integer directly
            count = result if isinstance(result, int) else getattr(result, "total", 0)
            logger.info(f"Final count: {count}")
            return count

        except Exception as e:
            logger.error(f"Error counting memories in Redis vectorstore: {e}")
            return 0
