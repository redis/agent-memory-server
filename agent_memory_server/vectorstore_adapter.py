"""VectorStore adapter for agent memory server.

This module provides an abstraction layer between the agent memory server
and LangChain VectorStore implementations, allowing for pluggable backends.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, TypeVar

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
        # Convert datetime objects to ISO strings for metadata
        created_at_str = memory.created_at.isoformat() if memory.created_at else None
        last_accessed_str = (
            memory.last_accessed.isoformat() if memory.last_accessed else None
        )
        updated_at_str = memory.updated_at.isoformat() if memory.updated_at else None
        persisted_at_str = (
            memory.persisted_at.isoformat() if memory.persisted_at else None
        )
        event_date_str = memory.event_date.isoformat() if memory.event_date else None

        metadata = {
            "id_": memory.id_,
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "namespace": memory.namespace,
            "created_at": created_at_str,
            "last_accessed": last_accessed_str,
            "updated_at": updated_at_str,
            "topics": memory.topics,
            "entities": memory.entities,
            "memory_hash": memory.memory_hash,
            "discrete_memory_extracted": memory.discrete_memory_extracted,
            "memory_type": memory.memory_type.value,
            "id": memory.id,
            "persisted_at": persisted_at_str,
            "extracted_from": memory.extracted_from,
            "event_date": event_date_str,
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

        # Parse datetime strings back to datetime objects
        def parse_datetime(dt_str: str | None) -> datetime | None:
            if dt_str:
                return datetime.fromisoformat(dt_str)
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
            id_=metadata.get("id_"),
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
            id=metadata.get("id"),
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
        self, filter_dict: dict[str, Any] | None
    ) -> Any:
        """Convert standard filter dictionary to backend-specific format.

        For most LangChain VectorStores, filtering capabilities vary significantly.
        This method provides a basic filter format that works with common backends.
        Complex filtering is handled via post-processing.
        """
        if not filter_dict:
            return None

        logger.debug(f"Converting filters for non-Redis backend: {filter_dict}")

        # Most LangChain VectorStores use simple key-value metadata filtering
        # For complex filters (lists, ranges), we rely on post-processing
        simple_filters = {}

        for field, value in filter_dict.items():
            if field in ["session_id", "user_id", "namespace", "memory_type"] and value:
                simple_filters[field] = value
            # Skip complex filters like topics/entities lists - handle in post-processing

        return simple_filters if simple_filters else None


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

            documents.append(self.memory_to_document(memory))

        # Add documents to the vector store
        try:
            # Most VectorStores support add_documents
            if hasattr(self.vectorstore, "aadd_documents"):
                ids = await self.vectorstore.aadd_documents(documents)
            elif hasattr(self.vectorstore, "add_documents"):
                ids = self.vectorstore.add_documents(documents)
            else:
                # Fallback to add_texts if add_documents not available
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                if hasattr(self.vectorstore, "aadd_texts"):
                    ids = await self.vectorstore.aadd_texts(texts, metadatas=metadatas)
                else:
                    ids = self.vectorstore.add_texts(texts, metadatas=metadatas)

            return ids or []
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
            # Build filter metadata based on provided filters
            filter_dict = {}

            if session_id and session_id.eq:
                filter_dict["session_id"] = session_id.eq
            if user_id and user_id.eq:
                filter_dict["user_id"] = user_id.eq
            if namespace and namespace.eq:
                filter_dict["namespace"] = namespace.eq
            if memory_type and memory_type.eq:
                filter_dict["memory_type"] = memory_type.eq

            # Handle topics and entities filters
            if topics:
                if topics.any:
                    # For 'any' filters, we'll search without filter and post-process
                    # since not all vectorstores support complex list filtering
                    pass
                elif topics.eq:
                    filter_dict["topics"] = topics.eq

            if entities:
                if entities.any:
                    # Similar to topics, handle in post-processing
                    pass
                elif entities.eq:
                    filter_dict["entities"] = entities.eq

            # For non-Redis backends, use simple metadata filtering where supported
            search_kwargs = {
                "k": limit + offset
            }  # Get more results for offset handling

            # Apply basic filters that the backend supports
            if filter_dict:
                backend_filter = self._convert_filters_to_backend_format(filter_dict)
                if backend_filter:
                    search_kwargs["filter"] = backend_filter
                    logger.debug(f"Applied backend filter: {backend_filter}")
                else:
                    logger.debug(
                        "No backend filters applied - using post-processing only"
                    )
            else:
                logger.debug("No filters to apply")

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
        """Count memories matching the given filters."""
        try:
            # Build filter
            filter_dict = {}
            if namespace:
                filter_dict["namespace"] = namespace
            if user_id:
                filter_dict["user_id"] = user_id
            if session_id:
                filter_dict["session_id"] = session_id

            # Most vector stores don't have a direct count method
            # We'll use a large similarity search and count results
            # This is not optimal but works as a fallback
            search_kwargs = {"k": 10000}  # Large number to get all results

            # Apply basic filters where supported by the backend
            if filter_dict:
                backend_filter = self._convert_filters_to_backend_format(filter_dict)
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
            if filter_dict:
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
    """Custom Redis adapter that uses proper indexing for server-side filtering."""

    def __init__(self, embeddings: Embeddings, redis_client=None):
        """Initialize Redis adapter with proper indexing.

        Args:
            embeddings: Embeddings instance
            redis_client: Optional Redis client (will create if None)
        """
        # Don't call super().__init__ since we manage our own Redis connection
        self.embeddings = embeddings
        self.redis_client = redis_client
        self._index = None

    async def _get_index(self):
        """Get the Redis search index with proper schema."""
        if self._index is None:
            from agent_memory_server.utils.redis import get_redis_conn, get_search_index

            if self.redis_client is None:
                self.redis_client = await get_redis_conn()

            self._index = get_search_index(self.redis_client)

            # Ensure the index exists
            from agent_memory_server.utils.redis import ensure_search_index_exists

            await ensure_search_index_exists(self.redis_client)

        return self._index

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memory records using Redis with proper indexing."""
        if not memories:
            return []

        try:
            # Ensure redis client is available
            if self.redis_client is None:
                from agent_memory_server.utils.redis import get_redis_conn

                self.redis_client = await get_redis_conn()

            # Use the actual Redis implementation
            from agent_memory_server.long_term_memory import index_long_term_memories

            # Call the actual Redis implementation with proper indexing
            await index_long_term_memories(
                memories=memories,
                redis_client=self.redis_client,
                deduplicate=False,  # Deduplication handled separately if needed
            )

            # Return the memory IDs, ensuring all are strings and filtering out None values
            result_ids = []
            for memory in memories:
                memory_id = memory.id_ or memory.id
                if memory_id is not None:
                    result_ids.append(str(memory_id))

            return result_ids

        except Exception as e:
            logger.error(f"Error adding memories to Redis: {e}")
            return []

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
        """Search memories using Redis with proper server-side filtering."""
        from datetime import datetime
        from functools import reduce

        from redisvl.query import VectorQuery, VectorRangeQuery
        from redisvl.utils.vectorize import OpenAITextVectorizer

        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults
        from agent_memory_server.utils.redis import safe_get

        try:
            # Ensure redis client is available
            if self.redis_client is None:
                from agent_memory_server.utils.redis import get_redis_conn

                self.redis_client = await get_redis_conn()

            # Get search index
            index = await self._get_index()

            # Create vector embedding for the query
            vectorizer = OpenAITextVectorizer()
            vector = await vectorizer.aembed(query)

            # Build filters using the Redis filter syntax
            filters = []
            if session_id:
                filters.append(session_id.to_filter())
            if user_id:
                filters.append(user_id.to_filter())
            if namespace:
                filters.append(namespace.to_filter())
            if created_at:
                filters.append(created_at.to_filter())
            if last_accessed:
                filters.append(last_accessed.to_filter())
            if topics:
                filters.append(topics.to_filter())
            if entities:
                filters.append(entities.to_filter())
            if memory_type:
                filters.append(memory_type.to_filter())
            if event_date:
                filters.append(event_date.to_filter())

            filter_expression = reduce(lambda x, y: x & y, filters) if filters else None

            # Create appropriate query based on distance threshold
            if distance_threshold is not None:
                q = VectorRangeQuery(
                    vector=vector,
                    vector_field_name="vector",
                    distance_threshold=distance_threshold,
                    num_results=limit,
                    return_score=True,
                    return_fields=[
                        "text",
                        "id_",
                        "dist",
                        "created_at",
                        "last_accessed",
                        "user_id",
                        "session_id",
                        "namespace",
                        "topics",
                        "entities",
                        "memory_type",
                        "memory_hash",
                        "id",
                        "persisted_at",
                        "extracted_from",
                        "event_date",
                    ],
                )
            else:
                q = VectorQuery(
                    vector=vector,
                    vector_field_name="vector",
                    num_results=limit,
                    return_score=True,
                    return_fields=[
                        "text",
                        "id_",
                        "dist",
                        "created_at",
                        "last_accessed",
                        "user_id",
                        "session_id",
                        "namespace",
                        "topics",
                        "entities",
                        "memory_type",
                        "memory_hash",
                        "id",
                        "persisted_at",
                        "extracted_from",
                        "event_date",
                    ],
                )

            if filter_expression:
                q.set_filter(filter_expression)

            q.paging(offset=offset, num=limit)

            # Execute the search
            search_result = await index.query(q)

            # Process results
            results = []
            memory_hashes = []

            for doc in search_result:
                # Skip duplicate hashes
                memory_hash = safe_get(doc, "memory_hash")
                if memory_hash in memory_hashes:
                    continue
                memory_hashes.append(memory_hash)

                # Parse topics and entities from comma-separated strings
                doc_topics = safe_get(doc, "topics", [])
                if isinstance(doc_topics, str):
                    doc_topics = doc_topics.split(",") if doc_topics else []

                doc_entities = safe_get(doc, "entities", [])
                if isinstance(doc_entities, str):
                    doc_entities = doc_entities.split(",") if doc_entities else []

                # Handle extracted_from field
                doc_extracted_from = safe_get(doc, "extracted_from", [])
                if isinstance(doc_extracted_from, str) and doc_extracted_from:
                    doc_extracted_from = doc_extracted_from.split(",")
                elif not doc_extracted_from:
                    doc_extracted_from = []

                # Handle event_date field
                doc_event_date = safe_get(doc, "event_date", 0)
                parsed_event_date = None
                if doc_event_date and int(doc_event_date) != 0:
                    parsed_event_date = datetime.fromtimestamp(int(doc_event_date))

                # Convert to MemoryRecordResult
                result = MemoryRecordResult(
                    id_=safe_get(doc, "id_"),
                    text=safe_get(doc, "text", ""),
                    dist=float(safe_get(doc, "vector_distance", 0)),
                    created_at=datetime.fromtimestamp(
                        int(safe_get(doc, "created_at", 0))
                    ),
                    updated_at=datetime.fromtimestamp(
                        int(safe_get(doc, "updated_at", 0))
                    ),
                    last_accessed=datetime.fromtimestamp(
                        int(safe_get(doc, "last_accessed", 0))
                    ),
                    user_id=safe_get(doc, "user_id"),
                    session_id=safe_get(doc, "session_id"),
                    namespace=safe_get(doc, "namespace"),
                    topics=doc_topics,
                    entities=doc_entities,
                    memory_hash=memory_hash,
                    memory_type=safe_get(doc, "memory_type", "message"),
                    id=safe_get(doc, "id"),
                    persisted_at=datetime.fromtimestamp(
                        int(safe_get(doc, "persisted_at", 0))
                    )
                    if safe_get(doc, "persisted_at", 0) != 0
                    else None,
                    extracted_from=doc_extracted_from,
                    event_date=parsed_event_date,
                )
                results.append(result)

            # Calculate total results
            total_results = len(results)
            try:
                # Check if search_result has a total attribute and use it
                total_attr = getattr(search_result, "total", None)
                if total_attr is not None:
                    total_results = int(total_attr)
            except (AttributeError, TypeError):
                # Fallback to list length if search_result is a list or doesn't have total
                total_results = (
                    len(search_result)
                    if isinstance(search_result, list)
                    else len(results)
                )

            logger.info(f"Found {len(results)} results for query")
            return MemoryRecordResults(
                total=total_results,
                memories=results,
                next_offset=offset + limit if offset + limit < total_results else None,
            )

        except Exception as e:
            logger.error(f"Error searching memories in Redis: {e}")
            # Return empty results on error
            return MemoryRecordResults(total=0, memories=[], next_offset=None)

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs using proper Redis key construction."""
        if not memory_ids:
            return 0

        try:
            from agent_memory_server.utils.keys import Keys

            if self.redis_client is None:
                from agent_memory_server.utils.redis import get_redis_conn

                self.redis_client = await get_redis_conn()

            deleted_count = 0

            # First, try to search for existing memories to get the proper keys and namespaces
            for memory_id in memory_ids:
                # Search for the memory to find its namespace
                try:
                    # Use a direct Redis FT.SEARCH to find the memory
                    index_name = Keys.search_index_name()
                    search_query = f"FT.SEARCH {index_name} (@id:{{{memory_id}}}) RETURN 3 id_ namespace"

                    search_results = await self.redis_client.execute_command(
                        search_query
                    )

                    if search_results and search_results[0] > 0:
                        # Found the memory, get its key and namespace
                        memory_key = search_results[1]
                        if isinstance(memory_key, bytes):
                            memory_key = memory_key.decode()

                        # Delete using the exact key returned by search
                        if await self.redis_client.delete(memory_key):
                            deleted_count += 1
                            logger.info(
                                f"Deleted memory {memory_id} with key {memory_key}"
                            )
                        continue

                except Exception as e:
                    logger.warning(f"Could not search for memory {memory_id}: {e}")

                # Fallback: try different possible key formats
                possible_keys = [
                    Keys.memory_key(memory_id, None),  # No namespace
                    f"memory:{memory_id}",
                    memory_id,  # Direct key
                ]

                # Also try with common namespaces if they exist
                for namespace in [None, "default", ""]:
                    if namespace:
                        possible_keys.append(Keys.memory_key(memory_id, namespace))

                for key in possible_keys:
                    try:
                        if await self.redis_client.delete(key):
                            deleted_count += 1
                            logger.info(
                                f"Deleted memory {memory_id} with fallback key {key}"
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Failed to delete key {key}: {e}")

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting memories from Redis: {e}")
            return 0

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories using Redis with proper filtering."""
        try:
            # Use the original Redis count logic
            from agent_memory_server.long_term_memory import count_long_term_memories

            # Use the correct parameter types - pass strings directly
            return await count_long_term_memories(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                redis_client=self.redis_client,
            )

        except Exception as e:
            logger.error(f"Error counting memories in Redis: {e}")
            return 0
