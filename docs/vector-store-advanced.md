# Advanced Vector Store Configuration

This guide covers advanced configuration patterns, performance optimization, custom implementations, and migration strategies for vector store backends in Redis Agent Memory Server.

## Advanced Factory Patterns

### Multi-Environment Factory

Create factories that adapt to different environments:

```python
# my_vectorstores.py
import os
from langchain_core.embeddings import Embeddings
from langchain_redis import Redis as LangchainRedis
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore

def create_adaptive_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Dynamically choose vectorstore based on environment."""

    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        # Use Pinecone for production
        return PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embeddings,
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
    elif environment == "staging":
        # Use Redis for staging
        return LangchainRedis(
            redis_url=os.getenv("REDIS_URL"),
            index_name="staging_memories",
            embeddings=embeddings
        )
    else:
        # Use Chroma for development
        return Chroma(
            persist_directory="./dev_chroma_data",
            collection_name="dev_memories",
            embedding_function=embeddings
        )
```

### High-Availability Factory

Create factories with resilience and failover capabilities:

```python
# resilient_factory.py
import os
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

def create_resilient_vectorstore(embeddings: Embeddings) -> VectorStore:
    """Create vectorstore with built-in resilience patterns."""

    # Try multiple backends in order of preference
    backend_preferences = [
        ("redis", _create_redis_backend),
        ("chroma", _create_chroma_backend),
        ("memory", _create_memory_backend)  # Fallback to in-memory
    ]

    last_error = None
    for backend_name, factory_func in backend_preferences:
        try:
            vectorstore = factory_func(embeddings)
            print(f"Successfully initialized {backend_name} vectorstore")
            return vectorstore
        except Exception as e:
            print(f"Failed to initialize {backend_name}: {e}")
            last_error = e
            continue

    raise Exception(f"All vectorstore backends failed. Last error: {last_error}")

def _create_redis_backend(embeddings: Embeddings) -> VectorStore:
    """Try Redis with connection validation."""
    from langchain_redis import Redis as LangchainRedis

    vectorstore = LangchainRedis(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        index_name="resilient_memories",
        embeddings=embeddings
    )

    # Validate connection
    vectorstore.client.ping()
    return vectorstore

def _create_chroma_backend(embeddings: Embeddings) -> VectorStore:
    """Fallback to Chroma."""
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=os.getenv("BACKUP_PERSIST_DIR", "./backup_chroma"),
        collection_name="backup_memories",
        embedding_function=embeddings
    )

def _create_memory_backend(embeddings: Embeddings) -> VectorStore:
    """Final fallback to in-memory store."""
    from langchain_core.vectorstores import InMemoryVectorStore

    return InMemoryVectorStore(embeddings)
```

### Multi-Backend Hybrid Factory

Combine multiple backends for different use cases:

```python
# hybrid_factory.py
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing import Dict, Any

class HybridVectorStore(VectorStore):
    """Hybrid vectorstore that routes operations based on content type."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.fast_store = self._create_fast_store(embeddings)  # Redis for recent data
        self.archive_store = self._create_archive_store(embeddings)  # S3/cheaper storage

    def _create_fast_store(self, embeddings: Embeddings) -> VectorStore:
        """Create fast vectorstore for recent/active memories."""
        from langchain_redis import Redis as LangchainRedis
        return LangchainRedis(
            redis_url=os.getenv("REDIS_URL"),
            index_name="fast_memories",
            embeddings=embeddings
        )

    def _create_archive_store(self, embeddings: Embeddings) -> VectorStore:
        """Create archive vectorstore for old/inactive memories."""
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=os.getenv("ARCHIVE_PERSIST_DIR", "./archive"),
            collection_name="archived_memories",
            embedding_function=embeddings
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] = None, **kwargs):
        """Add texts to appropriate store based on metadata."""
        if not metadatas:
            metadatas = [{}] * len(texts)

        # Route based on memory age or access patterns
        fast_texts, fast_meta = [], []
        archive_texts, archive_meta = [], []

        for text, meta in zip(texts, metadatas):
            # Route recent or high-access memories to fast store
            if self._should_use_fast_store(meta):
                fast_texts.append(text)
                fast_meta.append(meta)
            else:
                archive_texts.append(text)
                archive_meta.append(meta)

        # Add to appropriate stores
        results = []
        if fast_texts:
            results.extend(self.fast_store.add_texts(fast_texts, fast_meta, **kwargs))
        if archive_texts:
            results.extend(self.archive_store.add_texts(archive_texts, archive_meta, **kwargs))

        return results

    def similarity_search(self, query: str, k: int = 4, **kwargs):
        """Search both stores and combine results."""
        # Search fast store first (likely to have relevant recent data)
        fast_results = self.fast_store.similarity_search(query, k=k//2, **kwargs)
        archive_results = self.archive_store.similarity_search(query, k=k//2, **kwargs)

        # Combine and re-rank results
        all_results = fast_results + archive_results
        return all_results[:k]

    def _should_use_fast_store(self, metadata: dict) -> bool:
        """Determine if memory should go to fast store."""
        # Example routing logic
        access_count = metadata.get("access_count", 0)
        created_days_ago = self._days_since_created(metadata.get("created_at"))

        return access_count > 5 or created_days_ago < 30

    def _days_since_created(self, created_at: str) -> float:
        """Calculate days since creation."""
        if not created_at:
            return float('inf')
        # Implementation depends on your timestamp format
        return 0.0  # Placeholder

def create_hybrid_vectorstore(embeddings: Embeddings) -> HybridVectorStore:
    """Factory for hybrid vectorstore."""
    return HybridVectorStore(embeddings)
```



## Custom Adapter Implementation

### Advanced Custom Adapter

```python
# custom_advanced_adapter.py
from agent_memory_server.vectorstore_adapter import VectorStoreAdapter
from agent_memory_server.models import MemoryRecord, MemoryRecordResult
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import Optional, List
import logging

class AdvancedCustomAdapter(VectorStoreAdapter):
    """Advanced custom adapter with caching and batch operations."""

    def __init__(self, vectorstore, embeddings: Embeddings):
        super().__init__(vectorstore, embeddings)
        self.logger = logging.getLogger(__name__)
        self._embedding_cache = {}
        self._batch_size = 50

    async def add_memories(self, memories: List[MemoryRecord]) -> List[str]:
        """Add memories with optimized batching and caching."""
        if not memories:
            return []

        self.logger.info(f"Adding {len(memories)} memories in batches of {self._batch_size}")

        all_ids = []

        # Process in batches
        for i in range(0, len(memories), self._batch_size):
            batch = memories[i:i + self._batch_size]
            batch_ids = await self._add_memory_batch(batch)
            all_ids.extend(batch_ids)

        return all_ids

    async def _add_memory_batch(self, memories: List[MemoryRecord]) -> List[str]:
        """Add a batch of memories with optimizations."""

        # Prepare documents
        documents = []
        for memory in memories:
            # Generate embeddings with caching
            embedding = await self._get_cached_embedding(memory.text)

            doc = Document(
                id=memory.id,
                page_content=memory.text,
                metadata=self._prepare_metadata(memory)
            )
            documents.append(doc)

        # Add to vectorstore
        try:
            if hasattr(self.vectorstore, 'aadd_documents'):
                return await self.vectorstore.aadd_documents(documents)
            else:
                return self.vectorstore.add_documents(documents)
        except Exception as e:
            self.logger.error(f"Error adding batch: {e}")
            raise

    async def _get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding with caching for performance."""

        text_hash = hash(text)

        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        # Generate new embedding
        if hasattr(self.embeddings, 'aembed_query'):
            embedding = await self.embeddings.aembed_query(text)
        else:
            embedding = self.embeddings.embed_query(text)

        # Cache with size limit
        if len(self._embedding_cache) < 1000:  # Limit cache size
            self._embedding_cache[text_hash] = embedding

        return embedding

    def _prepare_metadata(self, memory: MemoryRecord) -> dict:
        """Prepare metadata optimized for the specific backend."""

        metadata = {
            "id_": memory.id,
            "user_id": memory.user_id,
            "namespace": memory.namespace or "default",
            "memory_type": memory.memory_type.value,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "access_count": memory.access_count or 0,
            "pinned": getattr(memory, "pinned", False)
        }

        # Add topics and entities if present
        if memory.topics:
            metadata["topics"] = memory.topics[:10]  # Limit array size

        if memory.entities:
            metadata["entities"] = memory.entities[:10]

        # Remove None values
        return {k: v for k, v in metadata.items() if v is not None}

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        namespace: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> MemoryRecordResult:
        """Search with advanced filtering and ranking."""

        # Build filter conditions
        filters = {}
        if namespace:
            filters["namespace"] = namespace
        if user_id:
            filters["user_id"] = user_id

        # Perform search with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Custom search implementation based on your vectorstore
                results = await self._perform_search(query, limit, filters, **kwargs)

                # Post-process results
                processed_results = self._post_process_results(results, query)

                return MemoryRecordResult(
                    memories=processed_results[:limit],
                    total_count=len(processed_results)
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Search attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"Search failed after {max_retries} attempts: {e}")
                    raise

    async def _perform_search(self, query: str, limit: int, filters: dict, **kwargs):
        """Perform the actual search operation."""
        # Implementation depends on your specific vectorstore
        # This is a template - implement based on your backend

        if hasattr(self.vectorstore, 'asimilarity_search'):
            return await self.vectorstore.asimilarity_search(
                query=query,
                k=limit,
                filter=filters,
                **kwargs
            )
        else:
            return self.vectorstore.similarity_search(
                query=query,
                k=limit,
                filter=filters,
                **kwargs
            )

    def _post_process_results(self, results: List[Document], query: str) -> List[MemoryRecord]:
        """Post-process search results for optimization."""

        processed = []

        for doc in results:
            try:
                # Convert document back to MemoryRecord
                memory = self._document_to_memory(doc)

                # Add computed relevance score if not present
                if not hasattr(memory, 'relevance_score'):
                    memory.relevance_score = self._calculate_relevance(doc, query)

                processed.append(memory)

            except Exception as e:
                self.logger.warning(f"Error processing result: {e}")
                continue

        # Sort by relevance
        processed.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)

        return processed

    def _calculate_relevance(self, doc: Document, query: str) -> float:
        """Calculate custom relevance score."""
        # Simple text similarity as fallback
        # Replace with more sophisticated scoring if needed

        text_lower = doc.page_content.lower()
        query_lower = query.lower()

        # Basic keyword matching score
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)

def create_advanced_custom_adapter(embeddings: Embeddings) -> AdvancedCustomAdapter:
    """Factory for advanced custom adapter."""

    # Use any vectorstore backend
    from langchain_chroma import Chroma

    vectorstore = Chroma(
        persist_directory=os.getenv("CUSTOM_PERSIST_DIR", "./custom_data"),
        collection_name="advanced_memories",
        embedding_function=embeddings
    )

    return AdvancedCustomAdapter(vectorstore, embeddings)
```

## Migration Strategies

### Data Export and Import

```python
# migration_tools.py
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from agent_memory_client import MemoryAPIClient

class VectorStoreMigrator:
    """Tool for migrating data between vector stores."""

    def __init__(self, source_client: MemoryAPIClient, target_client: MemoryAPIClient):
        self.source = source_client
        self.target = target_client

    async def migrate_all_memories(self, batch_size: int = 100) -> Dict[str, int]:
        """Migrate all memories from source to target."""

        print("Starting migration...")

        # Export all memories
        memories = await self.export_memories()
        print(f"Exported {len(memories)} memories")

        # Import in batches
        imported_count = await self.import_memories(memories, batch_size)

        # Verification
        verification_results = await self.verify_migration()

        return {
            "exported": len(memories),
            "imported": imported_count,
            "verification_passed": verification_results["success"],
            "missing_memories": verification_results["missing_count"]
        }

    async def export_memories(self, user_id: str = None, namespace: str = None) -> List[Dict[str, Any]]:
        """Export memories from source system."""

        memories = []
        offset = 0
        batch_size = 1000

        while True:
            # Search with pagination
            results = await self.source.search_long_term_memory(
                text="",  # Empty query to get all
                user_id=user_id,
                namespace=namespace,
                limit=batch_size,
                offset=offset
            )

            if not results.memories:
                break

            # Convert to exportable format
            for memory in results.memories:
                memory_dict = {
                    "id": memory.id,
                    "text": memory.text,
                    "memory_type": memory.memory_type,
                    "user_id": memory.user_id,
                    "session_id": memory.session_id,
                    "namespace": memory.namespace,
                    "topics": memory.topics,
                    "entities": memory.entities,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                    "access_count": memory.access_count,
                    "pinned": getattr(memory, "pinned", False)
                }
                memories.append(memory_dict)

            offset += batch_size
            print(f"Exported {len(memories)} memories so far...")

        return memories

    async def import_memories(self, memories: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Import memories to target system."""

        imported_count = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]

            # Convert to MemoryRecord format
            memory_records = []
            for mem_dict in batch:
                # Remove None values and prepare for import
                clean_dict = {k: v for k, v in mem_dict.items() if v is not None}
                memory_records.append(clean_dict)

            try:
                # Import batch
                result = await self.target.create_long_term_memories(memory_records)
                imported_count += len(result.memories)

                print(f"Imported batch {i//batch_size + 1}: {len(result.memories)} memories")

                # Small delay to avoid overwhelming the target system
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error importing batch {i//batch_size + 1}: {e}")
                # Continue with next batch

        return imported_count

    async def verify_migration(self, sample_size: int = 100) -> Dict[str, Any]:
        """Verify migration by sampling memories."""

        # Get sample from source
        source_sample = await self.source.search_long_term_memory(
            text="",
            limit=sample_size
        )

        missing_count = 0
        verified_count = 0

        for memory in source_sample.memories:
            # Try to find in target
            target_results = await self.target.search_long_term_memory(
                text=memory.text[:100],  # Use first 100 chars for matching
                user_id=memory.user_id,
                limit=5
            )

            # Look for exact match
            found = any(
                result.id == memory.id or result.text == memory.text
                for result in target_results.memories
            )

            if found:
                verified_count += 1
            else:
                missing_count += 1

        success_rate = verified_count / len(source_sample.memories) if source_sample.memories else 0

        return {
            "success": success_rate > 0.95,  # 95% success threshold
            "success_rate": success_rate,
            "verified_count": verified_count,
            "missing_count": missing_count,
            "sample_size": len(source_sample.memories)
        }

    async def export_to_file(self, filename: str, user_id: str = None, namespace: str = None):
        """Export memories to JSON file."""

        memories = await self.export_memories(user_id, namespace)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_count": len(memories),
            "user_id": user_id,
            "namespace": namespace,
            "memories": memories
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(memories)} memories to {filename}")

    async def import_from_file(self, filename: str, batch_size: int = 100) -> int:
        """Import memories from JSON file."""

        with open(filename, 'r', encoding='utf-8') as f:
            export_data = json.load(f)

        memories = export_data.get("memories", [])
        print(f"Found {len(memories)} memories in {filename}")

        return await self.import_memories(memories, batch_size)

# Usage example
async def migrate_redis_to_pinecone():
    """Example: Migrate from Redis to Pinecone."""

    # Source (Redis)
    source_client = MemoryAPIClient(
        base_url="http://localhost:8000",  # Current Redis setup
    )

    # Target (Pinecone) - Temporarily switch backend
    target_client = MemoryAPIClient(
        base_url="http://localhost:8001",  # New Pinecone setup
    )

    migrator = VectorStoreMigrator(source_client, target_client)

    # Option 1: Direct migration
    results = await migrator.migrate_all_memories(batch_size=50)
    print(f"Migration results: {results}")

    # Option 2: File-based migration (safer for large datasets)
    await migrator.export_to_file("memory_export.json")
    # ... Stop old server, start new server with Pinecone backend ...
    imported = await migrator.import_from_file("memory_export.json")
    print(f"Imported {imported} memories from file")
```

### Zero-Downtime Migration

```python
# zero_downtime_migration.py
import asyncio
from datetime import datetime, timedelta
from typing import Set

class ZeroDowntimeMigrator:
    """Perform migration with zero downtime using dual-write strategy."""

    def __init__(self, primary_client: MemoryAPIClient, secondary_client: MemoryAPIClient):
        self.primary = primary_client
        self.secondary = secondary_client
        self.migration_start_time = None

    async def start_dual_write_migration(self):
        """Start dual-write phase of migration."""

        self.migration_start_time = datetime.now()
        print(f"Starting dual-write migration at {self.migration_start_time}")

        # Phase 1: Start writing to both systems
        print("Phase 1: Enabling dual writes...")
        await self._enable_dual_writes()

        # Phase 2: Backfill historical data
        print("Phase 2: Backfilling historical data...")
        await self._backfill_historical_data()

        # Phase 3: Verify consistency
        print("Phase 3: Verifying data consistency...")
        consistency_check = await self._verify_consistency()

        if consistency_check["success"]:
            print("✅ Migration ready for cutover")
            return True
        else:
            print("❌ Consistency check failed")
            return False

    async def _enable_dual_writes(self):
        """Configure system to write to both primary and secondary."""
        # This would require modification to the memory server
        # to support dual writes during migration
        pass

    async def _backfill_historical_data(self):
        """Copy all historical data to secondary system."""

        migrator = VectorStoreMigrator(self.primary, self.secondary)

        # Only migrate data created before migration start
        cutoff_time = self.migration_start_time

        print(f"Backfilling data created before {cutoff_time}")

        # Export historical memories
        memories = []
        offset = 0
        batch_size = 1000

        while True:
            results = await self.primary.search_long_term_memory(
                text="",
                limit=batch_size,
                offset=offset,
                created_before=cutoff_time  # Only historical data
            )

            if not results.memories:
                break

            memories.extend(results.memories)
            offset += batch_size

            print(f"Collected {len(memories)} historical memories...")

        # Import to secondary
        imported = await migrator.import_memories(
            [self._memory_to_dict(mem) for mem in memories],
            batch_size=100
        )

        print(f"Backfilled {imported} historical memories")

    async def _verify_consistency(self) -> dict:
        """Verify both systems have consistent data."""

        # Sample recent memories from both systems
        sample_size = 1000

        primary_memories = await self.primary.search_long_term_memory(
            text="",
            limit=sample_size,
            created_after=self.migration_start_time - timedelta(hours=1)
        )

        secondary_memories = await self.secondary.search_long_term_memory(
            text="",
            limit=sample_size,
            created_after=self.migration_start_time - timedelta(hours=1)
        )

        # Compare memory IDs
        primary_ids = {mem.id for mem in primary_memories.memories}
        secondary_ids = {mem.id for mem in secondary_memories.memories}

        missing_in_secondary = primary_ids - secondary_ids
        extra_in_secondary = secondary_ids - primary_ids

        consistency_rate = len(primary_ids.intersection(secondary_ids)) / len(primary_ids) if primary_ids else 1.0

        return {
            "success": consistency_rate > 0.98,  # 98% consistency threshold
            "consistency_rate": consistency_rate,
            "missing_in_secondary": len(missing_in_secondary),
            "extra_in_secondary": len(extra_in_secondary),
            "primary_count": len(primary_ids),
            "secondary_count": len(secondary_ids)
        }

    def _memory_to_dict(self, memory) -> dict:
        """Convert memory object to dictionary."""
        return {
            "id": memory.id,
            "text": memory.text,
            "memory_type": memory.memory_type,
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "namespace": memory.namespace,
            "topics": memory.topics,
            "entities": memory.entities,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            "access_count": memory.access_count,
            "pinned": getattr(memory, "pinned", False)
        }

    async def complete_cutover(self):
        """Complete migration by switching traffic to secondary."""

        print("Completing cutover to secondary system...")

        # Final consistency check
        final_check = await self._verify_consistency()

        if not final_check["success"]:
            raise Exception("Final consistency check failed - aborting cutover")

        # At this point, you would:
        # 1. Update configuration to use secondary as primary
        # 2. Stop dual writes
        # 3. Decommission old primary

        print("✅ Cutover completed successfully")
        return final_check
```

This documentation covers advanced architectural patterns for vector store configuration, focusing on flexible factory patterns, custom implementations, and data migration strategies that work across different backends.
