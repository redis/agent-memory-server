"""
Redis Memory Server API Client

This module provides a client for the REST API of the Redis Memory Server.
"""

from typing import Any, Literal

import httpx
from pydantic import BaseModel

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
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryRequest,
    HealthCheckResponse,
    LongTermMemory,
    LongTermMemoryResults,
    MemoryPromptRequest,
    MemoryPromptResponse,
    SearchRequest,
    SessionListResponse,
    SessionMemory,
    SessionMemoryRequest,
    SessionMemoryResponse,
)


# Model name literals for model-specific window sizes
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


class MemoryClientConfig(BaseModel):
    """Configuration for the Memory API Client"""

    base_url: str
    timeout: float = 30.0
    default_namespace: str | None = None


class MemoryAPIClient:
    """
    Client for the Redis Memory Server REST API.

    This client provides methods to interact with all server endpoints:
    - Health check
    - Session management (list, get, put, delete)
    - Long-term memory (create, search)
    """

    def __init__(self, config: MemoryClientConfig):
        """
        Initialize the Memory API Client.

        Args:
            config: MemoryClientConfig instance with server connection details
        """
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Support using the client as an async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the client when exiting the context manager."""
        await self.close()

    async def health_check(self) -> HealthCheckResponse:
        """
        Check the health of the memory server.

        Returns:
            HealthCheckResponse with current server timestamp
        """
        response = await self._client.get("/health")
        response.raise_for_status()
        return HealthCheckResponse(**response.json())

    async def list_sessions(
        self, limit: int = 20, offset: int = 0, namespace: str | None = None
    ) -> SessionListResponse:
        """
        List available sessions with optional pagination and namespace filtering.

        Args:
            limit: Maximum number of sessions to return (default: 20)
            offset: Offset for pagination (default: 0)
            namespace: Optional namespace filter

        Returns:
            SessionListResponse containing session IDs and total count
        """
        params = {
            "limit": str(limit),
            "offset": str(offset),
        }
        if namespace is not None:
            params["namespace"] = namespace
        elif self.config.default_namespace is not None:
            params["namespace"] = self.config.default_namespace

        response = await self._client.get("/sessions/", params=params)
        response.raise_for_status()
        return SessionListResponse(**response.json())

    async def get_session_memory(
        self,
        session_id: str,
        namespace: str | None = None,
        window_size: int | None = None,
        model_name: ModelNameLiteral | None = None,
        context_window_max: int | None = None,
    ) -> SessionMemoryResponse:
        """
        Get memory for a session, including messages and context.

        Args:
            session_id: The session ID to retrieve memory for
            namespace: Optional namespace for the session
            window_size: Optional number of messages to include
            model_name: Optional model name to determine context window size
            context_window_max: Optional direct specification of context window tokens

        Returns:
            SessionMemoryResponse containing messages, context and metadata

        Raises:
            httpx.HTTPStatusError: If the session is not found (404) or other errors
        """
        params = {}

        if namespace is not None:
            params["namespace"] = namespace
        elif self.config.default_namespace is not None:
            params["namespace"] = self.config.default_namespace

        if window_size is not None:
            params["window_size"] = window_size

        if model_name is not None:
            params["model_name"] = model_name

        if context_window_max is not None:
            params["context_window_max"] = context_window_max

        response = await self._client.get(
            f"/sessions/{session_id}/memory", params=params
        )
        response.raise_for_status()
        return SessionMemoryResponse(**response.json())

    async def put_session_memory(
        self, session_id: str, memory: SessionMemory
    ) -> AckResponse:
        """
        Store session memory. Replaces existing session memory if it exists.

        Args:
            session_id: The session ID to store memory for
            memory: SessionMemory object with messages and optional context

        Returns:
            AckResponse indicating success
        """
        # If namespace not specified in memory but set in config, use config's namespace
        if memory.namespace is None and self.config.default_namespace is not None:
            memory.namespace = self.config.default_namespace

        response = await self._client.put(
            f"/sessions/{session_id}/memory", json=memory.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return AckResponse(**response.json())

    async def delete_session_memory(
        self, session_id: str, namespace: str | None = None
    ) -> AckResponse:
        """
        Delete memory for a session.

        Args:
            session_id: The session ID to delete memory for
            namespace: Optional namespace for the session

        Returns:
            AckResponse indicating success
        """
        params = {}
        if namespace is not None:
            params["namespace"] = namespace
        elif self.config.default_namespace is not None:
            params["namespace"] = self.config.default_namespace

        response = await self._client.delete(
            f"/sessions/{session_id}/memory", params=params
        )
        response.raise_for_status()
        return AckResponse(**response.json())

    async def create_long_term_memory(
        self, memories: list[LongTermMemory]
    ) -> AckResponse:
        """
        Create long-term memories for later retrieval.

        Args:
            memories: List of LongTermMemory objects to store

        Returns:
            AckResponse indicating success

        Raises:
            httpx.HTTPStatusError: If long-term memory is disabled (400) or other errors
        """
        # Apply default namespace if needed
        if self.config.default_namespace is not None:
            for memory in memories:
                if memory.namespace is None:
                    memory.namespace = self.config.default_namespace

        payload = CreateLongTermMemoryRequest(memories=memories)
        response = await self._client.post(
            "/long-term-memory", json=payload.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return AckResponse(**response.json())

    async def search_long_term_memory(
        self,
        text: str,
        session_id: SessionId | dict[str, Any] | None = None,
        namespace: Namespace | dict[str, Any] | None = None,
        topics: Topics | dict[str, Any] | None = None,
        entities: Entities | dict[str, Any] | None = None,
        created_at: CreatedAt | dict[str, Any] | None = None,
        last_accessed: LastAccessed | dict[str, Any] | None = None,
        user_id: UserId | dict[str, Any] | None = None,
        distance_threshold: float | None = None,
        memory_type: MemoryType | dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> LongTermMemoryResults:
        """
        Search long-term memories using semantic search and filters.

        Args:
            text: Search query text for semantic similarity
            session_id: Optional session ID filter
            namespace: Optional namespace filter
            topics: Optional topics filter
            entities: Optional entities filter
            created_at: Optional creation date filter
            last_accessed: Optional last accessed date filter
            user_id: Optional user ID filter
            distance_threshold: Optional distance threshold for search results
            limit: Maximum number of results to return (default: 10)
            offset: Offset for pagination (default: 0)

        Returns:
            LongTermMemoryResults with matching memories and metadata

        Raises:
            httpx.HTTPStatusError: If long-term memory is disabled (400) or other errors
        """
        # Convert dictionary filters to their proper filter objects if needed
        if isinstance(session_id, dict):
            session_id = SessionId(**session_id)
        if isinstance(namespace, dict):
            namespace = Namespace(**namespace)
        if isinstance(topics, dict):
            topics = Topics(**topics)
        if isinstance(entities, dict):
            entities = Entities(**entities)
        if isinstance(created_at, dict):
            created_at = CreatedAt(**created_at)
        if isinstance(last_accessed, dict):
            last_accessed = LastAccessed(**last_accessed)
        if isinstance(user_id, dict):
            user_id = UserId(**user_id)
        if isinstance(memory_type, dict):
            memory_type = MemoryType(**memory_type)

        # Apply default namespace if needed and no namespace filter specified
        if namespace is None and self.config.default_namespace is not None:
            namespace = Namespace(eq=self.config.default_namespace)

        payload = SearchRequest(
            text=text,
            session_id=session_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            user_id=user_id,
            distance_threshold=distance_threshold,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        )

        response = await self._client.post(
            "/long-term-memory/search", json=payload.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return LongTermMemoryResults(**response.json())

    async def memory_prompt(
        self,
        query: str,
        session_id: str | None = None,
        namespace: str | None = None,
        window_size: int | None = None,
        model_name: ModelNameLiteral | None = None,
        context_window_max: int | None = None,
        long_term_search: SearchRequest | None = None,
    ) -> MemoryPromptResponse:
        """
        Hydrate a user query with memory context and return a prompt
        ready to send to an LLM.

        This method can retrieve relevant session history and long-term memories
        to provide context for the query.

        Args:
            query: The user's query text
            session_id: Optional session ID to retrieve history from
            namespace: Optional namespace for session and long-term memories
            window_size: Optional number of messages to include from session history
            model_name: Optional model name to determine context window size
            context_window_max: Optional direct specification of context window max tokens
            long_term_search: Optional SearchRequest for specific long-term memory filtering

        Returns:
            MemoryPromptResponse containing a list of messages with context

        Raises:
            httpx.HTTPStatusError: If the request fails or if neither session_id nor long_term_search is provided
        """
        # Prepare the request payload
        session_params = None
        if session_id is not None:
            session_params = SessionMemoryRequest(
                session_id=session_id,
                namespace=namespace or self.config.default_namespace,
                window_size=window_size or 12,  # Default from settings
                model_name=model_name,
                context_window_max=context_window_max,
            )

        # If no explicit long_term_search is provided but we have a query, create a basic one
        if long_term_search is None and query:
            # Use default namespace from config if none provided
            _namespace = None
            if namespace is not None:
                _namespace = Namespace(eq=namespace)
            elif self.config.default_namespace is not None:
                _namespace = Namespace(eq=self.config.default_namespace)

            long_term_search = SearchRequest(
                text=query,
                namespace=_namespace,
            )

        # Create the request payload
        payload = MemoryPromptRequest(
            query=query,
            session=session_params,
            long_term_search=long_term_search,
        )

        # Make the API call
        response = await self._client.post(
            "/memory-prompt", json=payload.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        data = response.json()
        return MemoryPromptResponse(**data)

    async def hydrate_memory_prompt(
        self,
        query: str,
        session_id: SessionId | dict[str, Any] | None = None,
        namespace: Namespace | dict[str, Any] | None = None,
        topics: Topics | dict[str, Any] | None = None,
        entities: Entities | dict[str, Any] | None = None,
        created_at: CreatedAt | dict[str, Any] | None = None,
        last_accessed: LastAccessed | dict[str, Any] | None = None,
        user_id: UserId | dict[str, Any] | None = None,
        distance_threshold: float | None = None,
        memory_type: MemoryType | dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        window_size: int = 12,
        model_name: ModelNameLiteral | None = None,
        context_window_max: int | None = None,
    ) -> MemoryPromptResponse:
        """
        Hydrate a user query with relevant session history and long-term memories.

        This method enriches the user's query by retrieving:
        1. Context from the conversation session (if session_id is provided)
        2. Relevant long-term memories related to the query

        Args:
            query: The user's query text
            session_id: Optional filter for session ID
            namespace: Optional filter for namespace
            topics: Optional filter for topics in long-term memories
            entities: Optional filter for entities in long-term memories
            created_at: Optional filter for creation date
            last_accessed: Optional filter for last access date
            user_id: Optional filter for user ID
            distance_threshold: Optional distance threshold for semantic search
            memory_type: Optional filter for memory type
            limit: Maximum number of long-term memory results (default: 10)
            offset: Offset for pagination (default: 0)
            window_size: Number of messages to include from session history (default: 12)
            model_name: Optional model name to determine context window size
            context_window_max: Optional direct specification of context window max tokens

        Returns:
            MemoryPromptResponse containing a list of messages with context

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        # Convert dictionary filters to their proper filter objects if needed
        if isinstance(session_id, dict):
            session_id = SessionId(**session_id)
        if isinstance(namespace, dict):
            namespace = Namespace(**namespace)
        if isinstance(topics, dict):
            topics = Topics(**topics)
        if isinstance(entities, dict):
            entities = Entities(**entities)
        if isinstance(created_at, dict):
            created_at = CreatedAt(**created_at)
        if isinstance(last_accessed, dict):
            last_accessed = LastAccessed(**last_accessed)
        if isinstance(user_id, dict):
            user_id = UserId(**user_id)
        if isinstance(memory_type, dict):
            memory_type = MemoryType(**memory_type)

        # Apply default namespace if needed and no namespace filter specified
        if namespace is None and self.config.default_namespace is not None:
            namespace = Namespace(eq=self.config.default_namespace)

        # Extract session_id value if it exists
        session_params = None
        _session_id = None
        if session_id and hasattr(session_id, "eq") and session_id.eq:
            _session_id = session_id.eq

        if _session_id:
            # Get namespace value if it exists
            _namespace = None
            if namespace and hasattr(namespace, "eq"):
                _namespace = namespace.eq
            elif self.config.default_namespace:
                _namespace = self.config.default_namespace

            session_params = SessionMemoryRequest(
                session_id=_session_id,
                namespace=_namespace,
                window_size=window_size,
                model_name=model_name,
                context_window_max=context_window_max,
            )

        # Create search request for long-term memory
        search_payload = SearchRequest(
            text=query,
            session_id=session_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            user_id=user_id,
            distance_threshold=distance_threshold,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        )

        # Create the request payload
        payload = MemoryPromptRequest(
            query=query,
            session=session_params,
            long_term_search=search_payload,
        )

        # Make the API call
        response = await self._client.post(
            "/memory-prompt", json=payload.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        data = response.json()
        return MemoryPromptResponse(**data)


# Helper function to create a memory client
async def create_memory_client(
    base_url: str, timeout: float = 30.0, default_namespace: str | None = None
) -> MemoryAPIClient:
    """
    Create and initialize a Memory API Client.

    Args:
        base_url: Base URL of the memory server (e.g., 'http://localhost:8000')
        timeout: Request timeout in seconds (default: 30.0)
        default_namespace: Optional default namespace to use for operations

    Returns:
        Initialized MemoryAPIClient instance
    """
    config = MemoryClientConfig(
        base_url=base_url,
        timeout=timeout,
        default_namespace=default_namespace,
    )
    client = MemoryAPIClient(config)

    # Test connection with a health check
    try:
        await client.health_check()
    except Exception as e:
        await client.close()
        raise ConnectionError(
            f"Failed to connect to memory server at {base_url}: {e}"
        ) from e

    return client
