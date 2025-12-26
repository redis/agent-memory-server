import contextlib
import os
import time
from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import docket
import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.embeddings import Embeddings
from redis.asyncio import Redis as AsyncRedis
from testcontainers.compose import DockerCompose

from agent_memory_server.api import router as memory_router
from agent_memory_server.config import settings
from agent_memory_server.dependencies import HybridBackgroundTasks
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.llm import LLMClient
from agent_memory_server.models import (
    MemoryMessage,
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
)

# Import the module to access its global for resetting
from agent_memory_server.utils import redis as redis_utils_module
from agent_memory_server.utils.keys import Keys
from agent_memory_server.vectorstore_adapter import VectorStoreAdapter


# from agent_memory_server.utils.redis import ensure_search_index_exists  # Not used currently


load_dotenv()


@pytest.fixture()
def memory_message():
    """Create a sample memory message"""
    return MemoryMessage(role="user", content="Hello, world!")


@pytest.fixture()
def memory_messages():
    """Create a list of sample memory messages"""
    return [
        MemoryMessage(role="user", content="What is the capital of France?"),
        MemoryMessage(role="assistant", content="The capital of France is Paris."),
        MemoryMessage(role="user", content="And what about Germany?"),
        MemoryMessage(role="assistant", content="The capital of Germany is Berlin."),
    ]


@pytest.fixture()
def mock_llm_client():
    """Create a mock LLM client"""
    return AsyncMock(spec=LLMClient)

    # We won't set default side effects here, allowing tests to set their own mocks
    # This prevents conflicts with tests that need specific return values


@pytest.fixture(autouse=True)
async def search_index(async_redis_client):
    """Create a Redis connection pool for testing"""
    # Reset the cached index in redis_utils_module
    redis_utils_module._index = None

    yield
    return

    await async_redis_client.flushdb()

    try:
        try:
            await async_redis_client.execute_command(
                "FT.INFO", settings.redisvl_index_name
            )
            await async_redis_client.execute_command(
                "FT.DROPINDEX", settings.redisvl_index_name
            )
        except Exception as e:
            if "unknown index name".lower() not in str(e).lower():
                pass

        # Skip ensure_search_index_exists for now - let LangChain handle it
        # await ensure_search_index_exists(async_redis_client)

    except Exception:
        raise

    yield

    # Clean up after tests
    await async_redis_client.flushdb()
    with contextlib.suppress(Exception):
        await async_redis_client.execute_command(
            "FT.DROPINDEX", settings.redisvl_index_name
        )


@pytest.fixture()
async def session(use_test_redis_connection, async_redis_client, request):
    """Set up a test session with Redis data for testing"""
    import logging

    logging.getLogger(__name__)

    try:
        session_id = "test-session"
        namespace = "test-namespace"

        # Create working memory data
        from agent_memory_server.models import MemoryMessage, WorkingMemory

        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        working_memory = WorkingMemory(
            messages=messages,
            memories=[],  # No structured memories for this test
            context="Sample context",
            user_id="test-user",
            tokens=150,
            session_id=session_id,
            namespace=namespace,
        )

        # Store in unified working memory format
        from agent_memory_server.working_memory import set_working_memory

        await set_working_memory(
            working_memory=working_memory,
            redis_client=use_test_redis_connection,
        )

        # Also add session to sessions list for compatibility
        sessions_key = Keys.sessions_key(namespace=namespace)
        current_time = int(time.time())
        await use_test_redis_connection.zadd(sessions_key, {session_id: current_time})

        # Index the messages as long-term memories directly without background tasks
        from redisvl.utils.vectorize import OpenAITextVectorizer
        from ulid import ULID

        from agent_memory_server.models import MemoryRecord

        # Create MemoryRecord objects for each message
        long_term_memories = []
        for msg in messages:
            memory = MemoryRecord(
                id=str(ULID()),
                text=f"{msg.role}: {msg.content}",
                session_id=session_id,
                namespace=namespace,
                user_id="test-user",
            )
            long_term_memories.append(memory)

        # Index the memories directly (only if tests explicitly opt-in to using
        # real API keys). This prevents tests that don't require external
        # services from accidentally making network calls when OPENAI_API_KEY
        # is set in the environment.
        import os

        requires_api_keys_marker = request.node.get_closest_marker("requires_api_keys")
        has_api_key = bool(os.getenv("OPENAI_API_KEY"))

        if not (has_api_key and requires_api_keys_marker):
            # Skip embedding creation if no API key is configured or the test
            # has not been marked as requiring API keys. Tests can still run
            # with an empty semantic index.
            embeddings = []
        else:
            vectorizer = OpenAITextVectorizer()
            embeddings = await vectorizer.aembed_many(
                [memory.text for memory in long_term_memories],
                batch_size=20,
                as_buffer=True,
            )

        # Only index if we have embeddings
        if embeddings:
            async with use_test_redis_connection.pipeline(transaction=False) as pipe:
                for idx, vector in enumerate(embeddings):
                    memory = long_term_memories[idx]
                    id_ = memory.id if memory.id else str(ULID())
                    key = Keys.memory_key(id_)

                    # Generate memory hash for the memory
                    from agent_memory_server.long_term_memory import (
                        generate_memory_hash,
                    )

                    memory_hash = generate_memory_hash(memory)

                    await pipe.hset(  # type: ignore
                        key,
                        mapping={
                            "text": memory.text,
                            "id_": id_,
                            "session_id": memory.session_id or "",
                            "user_id": memory.user_id or "",
                            "last_accessed": int(memory.last_accessed.timestamp())
                            if memory.last_accessed
                            else int(time.time()),
                            "created_at": int(memory.created_at.timestamp())
                            if memory.created_at
                            else int(time.time()),
                            "namespace": memory.namespace or "",
                            "memory_hash": memory_hash,
                            "vector": vector,
                            "topics": "",
                            "entities": "",
                        },
                    )

                await pipe.execute()

        return session_id
    except Exception:
        raise


@pytest.fixture(scope="session", autouse=True)
def redis_container(request):
    """
    If using xdist, create a unique Compose project for each xdist worker by
    setting COMPOSE_PROJECT_NAME. That prevents collisions on container/volume
    names.
    """
    # In xdist, the config has "workerid" in workerinput
    workerinput = getattr(request.config, "workerinput", {})
    worker_id = workerinput.get("workerid", "master")

    # Set the Compose project name so containers do not clash across workers
    os.environ["COMPOSE_PROJECT_NAME"] = f"redis_test_{worker_id}"
    os.environ.setdefault("REDIS_IMAGE", "redis:8.0.3")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    compose = DockerCompose(
        context=current_dir,
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    compose.start()

    yield compose

    compose.stop()


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    host, port = redis_container.get_service_host_and_port("redis", 6379)

    # On macOS, testcontainers sometimes returns 0.0.0.0 which doesn't work
    # Replace with localhost if we get 0.0.0.0
    if host == "0.0.0.0":
        host = "localhost"

    redis_url = f"redis://{host}:{port}"

    # Verify the connection works before returning with retries
    import time

    import redis

    max_retries = 10
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            client = redis.Redis.from_url(redis_url)
            client.ping()
            break  # Connection successful
        except Exception as e:
            if attempt == max_retries - 1:
                raise ConnectionError(
                    f"Failed to connect to Redis at {redis_url} after {max_retries} attempts: {e}"
                ) from e
            time.sleep(retry_delay)

    return redis_url


@pytest.fixture()
def async_redis_client(use_test_redis_connection):
    """
    An async Redis client that uses the same connection as other test fixtures.
    """
    return use_test_redis_connection


@pytest.fixture()
def mock_async_redis_client():
    """Create a mock async Redis client"""
    return AsyncMock(spec=AsyncRedis)


@pytest.fixture(autouse=True)
def use_test_redis_connection(redis_url: str):
    """Replace the Redis connection with a test one"""
    replacement_redis = AsyncRedis.from_url(redis_url)

    # Create a mock get_redis_conn function that always returns the replacement_redis
    async def mock_get_redis_conn(*args, **kwargs):
        # Ignore any URL parameter and always return the replacement_redis
        return replacement_redis

    # Create a patched Docket class that uses the test Redis URL
    original_docket_init = docket.Docket.__init__

    def patched_docket_init(self, name, url=None, *args, **kwargs):
        # Use the test Redis URL instead of the default one
        return original_docket_init(self, name, *args, url=redis_url, **kwargs)

    # Reset all global state and patch get_redis_conn
    import agent_memory_server.utils.redis
    import agent_memory_server.vectorstore_factory

    with (
        patch("agent_memory_server.utils.redis.get_redis_conn", mock_get_redis_conn),
        patch("docket.docket.Docket.__init__", patched_docket_init),
        patch("agent_memory_server.working_memory.get_redis_conn", mock_get_redis_conn),
        patch("agent_memory_server.api.get_redis_conn", mock_get_redis_conn),
        patch(
            "agent_memory_server.long_term_memory.get_redis_conn", mock_get_redis_conn
        ),
        patch.object(settings, "redis_url", redis_url),
    ):
        # Reset global state to force recreation with test Redis
        agent_memory_server.utils.redis._redis_pool = None
        agent_memory_server.utils.redis._index = None
        agent_memory_server.vectorstore_factory._adapter = None

        yield replacement_redis

        # Clean up global state after test
        agent_memory_server.utils.redis._redis_pool = None
        agent_memory_server.utils.redis._index = None
        agent_memory_server.vectorstore_factory._adapter = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys",
    )
    parser.addoption(
        "--run-integration-tests",
        action="store_true",
        default=False,
        help="Run integration tests (requires running memory server)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_api_keys: mark test as requiring API keys",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a benchmark test",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test (requires running memory server)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if item.get_closest_marker("integration") and not config.getoption(
            "--run-integration-tests"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Not running integration tests. Use --run-integration-tests to run these tests."
                )
            )

        if item.get_closest_marker("requires_api_keys") and not config.getoption(
            "--run-api-tests"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Not running tests that require API keys. Use --run-api-tests to run these tests."
                )
            )


@pytest.fixture()
def mock_background_tasks():
    """Create a mock HybridBackgroundTasks instance"""
    return mock.Mock(name="HybridBackgroundTasks", spec=HybridBackgroundTasks)


@pytest.fixture()
def app(use_test_redis_connection):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)

    return app


@pytest.fixture()
def app_with_mock_background_tasks(use_test_redis_connection):
    """Create a test FastAPI app with routers and mocked background tasks"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)

    # Override the get_redis_conn function to return the test Redis connection
    async def mock_get_redis_conn(*args, **kwargs):
        return use_test_redis_connection

    # Override the dependencies
    from agent_memory_server.utils.redis import get_redis_conn

    app.dependency_overrides[get_redis_conn] = mock_get_redis_conn

    return app


@pytest.fixture(autouse=True)
def disable_auth_for_tests():
    """Disable authentication for all tests"""
    original_value = settings.disable_auth
    settings.disable_auth = True
    yield
    settings.disable_auth = original_value


@pytest.fixture()
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture()
async def client_with_mock_background_tasks(
    app_with_mock_background_tasks, mock_background_tasks
):
    """Client with mocked background tasks - patches the HybridBackgroundTasks class"""
    # Patch the HybridBackgroundTasks class to return our mock
    # We need to patch it in multiple places since FastAPI creates instances directly
    patches = [
        mock.patch(
            "agent_memory_server.api.HybridBackgroundTasks",
            return_value=mock_background_tasks,
        ),
        mock.patch(
            "agent_memory_server.dependencies.HybridBackgroundTasks",
            return_value=mock_background_tasks,
        ),
        mock.patch("fastapi.BackgroundTasks", return_value=mock_background_tasks),
    ]

    with patches[0], patches[1], patches[2]:
        async with AsyncClient(
            transport=ASGITransport(app=app_with_mock_background_tasks),
            base_url="http://test",
        ) as client:
            yield client


@pytest.fixture()
def mock_vectorstore_adapter():
    """Create a mock vectorstore adapter and patch get_vectorstore_adapter.

    This fixture provides a MockVectorStoreAdapter that doesn't require real
    embeddings or API keys, suitable for unit tests that don't need actual
    vector search functionality.

    Usage:
        def test_something(mock_vectorstore_adapter):
            # mock_vectorstore_adapter is already patched as the global adapter
            # You can also access the adapter instance directly:
            mock_vectorstore_adapter.memories["id"] = some_memory
    """
    adapter = MockVectorStoreAdapter()

    async def mock_get_vectorstore_adapter():
        return adapter

    with (
        patch(
            "agent_memory_server.vectorstore_factory.get_vectorstore_adapter",
            mock_get_vectorstore_adapter,
        ),
        patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            mock_get_vectorstore_adapter,
        ),
    ):
        # Also reset the global adapter to None to force re-creation
        import agent_memory_server.vectorstore_factory

        original_adapter = agent_memory_server.vectorstore_factory._adapter
        agent_memory_server.vectorstore_factory._adapter = None

        yield adapter

        # Restore original adapter
        agent_memory_server.vectorstore_factory._adapter = original_adapter


class MockEmbeddings(Embeddings):
    """Mock embeddings that return fixed-dimension vectors without API calls."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings for documents."""
        return [[0.1] * self.dimensions for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return mock embedding for a query."""
        return [0.1] * self.dimensions


class MockVectorStoreAdapter(VectorStoreAdapter):
    """Mock VectorStoreAdapter for testing without real embeddings."""

    def __init__(self):
        self.memories: dict[str, MemoryRecord] = {}
        self.embeddings = MockEmbeddings()
        self.vectorstore = MagicMock()

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memories to the mock store."""
        ids = []
        for memory in memories:
            self.memories[memory.id] = memory
            ids.append(memory.id)
        return ids

    async def search_memories(
        self,
        query: str,
        session_id: Any = None,
        user_id: Any = None,
        namespace: Any = None,
        created_at: Any = None,
        last_accessed: Any = None,
        topics: Any = None,
        entities: Any = None,
        memory_type: Any = None,
        event_date: Any = None,
        memory_hash: Any = None,
        id: Any = None,
        discrete_memory_extracted: Any = None,
        distance_threshold: float | None = None,
        server_side_recency: bool | None = None,
        recency_params: dict | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories in the mock store."""
        results = []
        for memory in list(self.memories.values()):
            # Apply basic filters
            if (
                namespace
                and hasattr(namespace, "eq")
                and namespace.eq
                and memory.namespace != namespace.eq
            ):
                continue
            if (
                user_id
                and hasattr(user_id, "eq")
                and user_id.eq
                and memory.user_id != user_id.eq
            ):
                continue
            if (
                session_id
                and hasattr(session_id, "eq")
                and session_id.eq
                and memory.session_id != session_id.eq
            ):
                continue
            if (
                memory_hash
                and hasattr(memory_hash, "eq")
                and memory_hash.eq
                and memory.memory_hash != memory_hash.eq
            ):
                continue
            if memory_type and hasattr(memory_type, "eq") and memory_type.eq:
                mem_type_val = (
                    memory.memory_type.value
                    if hasattr(memory.memory_type, "value")
                    else str(memory.memory_type)
                )
                if mem_type_val != memory_type.eq:
                    continue

            result = MemoryRecordResult(
                id=memory.id,
                text=memory.text,
                dist=0.1,
                created_at=memory.created_at or datetime.now(UTC),
                updated_at=memory.updated_at or datetime.now(UTC),
                last_accessed=memory.last_accessed or datetime.now(UTC),
                user_id=memory.user_id,
                session_id=memory.session_id,
                namespace=memory.namespace,
                topics=memory.topics or [],
                entities=memory.entities or [],
                memory_hash=memory.memory_hash or "",
                memory_type=memory.memory_type.value
                if hasattr(memory.memory_type, "value")
                else str(memory.memory_type),
                persisted_at=memory.persisted_at,
            )
            results.append(result)

        # Apply pagination
        paginated = results[offset : offset + limit]
        next_offset = offset + limit if len(results) > offset + limit else None

        return MemoryRecordResults(
            memories=paginated,
            total=len(results),
            next_offset=next_offset,
        )

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories from the mock store."""
        deleted = 0
        for memory_id in memory_ids:
            if memory_id in self.memories:
                del self.memories[memory_id]
                deleted += 1
        return deleted

    async def update_memories(self, memories: list[MemoryRecord]) -> int:
        """Update memories in the mock store."""
        updated = 0
        for memory in memories:
            self.memories[memory.id] = memory
            updated += 1
        return updated

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories in the mock store."""
        count = 0
        for memory in self.memories.values():
            if namespace and memory.namespace != namespace:
                continue
            if user_id and memory.user_id != user_id:
                continue
            if session_id and memory.session_id != session_id:
                continue
            count += 1
        return count

    async def list_memories(
        self,
        session_id: Any = None,
        user_id: Any = None,
        namespace: Any = None,
        created_at: Any = None,
        last_accessed: Any = None,
        topics: Any = None,
        entities: Any = None,
        memory_type: Any = None,
        event_date: Any = None,
        memory_hash: Any = None,
        id: Any = None,
        discrete_memory_extracted: Any = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """List memories in the mock store using filters without semantic search."""
        results = []
        for memory in list(self.memories.values()):
            # Apply basic filters
            if (
                namespace
                and hasattr(namespace, "eq")
                and namespace.eq
                and memory.namespace != namespace.eq
            ):
                continue
            if (
                user_id
                and hasattr(user_id, "eq")
                and user_id.eq
                and memory.user_id != user_id.eq
            ):
                continue
            if (
                session_id
                and hasattr(session_id, "eq")
                and session_id.eq
                and memory.session_id != session_id.eq
            ):
                continue
            if (
                memory_hash
                and hasattr(memory_hash, "eq")
                and memory_hash.eq
                and memory.memory_hash != memory_hash.eq
            ):
                continue
            if id and hasattr(id, "eq") and id.eq and memory.id != id.eq:
                continue
            if memory_type and hasattr(memory_type, "eq") and memory_type.eq:
                mem_type_val = (
                    memory.memory_type.value
                    if hasattr(memory.memory_type, "value")
                    else str(memory.memory_type)
                )
                if mem_type_val != memory_type.eq:
                    continue

            result = MemoryRecordResult(
                id=memory.id,
                text=memory.text,
                dist=0.0,  # No distance for filter-only queries
                created_at=memory.created_at or datetime.now(UTC),
                updated_at=memory.updated_at or datetime.now(UTC),
                last_accessed=memory.last_accessed or datetime.now(UTC),
                user_id=memory.user_id,
                session_id=memory.session_id,
                namespace=memory.namespace,
                topics=memory.topics or [],
                entities=memory.entities or [],
                memory_hash=memory.memory_hash or "",
                memory_type=memory.memory_type.value
                if hasattr(memory.memory_type, "value")
                else str(memory.memory_type),
                persisted_at=memory.persisted_at,
            )
            results.append(result)

        # Apply pagination
        paginated = results[offset : offset + limit]
        next_offset = offset + limit if len(results) > offset + limit else None

        return MemoryRecordResults(
            memories=paginated,
            total=len(results),
            next_offset=next_offset,
        )
