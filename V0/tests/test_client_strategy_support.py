"""Tests for client long_term_memory_strategy parameter support."""

import pytest
from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import MemoryStrategyConfig


@pytest.fixture
async def memory_client(use_test_redis_connection):
    """Create a memory client for testing."""
    from agent_memory_client import __version__

    config = MemoryClientConfig(
        base_url="http://test",
        disable_auth=True,
    )

    # Import here to avoid circular imports
    from httpx import ASGITransport, AsyncClient

    from agent_memory_server.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={
            "User-Agent": f"agent-memory-client/{__version__}",
            "X-Client-Version": __version__,
        },
    ) as http_client:
        client = MemoryAPIClient(config=config)
        client._client = http_client
        yield client


@pytest.mark.asyncio
async def test_get_or_create_working_memory_with_strategy(
    memory_client: MemoryAPIClient,
):
    """Test get_or_create_working_memory with long_term_memory_strategy parameter."""
    session_id = "test-strategy-session-1"

    # Create with custom strategy
    strategy = MemoryStrategyConfig(
        strategy="summary", config={"max_summary_length": 500}
    )

    created, memory = await memory_client.get_or_create_working_memory(
        session_id=session_id,
        long_term_memory_strategy=strategy,
    )

    assert created is True
    assert memory.session_id == session_id
    assert memory.long_term_memory_strategy.strategy == "summary"
    assert memory.long_term_memory_strategy.config == {"max_summary_length": 500}

    # Get existing session - strategy should be preserved
    created2, memory2 = await memory_client.get_or_create_working_memory(
        session_id=session_id,
    )

    assert created2 is False
    assert memory2.session_id == session_id
    assert memory2.long_term_memory_strategy.strategy == "summary"
    assert memory2.long_term_memory_strategy.config == {"max_summary_length": 500}


@pytest.mark.asyncio
async def test_get_or_create_working_memory_tool_with_strategy(
    memory_client: MemoryAPIClient,
):
    """Test get_or_create_working_memory_tool with long_term_memory_strategy parameter."""
    session_id = "test-strategy-session-2"

    # Create with preferences strategy
    strategy = MemoryStrategyConfig(strategy="preferences", config={})

    result = await memory_client.get_or_create_working_memory_tool(
        session_id=session_id,
        long_term_memory_strategy=strategy,
    )

    assert result["created"] is True
    assert result["session_id"] == session_id

    # Verify the strategy was applied by getting the session
    created, memory = await memory_client.get_or_create_working_memory(
        session_id=session_id,
    )

    assert created is False
    assert memory.long_term_memory_strategy.strategy == "preferences"


@pytest.mark.asyncio
async def test_get_or_create_working_memory_with_custom_strategy(
    memory_client: MemoryAPIClient,
):
    """Test get_or_create_working_memory with custom strategy."""
    session_id = "test-strategy-session-3"

    # Create with custom strategy
    strategy = MemoryStrategyConfig(
        strategy="custom",
        config={
            "custom_prompt": "Extract technical decisions from: {message}\nReturn JSON."
        },
    )

    created, memory = await memory_client.get_or_create_working_memory(
        session_id=session_id,
        long_term_memory_strategy=strategy,
    )

    assert created is True
    assert memory.session_id == session_id
    assert memory.long_term_memory_strategy.strategy == "custom"
    assert "custom_prompt" in memory.long_term_memory_strategy.config


@pytest.mark.asyncio
async def test_get_or_create_working_memory_default_strategy(
    memory_client: MemoryAPIClient,
):
    """Test get_or_create_working_memory uses default strategy when not specified."""
    session_id = "test-strategy-session-4"

    # Create without specifying strategy
    created, memory = await memory_client.get_or_create_working_memory(
        session_id=session_id,
    )

    assert created is True
    assert memory.session_id == session_id
    # Should default to discrete strategy
    assert memory.long_term_memory_strategy.strategy == "discrete"
    assert memory.long_term_memory_strategy.config == {}
