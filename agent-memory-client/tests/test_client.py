"""
Test file for the enhanced Memory API Client functionality.

Tests for new features like lifecycle management, batch operations,
pagination utilities, validation, and enhanced convenience methods.
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import (
    AckResponse,
    ClientMemoryRecord,
    MemoryMessage,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
    RecencyConfig,
    WorkingMemoryResponse,
)


@pytest.fixture
async def enhanced_test_client() -> AsyncGenerator[MemoryAPIClient, None]:
    """Create a memory client for testing with mocked HTTP client."""
    config = MemoryClientConfig(
        base_url="http://test", default_namespace="test-namespace"
    )
    client = MemoryAPIClient(config)

    # Mock the HTTP client to avoid actual network calls
    client._client = AsyncMock(spec=httpx.AsyncClient)

    yield client

    await client.close()


class TestMemoryLifecycleManagement:
    """Tests for memory lifecycle management methods."""

    @pytest.mark.asyncio
    async def test_promote_working_memories_to_long_term(self, enhanced_test_client):
        """Test promoting specific working memories to long-term storage."""
        session_id = "test-session"

        # Create test memories
        memories = [
            ClientMemoryRecord(
                id="memory-1",
                text="User prefers dark mode",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            ClientMemoryRecord(
                id="memory-2",
                text="User completed project setup",
                memory_type=MemoryTypeEnum.EPISODIC,
            ),
        ]

        # Mock working memory response
        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=memories,
            data={},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_working_memory") as mock_get,
            patch.object(
                enhanced_test_client, "create_long_term_memory"
            ) as mock_create,
        ):
            mock_get.return_value = working_memory_response
            mock_create.return_value = AckResponse(status="ok")

            # Test promoting all memories
            result = await enhanced_test_client.promote_working_memories_to_long_term(
                session_id=session_id
            )

            assert result.status == "ok"
            mock_get.assert_called_once_with(
                session_id=session_id,
                user_id=None,
                namespace=None,
                model_name=None,
                context_window_max=None,
            )
            mock_create.assert_called_once_with(memories)

    @pytest.mark.asyncio
    async def test_promote_specific_memory_ids(self, enhanced_test_client):
        """Test promoting only specific memory IDs."""
        session_id = "test-session"

        memories = [
            ClientMemoryRecord(
                id="memory-1",
                text="User prefers dark mode",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            ClientMemoryRecord(
                id="memory-2",
                text="User completed project setup",
                memory_type=MemoryTypeEnum.EPISODIC,
            ),
        ]

        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=memories,
            data={},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_working_memory") as mock_get,
            patch.object(
                enhanced_test_client, "create_long_term_memory"
            ) as mock_create,
        ):
            mock_get.return_value = working_memory_response
            mock_create.return_value = AckResponse(status="ok")

            # Test promoting only specific memory
            result = await enhanced_test_client.promote_working_memories_to_long_term(
                session_id=session_id, memory_ids=["memory-1"]
            )

            assert result.status == "ok"
            # Should only promote memory-1
            mock_create.assert_called_once()
            promoted_memories = mock_create.call_args[0][0]
            assert len(promoted_memories) == 1
            assert promoted_memories[0].id == "memory-1"

    @pytest.mark.asyncio
    async def test_promote_no_memories(self, enhanced_test_client):
        """Test promoting when no memories exist."""
        session_id = "test-session"

        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={},
            context=None,
            user_id=None,
        )

        with patch.object(enhanced_test_client, "get_working_memory") as mock_get:
            mock_get.return_value = working_memory_response

            result = await enhanced_test_client.promote_working_memories_to_long_term(
                session_id=session_id
            )

            assert result.status == "ok"


class TestBatchOperations:
    """Tests for batch operations."""

    @pytest.mark.asyncio
    async def test_bulk_create_long_term_memories(self, enhanced_test_client):
        """Test bulk creation of long-term memories with batching."""
        # Create test memory batches
        batch1 = [
            ClientMemoryRecord(
                text=f"Memory {i}",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
            for i in range(50)
        ]
        batch2 = [
            ClientMemoryRecord(
                text=f"Memory {i}",
                memory_type=MemoryTypeEnum.EPISODIC,
            )
            for i in range(30)
        ]

        memory_batches = [batch1, batch2]

        with patch.object(
            enhanced_test_client, "create_long_term_memory"
        ) as mock_create:
            mock_create.return_value = AckResponse(status="ok")

            # Test with default batch size
            results = await enhanced_test_client.bulk_create_long_term_memories(
                memory_batches=memory_batches,
                batch_size=25,
                delay_between_batches=0,  # No delay for test speed
            )

            # Should have created 4 batches: 25+25 for batch1, 25+5 for batch2
            assert len(results) == 4
            assert all(result.status == "ok" for result in results)
            assert mock_create.call_count == 4

    @pytest.mark.asyncio
    async def test_bulk_create_with_delay(self, enhanced_test_client):
        """Test bulk creation with rate limiting delay."""
        batch = [
            ClientMemoryRecord(
                text="Test memory",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
        ]

        with (
            patch.object(
                enhanced_test_client, "create_long_term_memory"
            ) as mock_create,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_create.return_value = AckResponse(status="ok")

            asyncio.get_event_loop().time()
            await enhanced_test_client.bulk_create_long_term_memories(
                memory_batches=[batch],
                delay_between_batches=0.1,
            )

            # Should have called sleep (though mocked)
            mock_sleep.assert_called_with(0.1)


class TestPaginationUtilities:
    """Tests for pagination utilities."""

    @pytest.mark.asyncio
    async def test_search_all_long_term_memories(self, enhanced_test_client):
        """Test auto-paginating search for long-term memories."""
        # Mock responses for pagination
        first_response = MemoryRecordResults(
            total=150,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(50)
            ],
            next_offset=50,
        )

        second_response = MemoryRecordResults(
            total=150,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(50, 100)
            ],
            next_offset=100,
        )

        third_response = MemoryRecordResults(
            total=150,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(100, 130)  # Less than batch_size, indicating end
            ],
            next_offset=None,
        )

        with patch.object(
            enhanced_test_client, "search_long_term_memory"
        ) as mock_search:
            mock_search.side_effect = [first_response, second_response, third_response]

            # Collect all results
            all_memories = []
            async for memory in enhanced_test_client.search_all_long_term_memories(
                text="test query", batch_size=50
            ):
                all_memories.append(memory)

            # Should have retrieved all 130 memories
            assert len(all_memories) == 130
            assert all_memories[0].id == "memory-0"
            assert all_memories[-1].id == "memory-129"

            # Should have made 3 API calls
            assert mock_search.call_count == 3


class TestRecencyConfig:
    @pytest.mark.asyncio
    async def test_recency_config_descriptive_parameters(self, enhanced_test_client):
        """Test that RecencyConfig descriptive parameters are properly sent to API."""
        with patch.object(enhanced_test_client._client, "post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = MemoryRecordResults(
                total=0, memories=[], next_offset=None
            ).model_dump()
            mock_post.return_value = mock_response

            rc = RecencyConfig(
                recency_boost=True,
                semantic_weight=0.8,
                recency_weight=0.2,
                freshness_weight=0.6,
                novelty_weight=0.4,
                half_life_last_access_days=7,
                half_life_created_days=30,
                server_side_recency=True,
            )

            await enhanced_test_client.search_long_term_memory(
                text="search query", recency=rc, limit=5
            )

            # Verify payload contains descriptive parameter names
            args, kwargs = mock_post.call_args
            assert args[0] == "/v1/long-term-memory/search"
            body = kwargs["json"]
            assert body["recency_boost"] is True
            assert body["recency_semantic_weight"] == 0.8
            assert body["recency_recency_weight"] == 0.2
            assert body["recency_freshness_weight"] == 0.6
            assert body["recency_novelty_weight"] == 0.4
            assert body["recency_half_life_last_access_days"] == 7
            assert body["recency_half_life_created_days"] == 30
            assert body["server_side_recency"] is True


class TestClientSideValidation:
    """Tests for client-side validation methods."""

    def test_validate_memory_record_success(self, enhanced_test_client):
        """Test successful memory record validation."""
        memory = ClientMemoryRecord(
            text="Valid memory text",
            memory_type=MemoryTypeEnum.SEMANTIC,
            id="01HN0000000000000000000000",  # Valid ULID
        )

        # Should not raise
        enhanced_test_client.validate_memory_record(memory)

    def test_validate_memory_record_empty_text(self, enhanced_test_client):
        """Test validation failure for empty text."""
        memory = ClientMemoryRecord(
            text="",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        with pytest.raises(ValueError, match="Memory text cannot be empty"):
            enhanced_test_client.validate_memory_record(memory)

    def test_validate_memory_record_invalid_type(self, enhanced_test_client):
        """Test validation failure for invalid memory type."""
        # Test with a valid memory but manually set invalid type
        memory = ClientMemoryRecord(
            text="Valid text",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        # Manually override the memory type to test validation
        memory.memory_type = "invalid_type"  # type: ignore

        with pytest.raises(ValueError, match="Invalid memory type"):
            enhanced_test_client.validate_memory_record(memory)

    def test_validate_memory_record_invalid_id(self, enhanced_test_client):
        """Test validation failure for invalid ID format."""
        memory = ClientMemoryRecord(
            text="Valid text",
            memory_type=MemoryTypeEnum.SEMANTIC,
            id="invalid-id-format",
        )

        with pytest.raises(ValueError, match="Invalid ID format"):
            enhanced_test_client.validate_memory_record(memory)

    def test_validate_search_filters_success(self, enhanced_test_client):
        """Test successful search filter validation."""
        filters = {
            "limit": 10,
            "offset": 0,
            "distance_threshold": 0.5,
            "session_id": "test-session",
        }

        # Should not raise
        enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_key(self, enhanced_test_client):
        """Test validation failure for invalid filter key."""
        filters = {"invalid_key": "value"}

        with pytest.raises(ValueError, match="Invalid filter key"):
            enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_limit(self, enhanced_test_client):
        """Test validation failure for invalid limit."""
        filters = {"limit": -1}

        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_offset(self, enhanced_test_client):
        """Test validation failure for invalid offset."""
        filters = {"offset": -1}

        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_distance(self, enhanced_test_client):
        """Test validation failure for invalid distance threshold."""
        filters = {"distance_threshold": -0.5}

        with pytest.raises(
            ValueError, match="Distance threshold must be a non-negative number"
        ):
            enhanced_test_client.validate_search_filters(**filters)


class TestEnhancedConvenienceMethods:
    """Tests for enhanced convenience methods."""

    @pytest.mark.asyncio
    async def test_update_working_memory_data_merge(self, enhanced_test_client):
        """Test updating working memory data with merge strategy."""
        session_id = "test-session"

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={"existing_key": "existing_value", "shared_key": "old_value"},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_working_memory") as mock_get,
            patch.object(enhanced_test_client, "put_working_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            updates = {"new_key": "new_value", "shared_key": "new_value"}

            await enhanced_test_client.update_working_memory_data(
                session_id=session_id,
                data_updates=updates,
                merge_strategy="merge",
            )

            # Check that put was called with merged data
            mock_put.assert_called_once()
            working_memory_arg = mock_put.call_args[0][1]
            expected_data = {
                "existing_key": "existing_value",
                "shared_key": "new_value",
                "new_key": "new_value",
            }
            assert working_memory_arg.data == expected_data

    @pytest.mark.asyncio
    async def test_update_working_memory_data_replace(self, enhanced_test_client):
        """Test updating working memory data with replace strategy."""
        session_id = "test-session"

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={"existing_key": "existing_value"},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_working_memory") as mock_get,
            patch.object(enhanced_test_client, "put_working_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            updates = {"new_key": "new_value"}

            await enhanced_test_client.update_working_memory_data(
                session_id=session_id,
                data_updates=updates,
                merge_strategy="replace",
            )

            # Check that put was called with replaced data
            working_memory_arg = mock_put.call_args[0][1]
            assert working_memory_arg.data == updates

    @pytest.mark.asyncio
    async def test_update_working_memory_data_deep_merge(self, enhanced_test_client):
        """Test updating working memory data with deep merge strategy."""
        session_id = "test-session"

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={
                "nested": {"existing": "value", "shared": "old"},
                "top_level": "existing",
            },
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_working_memory") as mock_get,
            patch.object(enhanced_test_client, "put_working_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            updates = {
                "nested": {"new": "value", "shared": "new"},
                "new_top": "new",
            }

            await enhanced_test_client.update_working_memory_data(
                session_id=session_id,
                data_updates=updates,
                merge_strategy="deep_merge",
            )

            # Check deep merge result
            working_memory_arg = mock_put.call_args[0][1]
            expected_data = {
                "nested": {"existing": "value", "shared": "new", "new": "value"},
                "top_level": "existing",
                "new_top": "new",
            }
            assert working_memory_arg.data == expected_data

    @pytest.mark.asyncio
    async def test_append_messages_to_working_memory(self, enhanced_test_client):
        """Test appending messages to existing working memory."""
        session_id = "test-session"

        existing_messages = [
            MemoryMessage(role="user", content="First message"),
        ]

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=existing_messages,
            memories=[],
            data={},
            context=None,
            user_id=None,
        )

        new_messages = [
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]

        with (
            patch.object(enhanced_test_client, "get_working_memory") as mock_get,
            patch.object(enhanced_test_client, "put_working_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            await enhanced_test_client.append_messages_to_working_memory(
                session_id=session_id,
                messages=new_messages,
            )

            # Check that messages were appended
            working_memory_arg = mock_put.call_args[0][1]
            assert len(working_memory_arg.messages) == 3
            assert working_memory_arg.messages[0].content == "First message"
            assert working_memory_arg.messages[1].content == "Second message"
            assert working_memory_arg.messages[2].content == "Third message"

    def test_deep_merge_dicts(self, enhanced_test_client):
        """Test the deep merge dictionary utility method."""
        base = {
            "a": {"nested": {"deep": "value1", "shared": "old"}},
            "b": "simple",
        }

        updates = {
            "a": {"nested": {"shared": "new", "additional": "value2"}},
            "c": "new_simple",
        }

        result = enhanced_test_client._deep_merge_dicts(base, updates)

        expected = {
            "a": {
                "nested": {
                    "deep": "value1",
                    "shared": "new",
                    "additional": "value2",
                }
            },
            "b": "simple",
            "c": "new_simple",
        }

        assert result == expected

    def test_is_valid_ulid(self, enhanced_test_client):
        """Test ULID validation utility method."""
        # Valid ULID
        assert enhanced_test_client._is_valid_ulid("01HN0000000000000000000000")

        # Invalid ULID
        assert not enhanced_test_client._is_valid_ulid("invalid-id")
        assert not enhanced_test_client._is_valid_ulid("")
        assert not enhanced_test_client._is_valid_ulid("too-short")


class TestErrorHandling:
    """Tests for error handling in new methods."""

    @pytest.mark.asyncio
    async def test_bulk_create_handles_failures(self, enhanced_test_client):
        """Test that bulk create handles individual batch failures."""
        batch = [
            ClientMemoryRecord(
                text="Test memory",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
        ]

        with patch.object(
            enhanced_test_client, "create_long_term_memory"
        ) as mock_create:
            # First call succeeds, second fails, third succeeds
            mock_create.side_effect = [
                AckResponse(status="ok"),
                Exception("API Error"),
                AckResponse(status="ok"),
            ]

            # Should raise the exception from the second batch
            with pytest.raises(Exception, match="API Error"):
                await enhanced_test_client.bulk_create_long_term_memories(
                    memory_batches=[batch, batch, batch],
                    delay_between_batches=0,
                )

    @pytest.mark.asyncio
    async def test_pagination_handles_empty_results(self, enhanced_test_client):
        """Test pagination utilities handle empty result sets."""
        empty_response = MemoryRecordResults(
            total=0,
            memories=[],
            next_offset=None,
        )

        with patch.object(
            enhanced_test_client, "search_long_term_memory"
        ) as mock_search:
            mock_search.return_value = empty_response

            # Should handle empty results gracefully
            all_memories = []
            async for memory in enhanced_test_client.search_all_long_term_memories(
                text="test query"
            ):
                all_memories.append(memory)

            assert len(all_memories) == 0
            assert mock_search.call_count == 1

    def test_validation_with_none_values(self, enhanced_test_client):
        """Test validation handles None values appropriately."""
        memory = ClientMemoryRecord(
            text="Valid text",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        # ClientMemoryRecord generates a ULID ID by default, so this should pass

        # Should not raise
        enhanced_test_client.validate_memory_record(memory)

    @pytest.mark.asyncio
    async def test_get_or_create_handles_404_correctly(self, enhanced_test_client):
        """Test that get_or_create_working_memory properly handles 404 errors.

        This test verifies the fix for a bug where _handle_http_error would raise
        MemoryNotFoundError, but then the code would re-raise the original
        HTTPStatusError, preventing get_or_create_working_memory from catching
        the MemoryNotFoundError and creating a new session.
        """

        session_id = "nonexistent-session"

        # Mock get_working_memory to raise MemoryNotFoundError (simulating 404)
        async def mock_get_working_memory(*args, **kwargs):
            # Simulate what happens when the server returns 404
            response = Mock()
            response.status_code = 404
            response.url = f"http://test/v1/working-memory/{session_id}"
            raise httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=response
            )

        # Mock put_working_memory to return a created session
        async def mock_put_working_memory(*args, **kwargs):
            return WorkingMemoryResponse(
                session_id=session_id,
                messages=[],
                memories=[],
                data={},
                context=None,
                user_id=None,
            )

        with (
            patch.object(
                enhanced_test_client,
                "get_working_memory",
                side_effect=mock_get_working_memory,
            ),
            patch.object(
                enhanced_test_client,
                "put_working_memory",
                side_effect=mock_put_working_memory,
            ),
        ):
            # This should NOT raise an exception - it should create a new session
            created, memory = await enhanced_test_client.get_or_create_working_memory(
                session_id=session_id
            )

            # Verify that a new session was created
            assert created is True
            assert memory.session_id == session_id


class TestContextUsagePercentage:
    """Tests for context usage percentage functionality."""

    @pytest.mark.asyncio
    async def test_working_memory_response_with_context_percentages(
        self, enhanced_test_client
    ):
        """Test that WorkingMemoryResponse properly handles both context percentage fields."""
        session_id = "test-session"

        # Test with both context percentages set
        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={},
            context=None,
            user_id=None,
            context_percentage_total_used=45.5,
            context_percentage_until_summarization=65.0,
        )

        assert working_memory_response.context_percentage_total_used == 45.5
        assert working_memory_response.context_percentage_until_summarization == 65.0
        assert working_memory_response.session_id == session_id

        # Test with None context percentages (default)
        working_memory_response_none = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={},
            context=None,
            user_id=None,
        )

        assert working_memory_response_none.context_percentage_total_used is None
        assert (
            working_memory_response_none.context_percentage_until_summarization is None
        )

    @pytest.mark.asyncio
    async def test_context_percentages_serialization(self, enhanced_test_client):
        """Test that both context percentage fields are properly serialized."""
        session_id = "test-session"

        # Create response with both context percentages
        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={},
            context=None,
            user_id=None,
            context_percentage_total_used=75.0,
            context_percentage_until_summarization=85.5,
        )

        # Test model_dump includes both fields
        dumped = working_memory_response.model_dump()
        assert "context_percentage_total_used" in dumped
        assert "context_percentage_until_summarization" in dumped
        assert dumped["context_percentage_total_used"] == 75.0
        assert dumped["context_percentage_until_summarization"] == 85.5

        # Test JSON serialization
        json_data = working_memory_response.model_dump_json()
        assert "context_percentage_total_used" in json_data
        assert "context_percentage_until_summarization" in json_data
        assert "75.0" in json_data
        assert "85.5" in json_data

    @pytest.mark.asyncio
    async def test_context_percentages_validation(self, enhanced_test_client):
        """Test that both context percentage fields accept valid values."""
        session_id = "test-session"

        # Test valid percentages
        valid_percentages = [0.0, 25.5, 50.0, 99.9, 100.0, None]

        for percentage in valid_percentages:
            working_memory_response = WorkingMemoryResponse(
                session_id=session_id,
                messages=[],
                memories=[],
                data={},
                context=None,
                user_id=None,
                context_percentage_total_used=percentage,
                context_percentage_until_summarization=percentage,
            )
            assert working_memory_response.context_percentage_total_used == percentage
            assert (
                working_memory_response.context_percentage_until_summarization
                == percentage
            )

    def test_working_memory_response_from_dict_with_context_percentages(self):
        """Test that WorkingMemoryResponse can be created from dict with both context percentage fields."""
        session_id = "test-session"

        # Test creating WorkingMemoryResponse from dict (simulating API response parsing)
        response_dict = {
            "session_id": session_id,
            "messages": [],
            "memories": [],
            "data": {},
            "context": None,
            "user_id": None,
            "context_percentage_total_used": 33.3,
            "context_percentage_until_summarization": 47.5,
            "tokens": 0,
            "namespace": None,
            "ttl_seconds": None,
            "last_accessed": "2024-01-01T00:00:00Z",
        }

        # This simulates what happens when the API client parses the JSON response
        result = WorkingMemoryResponse(**response_dict)

        # Verify both context percentage fields are included
        assert isinstance(result, WorkingMemoryResponse)
        assert result.context_percentage_total_used == 33.3
        assert result.context_percentage_until_summarization == 47.5
        assert result.session_id == session_id
