import json
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import tenacity
import ulid

from agent_memory_server.config import settings
from agent_memory_server.extraction import (
    extract_discrete_memories,
    extract_entities,
    extract_topics_bertopic,
    extract_topics_llm,
    handle_extraction,
)
from agent_memory_server.filters import DiscreteMemoryExtracted, MemoryType
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum


@pytest.fixture
def mock_bertopic():
    """Mock BERTopic model"""
    mock = Mock()
    # Mock transform to return topic indices and probabilities
    mock.transform.return_value = (np.array([1]), np.array([0.8]))
    # Mock get_topic to return topic terms
    mock.get_topic.side_effect = lambda x: [("technology", 0.8), ("business", 0.7)]
    return mock


@pytest.fixture
def mock_ner():
    """Mock NER pipeline"""

    def mock_ner_fn(text):
        return [
            {"word": "John", "entity": "PER", "score": 0.99},
            {"word": "Google", "entity": "ORG", "score": 0.98},
            {"word": "Mountain", "entity": "LOC", "score": 0.97},
            {"word": "##View", "entity": "LOC", "score": 0.97},
        ]

    return Mock(side_effect=mock_ner_fn)


@pytest.fixture
def sample_message_memories():
    """Sample message memories for testing discrete extraction"""
    return [
        MemoryRecord(
            id=str(ulid.ULID()),
            text="User mentioned they prefer window seats when flying",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="session-123",
            user_id="user-456",
        ),
        MemoryRecord(
            id=str(ulid.ULID()),
            text="User works as a data scientist at Google",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="session-123",
            user_id="user-456",
        ),
        MemoryRecord(
            id=str(ulid.ULID()),
            text="Already processed message",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="t",
            session_id="session-123",
            user_id="user-456",
        ),
    ]


@pytest.mark.asyncio
class TestTopicExtraction:
    @patch("agent_memory_server.extraction.get_topic_model")
    async def test_extract_topics_success(self, mock_get_topic_model, mock_bertopic):
        """Test successful topic extraction"""
        mock_get_topic_model.return_value = mock_bertopic
        text = "Discussion about AI technology and business"

        topics = extract_topics_bertopic(text)

        assert set(topics) == {"technology", "business"}
        mock_bertopic.transform.assert_called_once_with([text])

    @patch("agent_memory_server.extraction.get_topic_model")
    async def test_extract_topics_no_valid_topics(
        self, mock_get_topic_model, mock_bertopic
    ):
        """Test when no valid topics are found"""
        mock_bertopic.transform.return_value = (np.array([-1]), np.array([0.0]))
        mock_get_topic_model.return_value = mock_bertopic

        topics = extract_topics_bertopic("Test message")

        assert topics == []
        mock_bertopic.transform.assert_called_once()


@pytest.mark.asyncio
class TestEntityExtraction:
    @patch("agent_memory_server.extraction.get_ner_model")
    async def test_extract_entities_success(self, mock_get_ner_model, mock_ner):
        """Test successful entity extraction"""
        mock_get_ner_model.return_value = mock_ner
        text = "John works at Google in Mountain View"

        entities = extract_entities(text)

        assert set(entities) == {"John", "Google", "MountainView"}
        mock_ner.assert_called_once_with(text)

    @patch("agent_memory_server.extraction.get_ner_model")
    async def test_extract_entities_error(self, mock_get_ner_model):
        """Test handling of NER model error"""
        mock_get_ner_model.side_effect = Exception("Model error")

        entities = extract_entities("Test message")

        assert entities == []


@pytest.mark.asyncio
class TestHandleExtraction:
    @patch("agent_memory_server.extraction.extract_topics_llm")
    @patch("agent_memory_server.extraction.extract_entities")
    async def test_handle_extraction(
        self, mock_extract_entities, mock_extract_topics_llm
    ):
        """Test extraction with topics/entities"""
        mock_extract_topics_llm.return_value = ["AI", "business"]
        mock_extract_entities.return_value = ["John", "Sarah", "Google"]

        topics, entities = await handle_extraction(
            "John and Sarah discussed AI at Google."
        )

        # Check that topics are as expected
        assert mock_extract_topics_llm.called
        assert set(topics) == {"AI", "business"}
        assert len(topics) == 2

        # Check that entities are as expected
        assert mock_extract_entities.called
        assert set(entities) == {"John", "Sarah", "Google"}
        assert len(entities) == 3

    @patch("agent_memory_server.extraction.extract_topics_llm")
    @patch("agent_memory_server.extraction.extract_entities")
    async def test_handle_extraction_disabled_features(
        self, mock_extract_entities, mock_extract_topics_llm
    ):
        """Test when features are disabled"""
        # Temporarily disable features
        original_topic_setting = settings.enable_topic_extraction
        original_ner_setting = settings.enable_ner
        settings.enable_topic_extraction = False
        settings.enable_ner = False

        try:
            topics, entities = await handle_extraction("Test message")

            assert topics == []
            assert entities == []
            mock_extract_topics_llm.assert_not_called()
            mock_extract_entities.assert_not_called()
        finally:
            # Restore settings
            settings.enable_topic_extraction = original_topic_setting
            settings.enable_ner = original_ner_setting


@pytest.mark.asyncio
class TestDiscreteMemoryExtraction:
    """Test the extract_discrete_memories function"""

    @patch("agent_memory_server.long_term_memory.index_long_term_memories")
    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_extract_discrete_memories_basic_flow(
        self,
        mock_get_client,
        mock_get_adapter,
        mock_index_memories,
        sample_message_memories,
    ):
        """Test basic flow of discrete memory extraction"""
        # Mock the LLM client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"memories": [{"type": "semantic", "text": "User prefers window seats", "topics": ["travel"], "entities": ["User"]}]}'
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()

        # Only return unprocessed memories (discrete_memory_extracted='f')
        unprocessed_memories = [
            mem
            for mem in sample_message_memories
            if mem.discrete_memory_extracted == "f"
        ]

        # Mock search results - first call returns unprocessed memories (< 25, so loop will exit)
        mock_search_result_1 = Mock()
        mock_search_result_1.memories = (
            unprocessed_memories  # Only 2 memories, so loop exits after first call
        )

        mock_adapter.search_memories.return_value = mock_search_result_1
        mock_adapter.update_memories = AsyncMock(return_value=len(unprocessed_memories))
        mock_get_adapter.return_value = mock_adapter

        # Mock index_long_term_memories
        mock_index_memories.return_value = None

        # Run the extraction
        await extract_discrete_memories(deduplicate=True)

        # Verify that search was called only once (since < 25 memories returned)
        assert mock_adapter.search_memories.call_count == 1

        # Check first search call
        first_call = mock_adapter.search_memories.call_args_list[0]
        assert first_call[1]["query"] == ""
        assert isinstance(first_call[1]["memory_type"], MemoryType)
        assert first_call[1]["memory_type"].eq == "message"
        assert isinstance(
            first_call[1]["discrete_memory_extracted"], DiscreteMemoryExtracted
        )
        assert first_call[1]["discrete_memory_extracted"].eq == "f"
        assert first_call[1]["limit"] == 25
        assert first_call[1]["offset"] == 0

        # Verify that update_memories was called once with batch of memories
        assert mock_adapter.update_memories.call_count == 1

        # Check that all memories were updated with discrete_memory_extracted='t'
        call_args = mock_adapter.update_memories.call_args_list[0]
        updated_memories = call_args[0][0]  # First positional argument
        assert len(updated_memories) == len(unprocessed_memories)
        for updated_memory in updated_memories:
            assert updated_memory.discrete_memory_extracted == "t"

        # Verify that LLM was called for each unprocessed memory
        assert mock_client.create_chat_completion.call_count == len(
            unprocessed_memories
        )

        # Verify that extracted memories were indexed
        mock_index_memories.assert_called_once()
        indexed_memories = mock_index_memories.call_args[0][0]
        assert len(indexed_memories) == len(
            unprocessed_memories
        )  # One extracted memory per message

        # Check that extracted memories have correct properties
        for memory in indexed_memories:
            assert memory.discrete_memory_extracted == "t"
            assert memory.memory_type in ["semantic", "episodic"]

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_extract_discrete_memories_no_unprocessed_memories(
        self,
        mock_get_client,
        mock_get_adapter,
    ):
        """Test when there are no unprocessed memories"""
        # Mock the vectorstore adapter to return no memories
        mock_adapter = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.memories = []
        mock_adapter.search_memories.return_value = mock_search_result
        mock_get_adapter.return_value = mock_adapter

        # Mock the LLM client (should not be called)
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Run the extraction
        await extract_discrete_memories(deduplicate=True)

        # Verify that search was called once
        mock_adapter.search_memories.assert_called_once()

        # Verify that LLM was not called since no memories to process
        mock_client.create_chat_completion.assert_not_called()

        # Verify that update was not called
        mock_adapter.update_memories.assert_not_called()

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_extract_discrete_memories_handles_empty_text(
        self,
        mock_get_client,
        mock_get_adapter,
    ):
        """Test handling of memories with empty text"""
        # Create a memory with empty text
        empty_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
        )

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_search_result_1 = Mock()
        mock_search_result_1.memories = [empty_memory]
        mock_search_result_2 = Mock()
        mock_search_result_2.memories = []

        mock_adapter.search_memories.side_effect = [
            mock_search_result_1,
            mock_search_result_2,
        ]
        mock_adapter.delete_memories = AsyncMock(return_value=1)
        mock_get_adapter.return_value = mock_adapter

        # Mock the LLM client (should not be called)
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Run the extraction
        await extract_discrete_memories(deduplicate=True)

        # Verify that delete was called for the empty memory
        mock_adapter.delete_memories.assert_called_once_with([empty_memory.id])

        # Verify that LLM was not called
        mock_client.create_chat_completion.assert_not_called()

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_extract_discrete_memories_handles_missing_id(
        self,
        mock_get_client,
        mock_get_adapter,
    ):
        """Test handling of memories with missing ID"""
        # Create a memory with no ID - simulate this by creating a mock that has id=None
        no_id_memory = Mock()
        no_id_memory.id = None
        no_id_memory.text = "Some text"
        no_id_memory.memory_type = MemoryTypeEnum.MESSAGE
        no_id_memory.discrete_memory_extracted = "f"

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_search_result_1 = Mock()
        mock_search_result_1.memories = [no_id_memory]
        mock_search_result_2 = Mock()
        mock_search_result_2.memories = []

        mock_adapter.search_memories.side_effect = [
            mock_search_result_1,
            mock_search_result_2,
        ]
        mock_get_adapter.return_value = mock_adapter

        # Mock the LLM client - need to set it up properly in case it gets called
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"memories": [{"type": "semantic", "text": "Extracted memory", "topics": [], "entities": []}]}'
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Run the extraction
        await extract_discrete_memories(deduplicate=True)

        # The current implementation processes memories with missing IDs
        # The LLM will be called since the memory has text
        mock_client.create_chat_completion.assert_called_once()

        # Verify that update was called with the processed memory
        mock_adapter.update_memories.assert_called_once()

    @patch("agent_memory_server.long_term_memory.index_long_term_memories")
    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_extract_discrete_memories_pagination(
        self,
        mock_get_client,
        mock_get_adapter,
        mock_index_memories,
    ):
        """Test that pagination works correctly"""
        # Create more than 25 memories to test pagination
        many_memories = []
        for i in range(30):
            memory = MemoryRecord(
                id=str(ulid.ULID()),
                text=f"Message {i}",
                memory_type=MemoryTypeEnum.MESSAGE,
                discrete_memory_extracted="f",
            )
            many_memories.append(memory)

        # Mock the LLM client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"memories": [{"type": "semantic", "text": "Extracted memory", "topics": [], "entities": []}]}'
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()

        # First call returns exactly 25 memories (triggers next page), second call returns remaining 5 (< 25, so loop exits)
        mock_search_result_1 = Mock()
        mock_search_result_1.memories = many_memories[:25]  # Exactly 25, so continues
        mock_search_result_2 = Mock()
        mock_search_result_2.memories = many_memories[25:]  # Only 5, so stops

        mock_adapter.search_memories.side_effect = [
            mock_search_result_1,
            mock_search_result_2,
        ]
        mock_adapter.update_memories = AsyncMock(return_value=1)
        mock_get_adapter.return_value = mock_adapter

        # Mock index_long_term_memories
        mock_index_memories.return_value = None

        # Run the extraction
        await extract_discrete_memories(deduplicate=True)

        # Verify that search was called 2 times (first returns 25, second returns 5, loop exits)
        assert mock_adapter.search_memories.call_count == 2

        # Check pagination offsets
        calls = mock_adapter.search_memories.call_args_list
        assert calls[0][1]["offset"] == 0
        assert calls[1][1]["offset"] == 25

        # Verify that all memories were processed in batch
        assert mock_adapter.update_memories.call_count == 1
        assert mock_client.create_chat_completion.call_count == 30

        # Verify that the batch update contains all 30 memories
        call_args = mock_adapter.update_memories.call_args_list[0]
        updated_memories = call_args[0][0]  # First positional argument
        assert len(updated_memories) == 30

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_discrete_memory_extracted_filter_integration(
        self,
        mock_get_client,
        mock_get_adapter,
    ):
        """Test that the DiscreteMemoryExtracted filter works correctly"""
        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.memories = []
        mock_adapter.search_memories.return_value = mock_search_result
        mock_get_adapter.return_value = mock_adapter

        # Mock the LLM client
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Run the extraction
        await extract_discrete_memories(deduplicate=True)

        # Verify that search was called with the correct filter
        mock_adapter.search_memories.assert_called_once()
        call_args = mock_adapter.search_memories.call_args

        # Check that DiscreteMemoryExtracted filter was used correctly
        discrete_filter = call_args[1]["discrete_memory_extracted"]
        assert isinstance(discrete_filter, DiscreteMemoryExtracted)
        assert discrete_filter.eq == "f"
        assert discrete_filter.field == "discrete_memory_extracted"

    @patch("agent_memory_server.long_term_memory.index_long_term_memories")
    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_extract_discrete_memories_llm_error_handling(
        self,
        mock_get_client,
        mock_get_adapter,
        mock_index_memories,
        sample_message_memories,
    ):
        """Test error handling when LLM returns invalid JSON"""
        # Mock the LLM client to return invalid JSON
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="invalid json"))]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        unprocessed_memories = [
            mem
            for mem in sample_message_memories
            if mem.discrete_memory_extracted == "f"
        ]

        mock_search_result_1 = Mock()
        mock_search_result_1.memories = unprocessed_memories[
            :1
        ]  # Just one memory to test error
        mock_search_result_2 = Mock()
        mock_search_result_2.memories = []

        mock_adapter.search_memories.side_effect = [
            mock_search_result_1,
            mock_search_result_2,
        ]
        mock_get_adapter.return_value = mock_adapter

        # Mock index_long_term_memories
        mock_index_memories.return_value = None

        # Run the extraction - should handle the error gracefully
        with pytest.raises(
            (json.JSONDecodeError, tenacity.RetryError)
        ):  # Should raise due to retry exhaustion
            await extract_discrete_memories(deduplicate=True)

        # Verify that LLM was called but update was not called due to error
        assert mock_client.create_chat_completion.call_count >= 1
        mock_adapter.update_memories.assert_not_called()


@pytest.mark.requires_api_keys
class TestTopicExtractionIntegration:
    @pytest.mark.asyncio
    async def test_bertopic_integration(self):
        """Integration test for BERTopic topic extraction (skipped if not available)"""

        # Save and set topic_model_source
        original_source = settings.topic_model_source
        original_enable_topic_extraction = settings.enable_topic_extraction
        original_enable_ner = settings.enable_ner
        settings.enable_topic_extraction = True
        settings.enable_ner = True
        settings.topic_model_source = "BERTopic"
        settings.topic_model = "MaartenGr/BERTopic_Wikipedia"

        sample_text = (
            "OpenAI and Google are leading companies in artificial intelligence."
        )
        try:
            # Try to import BERTopic and check model loading
            topics = extract_topics_bertopic(sample_text)
            assert isinstance(topics, list)
            expected_keywords = {
                "generative",
                "transformer",
                "neural",
                "learning",
                "trained",
                "multimodal",
                "generates",
                "models",
                "encoding",
                "text",
            }
            assert any(t.lower() in expected_keywords for t in topics)
        finally:
            settings.topic_model_source = original_source
            settings.enable_topic_extraction = original_enable_topic_extraction
            settings.enable_ner = original_enable_ner

    @pytest.mark.asyncio
    async def test_llm_integration(self):
        """Integration test for LLM-based topic extraction (skipped if no API key)"""

        # Save and set topic_model_source
        original_source = settings.topic_model_source
        settings.topic_model_source = "LLM"
        sample_text = (
            "OpenAI and Google are leading companies in artificial intelligence."
        )
        try:
            # Check for API key
            if not (settings.openai_api_key or settings.anthropic_api_key):
                pytest.skip("No LLM API key available for integration test.")
            topics = await extract_topics_llm(sample_text)
            assert isinstance(topics, list)
            assert any(
                t.lower() in ["technology", "business", "artificial intelligence"]
                for t in topics
            )
        finally:
            settings.topic_model_source = original_source


class TestHandleExtractionPathSelection:
    @pytest.mark.asyncio
    @patch("agent_memory_server.extraction.extract_topics_bertopic")
    @patch("agent_memory_server.extraction.extract_topics_llm")
    async def test_handle_extraction_path_selection(
        self, mock_extract_topics_llm, mock_extract_topics_bertopic
    ):
        """Test that handle_extraction uses the correct extraction path based on settings.topic_model_source"""

        sample_text = (
            "OpenAI and Google are leading companies in artificial intelligence."
        )
        original_source = settings.topic_model_source
        original_enable_topic_extraction = settings.enable_topic_extraction
        original_enable_ner = settings.enable_ner
        try:
            # Enable topic extraction and disable NER for clarity
            settings.enable_topic_extraction = True
            settings.enable_ner = False

            # Test BERTopic path
            settings.topic_model_source = "BERTopic"
            mock_extract_topics_bertopic.return_value = ["technology"]
            mock_extract_topics_llm.return_value = ["should not be called"]
            topics, _ = await handle_extraction(sample_text)
            mock_extract_topics_bertopic.assert_called_once()
            mock_extract_topics_llm.assert_not_called()
            assert topics == ["technology"]
            mock_extract_topics_bertopic.reset_mock()

            # Test LLM path
            settings.topic_model_source = "LLM"
            mock_extract_topics_llm.return_value = ["ai"]
            topics, _ = await handle_extraction(sample_text)
            mock_extract_topics_llm.assert_called_once()
            mock_extract_topics_bertopic.assert_not_called()
            assert topics == ["ai"]
        finally:
            settings.topic_model_source = original_source
            settings.enable_topic_extraction = original_enable_topic_extraction
            settings.enable_ner = original_enable_ner
