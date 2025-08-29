from unittest.mock import Mock, patch

import numpy as np
import pytest
import ulid

from agent_memory_server.config import settings
from agent_memory_server.extraction import (
    extract_entities,
    extract_topics_bertopic,
    extract_topics_llm,
    handle_extraction,
)
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
            topics = await extract_topics_llm(sample_text)
            assert isinstance(topics, list)
            # Expect some relevant topic
            assert len(topics) > 0
        finally:
            settings.topic_model_source = original_source


@pytest.mark.asyncio
class TestHandleExtractionPathSelection:
    @patch("agent_memory_server.extraction.extract_topics_bertopic")
    @patch("agent_memory_server.extraction.extract_topics_llm")
    @patch("agent_memory_server.extraction.extract_entities")
    async def test_handle_extraction_path_selection(
        self,
        mock_extract_entities,
        mock_extract_topics_llm,
        mock_extract_topics_bertopic,
    ):
        """Test that handle_extraction selects the correct extraction method"""

        # Test BERTopic path
        original_source = settings.topic_model_source
        settings.topic_model_source = "BERTopic"

        mock_extract_topics_bertopic.return_value = ["AI", "technology"]
        mock_extract_entities.return_value = ["OpenAI"]

        try:
            topics, entities = await handle_extraction("OpenAI develops AI")

            mock_extract_topics_bertopic.assert_called_once()
            mock_extract_topics_llm.assert_not_called()

        finally:
            settings.topic_model_source = original_source

        # Test LLM path
        settings.topic_model_source = "LLM"
        mock_extract_topics_bertopic.reset_mock()
        mock_extract_topics_llm.reset_mock()
        mock_extract_topics_llm.return_value = ["AI", "machine learning"]

        try:
            topics, entities = await handle_extraction("OpenAI develops AI")

            mock_extract_topics_llm.assert_called_once()
            # BERTopic should not be called for LLM path
            mock_extract_topics_bertopic.assert_not_called()

        finally:
            settings.topic_model_source = original_source
