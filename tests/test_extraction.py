from unittest.mock import Mock, patch

import numpy as np
import pytest

from agent_memory_server.config import settings
from agent_memory_server.extraction import (
    extract_entities,
    extract_topics,
    handle_extraction,
)


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


@pytest.mark.asyncio
class TestTopicExtraction:
    @patch("agent_memory_server.extraction.get_topic_model")
    async def test_extract_topics_success(self, mock_get_topic_model, mock_bertopic):
        """Test successful topic extraction"""
        mock_get_topic_model.return_value = mock_bertopic
        text = "Discussion about AI technology and business"

        topics = extract_topics(text)

        assert set(topics) == {"technology", "business"}
        mock_bertopic.transform.assert_called_once_with([text])

    @patch("agent_memory_server.extraction.get_topic_model")
    async def test_extract_topics_no_valid_topics(
        self, mock_get_topic_model, mock_bertopic
    ):
        """Test when no valid topics are found"""
        mock_bertopic.transform.return_value = (np.array([-1]), np.array([0.0]))
        mock_get_topic_model.return_value = mock_bertopic

        topics = extract_topics("Test message")

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
    @patch("agent_memory_server.extraction.extract_topics")
    @patch("agent_memory_server.extraction.extract_entities")
    async def test_handle_extraction(self, mock_extract_entities, mock_extract_topics):
        """Test extraction with topics/entities"""
        mock_extract_topics.return_value = ["AI", "business"]
        mock_extract_entities.return_value = ["John", "Sarah", "Google"]

        topics, entities = await handle_extraction(
            "John and Sarah discussed AI at Google."
        )

        # Check that both existing and new topics are present
        assert "AI" in topics
        assert "business" in topics
        assert len(topics) == 2

        # Check that both existing and new entities are present
        assert "Sarah" in entities
        assert "John" in entities
        assert "Google" in entities
        assert len(entities) == 3

    @patch("agent_memory_server.extraction.extract_topics")
    @patch("agent_memory_server.extraction.extract_entities")
    async def test_handle_extraction_disabled_features(
        self, mock_extract_entities, mock_extract_topics
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
            mock_extract_topics.assert_not_called()
            mock_extract_entities.assert_not_called()
        finally:
            # Restore settings
            settings.enable_topic_extraction = original_topic_setting
            settings.enable_ner = original_ner_setting
