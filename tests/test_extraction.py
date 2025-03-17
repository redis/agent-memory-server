import pytest

from redis_memory_server.extraction import (
    EXTRACTION_PROMPT,
    extract_topics_and_entities,
    handle_extraction,
)
from redis_memory_server.models import MemoryMessage


class ChatResponse:
    """Mock ChatResponse class to match the real one"""

    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage or {"total_tokens": 100}
        self.total_tokens = usage.get("total_tokens", 100) if usage else 100


@pytest.mark.asyncio
class TestTopicAndEntityExtraction:
    async def test_extract_topics_and_entities_success(self, mock_openai_client):
        """Test successful topic and entity extraction"""
        message = "John and Sarah discussed AI technology at Google's headquarters in Mountain View."

        # Mock LLM response with properly formatted JSON
        mock_response = ChatResponse(
            choices=[
                {
                    "message": {
                        "content": '{"topics": ["AI technology", "business meeting"], "entities": ["John", "Sarah", "Google", "Mountain View"]}'
                    }
                }
            ]
        )
        mock_openai_client.create_chat_completion.return_value = mock_response

        topics, entities = await extract_topics_and_entities(
            message, mock_openai_client
        )

        assert topics == ["AI technology", "business meeting"]
        assert entities == ["John", "Sarah", "Google", "Mountain View"]
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message)
        )

    async def test_extract_topics_and_entities_invalid_json(self, mock_openai_client):
        """Test handling of invalid JSON response"""
        message = "Test message"
        mock_response = ChatResponse(choices=[{"message": {"content": "invalid json"}}])
        mock_openai_client.create_chat_completion.return_value = mock_response

        topics, entities = await extract_topics_and_entities(
            message, mock_openai_client
        )

        assert topics == []
        assert entities == []
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message)
        )

    async def test_extract_topics_and_entities_invalid_format(self, mock_openai_client):
        """Test handling of valid JSON but invalid format"""
        message = "Test message"
        mock_response = ChatResponse(
            choices=[{"message": {"content": '{"wrong_key": "wrong_value"}'}}]
        )
        mock_openai_client.create_chat_completion.return_value = mock_response

        topics, entities = await extract_topics_and_entities(
            message, mock_openai_client
        )

        assert topics == []
        assert entities == []
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message)
        )

    async def test_extract_topics_and_entities_exception(self, mock_openai_client):
        """Test handling of LLM client exception"""
        message = "Test message"
        mock_openai_client.create_chat_completion.side_effect = Exception("API error")

        topics, entities = await extract_topics_and_entities(
            message, mock_openai_client
        )

        assert topics == []
        assert entities == []
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message)
        )


@pytest.mark.asyncio
class TestHandleExtraction:
    async def test_handle_extraction_new_message(self, mock_openai_client):
        """Test extraction for a new message without existing topics/entities"""
        message = MemoryMessage(
            role="user",
            content="John and Sarah discussed AI at Google.",
            topics=[],
            entities=[],
        )

        mock_response = ChatResponse(
            choices=[
                {
                    "message": {
                        "content": '{"topics": ["AI", "business discussion"], "entities": ["John", "Sarah", "Google"]}'
                    }
                }
            ]
        )
        mock_openai_client.create_chat_completion.return_value = mock_response

        updated_message = await handle_extraction(message, mock_openai_client)

        assert set(updated_message.topics) == {"AI", "business discussion"}
        assert set(updated_message.entities) == {"John", "Sarah", "Google"}
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message.content)
        )

    async def test_handle_extraction_existing_topics(self, mock_openai_client):
        """Test extraction with existing topics"""
        message = MemoryMessage(
            role="user",
            content="John and Sarah discussed AI at Google.",
            topics=["meeting"],
            entities=[],
        )

        mock_response = ChatResponse(
            choices=[
                {
                    "message": {
                        "content": '{"topics": ["AI", "business discussion"], "entities": ["John", "Sarah", "Google"]}'
                    }
                }
            ]
        )
        mock_openai_client.create_chat_completion.return_value = mock_response

        updated_message = await handle_extraction(message, mock_openai_client)

        assert set(updated_message.topics) == {"AI", "business discussion", "meeting"}
        assert set(updated_message.entities) == {"John", "Sarah", "Google"}
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message.content)
        )

    async def test_handle_extraction_skip_complete(self, mock_openai_client):
        """Test skipping extraction when topics and entities exist"""
        message = MemoryMessage(
            role="user",
            content="Test message",
            topics=["existing topic"],
            entities=["existing entity"],
        )

        updated_message = await handle_extraction(message, mock_openai_client)

        assert updated_message.topics == ["existing topic"]
        assert updated_message.entities == ["existing entity"]
        mock_openai_client.create_chat_completion.assert_not_called()

    async def test_handle_extraction_empty_response(self, mock_openai_client):
        """Test handling empty extraction response"""
        message = MemoryMessage(
            role="user",
            content="Test message",
            topics=[],
            entities=[],
        )

        mock_response = ChatResponse(
            choices=[{"message": {"content": '{"topics": [], "entities": []}'}}]
        )
        mock_openai_client.create_chat_completion.return_value = mock_response

        updated_message = await handle_extraction(message, mock_openai_client)

        assert updated_message.topics == []
        assert updated_message.entities == []
        mock_openai_client.create_chat_completion.assert_called_once_with(
            "gpt-4o-mini", EXTRACTION_PROMPT.format(message=message.content)
        )
