import json
from unittest.mock import AsyncMock, patch

import pytest
import ulid

from agent_memory_server.llm_client import ChatCompletionResponse
from agent_memory_server.memory_strategies import get_memory_strategy
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum


@pytest.fixture
def mock_vectorstore_adapter():
    """Mock vectorstore adapter for testing"""
    return AsyncMock()


async def extract_memories_using_strategy(test_memories: list[MemoryRecord]):
    """Helper function to extract memories using the new memory strategy system.

    This replaces the old extract_discrete_memories function for tests.
    """
    # Get the discrete memory strategy
    strategy = get_memory_strategy("discrete")

    all_extracted_memories = []

    for memory in test_memories:
        # Extract memories using the new strategy
        extracted_data = await strategy.extract_memories(memory.text)

        # Convert to MemoryRecord objects for compatibility with existing tests
        for memory_data in extracted_data:
            memory_record = MemoryRecord(
                id=str(ulid.ULID()),
                text=memory_data["text"],
                memory_type=memory_data.get("type", "semantic"),
                topics=memory_data.get("topics", []),
                entities=memory_data.get("entities", []),
                session_id=memory.session_id,
                user_id=memory.user_id,
                discrete_memory_extracted="t",
            )
            all_extracted_memories.append(memory_record)

    return all_extracted_memories


@pytest.mark.asyncio
class TestContextualGrounding:
    """Tests for contextual grounding in memory extraction.

    These tests ensure that when extracting memories from conversations,
    references to unnamed people, places, and relative times are properly
    grounded to absolute context.
    """

    async def test_pronoun_grounding_he_him(self):
        """Test grounding of 'he/him' pronouns to actual person names"""
        # Create test message with pronoun reference
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="John mentioned he prefers coffee over tea. I told him about the new cafe.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        # Mock the LLM response to properly ground the pronoun
        mock_response = ChatCompletionResponse(
            content=json.dumps(
                {
                    "memories": [
                        {
                            "type": "semantic",
                            "text": "John prefers coffee over tea",
                            "topics": ["preferences", "beverages"],
                            "entities": ["John", "coffee", "tea"],
                        },
                        {
                            "type": "episodic",
                            "text": "User recommended a new cafe to John",
                            "topics": ["recommendation", "cafe"],
                            "entities": ["User", "John", "cafe"],
                        },
                    ]
                }
            ),
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Extract memories using the new strategy system
            extracted_memories = await extract_memories_using_strategy([test_memory])

            # Check that extracted memories don't contain ungrounded pronouns
            memory_texts = [mem.text for mem in extracted_memories]
            assert any("John prefers coffee" in text for text in memory_texts)
            assert any(
                "John" in text and "recommended" in text for text in memory_texts
            )

            # Ensure no ungrounded pronouns remain
            for text in memory_texts:
                assert "he" not in text.lower() or "John" in text
                assert "him" not in text.lower() or "John" in text

    async def test_pronoun_grounding_she_her(self):
        """Test grounding of 'she/her' pronouns to actual person names"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Sarah said she loves hiking. I gave her some trail recommendations.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        # Mock the LLM response to properly ground the pronoun
        mock_response = ChatCompletionResponse(
            content=json.dumps(
                {
                    "memories": [
                        {
                            "type": "semantic",
                            "text": "Sarah loves hiking",
                            "topics": ["hobbies", "outdoor"],
                            "entities": ["Sarah", "hiking"],
                        },
                        {
                            "type": "episodic",
                            "text": "User provided trail recommendations to Sarah",
                            "topics": ["recommendation", "trails"],
                            "entities": ["User", "Sarah", "trails"],
                        },
                    ]
                }
            ),
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o-mini",
        )

        with patch(
            "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Extract memories using the new strategy system
            extracted_memories = await extract_memories_using_strategy([test_memory])
            memory_texts = [mem.text for mem in extracted_memories]

            assert any("Sarah loves hiking" in text for text in memory_texts)
            assert any(
                "Sarah" in text and "trail recommendations" in text
                for text in memory_texts
            )

            # Ensure no ungrounded pronouns remain
            for text in memory_texts:
                assert "she" not in text.lower() or "Sarah" in text
                assert "her" not in text.lower() or "Sarah" in text
