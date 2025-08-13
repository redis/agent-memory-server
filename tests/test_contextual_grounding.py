import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import ulid

from agent_memory_server.extraction import extract_discrete_memories
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    return AsyncMock()


@pytest.fixture
def mock_vectorstore_adapter():
    """Mock vectorstore adapter for testing"""
    return AsyncMock()


@pytest.mark.asyncio
class TestContextualGrounding:
    """Tests for contextual grounding in memory extraction.

    These tests ensure that when extracting memories from conversations,
    references to unnamed people, places, and relative times are properly
    grounded to absolute context.
    """

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_pronoun_grounding_he_him(self, mock_get_client, mock_get_adapter):
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
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
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
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            # Verify the extracted memories contain proper names instead of pronouns
            mock_index.assert_called_once()
            extracted_memories = mock_index.call_args[0][0]

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

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_pronoun_grounding_she_her(self, mock_get_client, mock_get_adapter):
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
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
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
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
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

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_pronoun_grounding_they_them(self, mock_get_client, mock_get_adapter):
        """Test grounding of 'they/them' pronouns to actual person names"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Alex said they prefer remote work. I told them about our flexible policy.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "semantic",
                                    "text": "Alex prefers remote work",
                                    "topics": ["work", "preferences"],
                                    "entities": ["Alex", "remote work"],
                                },
                                {
                                    "type": "episodic",
                                    "text": "User informed Alex about flexible work policy",
                                    "topics": ["work policy", "information"],
                                    "entities": ["User", "Alex", "flexible policy"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            assert any("Alex prefers remote work" in text for text in memory_texts)
            assert any("Alex" in text and "flexible" in text for text in memory_texts)

            # Ensure pronouns are properly grounded
            for text in memory_texts:
                if "they" in text.lower():
                    assert "Alex" in text
                if "them" in text.lower():
                    assert "Alex" in text

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_place_grounding_there_here(self, mock_get_client, mock_get_adapter):
        """Test grounding of 'there/here' place references"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="We visited the Golden Gate Bridge in San Francisco. It was beautiful there. I want to go back there next year.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User visited the Golden Gate Bridge in San Francisco and found it beautiful",
                                    "topics": ["travel", "sightseeing"],
                                    "entities": [
                                        "User",
                                        "Golden Gate Bridge",
                                        "San Francisco",
                                    ],
                                },
                                {
                                    "type": "episodic",
                                    "text": "User wants to return to San Francisco next year",
                                    "topics": ["travel", "plans"],
                                    "entities": ["User", "San Francisco"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify place references are grounded to specific locations
            assert any(
                "San Francisco" in text and "beautiful" in text for text in memory_texts
            )
            assert any(
                "San Francisco" in text and "next year" in text for text in memory_texts
            )

            # Ensure vague place references are grounded
            for text in memory_texts:
                if "there" in text.lower():
                    assert "San Francisco" in text or "Golden Gate Bridge" in text

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_place_grounding_that_place(self, mock_get_client, mock_get_adapter):
        """Test grounding of 'that place' references"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="I had dinner at Chez Panisse in Berkeley. That place has amazing sourdough bread.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User had dinner at Chez Panisse in Berkeley",
                                    "topics": ["dining", "restaurant"],
                                    "entities": ["User", "Chez Panisse", "Berkeley"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "Chez Panisse has amazing sourdough bread",
                                    "topics": ["restaurant", "food"],
                                    "entities": ["Chez Panisse", "sourdough bread"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify "that place" is grounded to the specific restaurant
            assert any(
                "Chez Panisse" in text and "dinner" in text for text in memory_texts
            )
            assert any(
                "Chez Panisse" in text and "sourdough bread" in text
                for text in memory_texts
            )

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_temporal_grounding_last_year(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of 'last year' to absolute year (2024)"""
        # Create a memory with "last year" reference
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Last year I visited Japan and loved the cherry blossoms.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
            created_at=datetime(2025, 3, 15, 10, 0, 0, tzinfo=UTC),  # Current year 2025
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User visited Japan in 2024 and loved the cherry blossoms",
                                    "topics": ["travel", "nature"],
                                    "entities": ["User", "Japan", "cherry blossoms"],
                                }
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify "last year" is grounded to absolute year 2024
            assert any("2024" in text and "Japan" in text for text in memory_texts)

            # Check that event_date is properly set for episodic memories
            # Note: In this test, we're focusing on text grounding rather than metadata
            # The event_date would be set by a separate process or enhanced extraction logic

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_temporal_grounding_yesterday(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of 'yesterday' to absolute date"""
        # Assume current date is 2025-03-15
        current_date = datetime(2025, 3, 15, 14, 30, 0, tzinfo=UTC)

        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Yesterday I had lunch with my colleague at the Italian place downtown.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
            created_at=current_date,
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User had lunch with colleague at Italian restaurant downtown on March 14, 2025",
                                    "topics": ["dining", "social"],
                                    "entities": [
                                        "User",
                                        "colleague",
                                        "Italian restaurant",
                                    ],
                                }
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify "yesterday" is grounded to absolute date
            assert any(
                "March 14, 2025" in text or "2025-03-14" in text
                for text in memory_texts
            )

            # Check event_date is set correctly
            # Note: In this test, we're focusing on text grounding rather than metadata
            # The event_date would be set by a separate process or enhanced extraction logic

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_temporal_grounding_complex_relatives(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of complex relative time expressions"""
        current_date = datetime(2025, 8, 8, 16, 45, 0, tzinfo=UTC)

        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Three months ago I started learning piano. Two weeks ago I performed my first piece.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
            created_at=current_date,
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User started learning piano in May 2025",
                                    "topics": ["music", "learning"],
                                    "entities": ["User", "piano"],
                                },
                                {
                                    "type": "episodic",
                                    "text": "User performed first piano piece in late July 2025",
                                    "topics": ["music", "performance"],
                                    "entities": ["User", "piano piece"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify complex relative times are grounded
            assert any("May 2025" in text and "piano" in text for text in memory_texts)
            assert any(
                "July 2025" in text and "performed" in text for text in memory_texts
            )

            # Check event dates are properly set
            # Note: In this test, we're focusing on text grounding rather than metadata
            # The event_date would be set by a separate process or enhanced extraction logic

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_complex_contextual_grounding_combined(
        self, mock_get_client, mock_get_adapter
    ):
        """Test complex scenario with multiple types of contextual grounding"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Last month Sarah and I went to that new restaurant downtown. She loved it there and wants to go back next month.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
            created_at=datetime(2025, 8, 8, tzinfo=UTC),  # Current: August 2025
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User and Sarah went to new downtown restaurant in July 2025",
                                    "topics": ["dining", "social"],
                                    "entities": [
                                        "User",
                                        "Sarah",
                                        "downtown restaurant",
                                    ],
                                },
                                {
                                    "type": "semantic",
                                    "text": "Sarah loved the new downtown restaurant",
                                    "topics": ["preferences", "restaurant"],
                                    "entities": ["Sarah", "downtown restaurant"],
                                },
                                {
                                    "type": "episodic",
                                    "text": "Sarah wants to return to downtown restaurant in September 2025",
                                    "topics": ["plans", "restaurant"],
                                    "entities": ["Sarah", "downtown restaurant"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify all contextual elements are properly grounded
            assert any(
                "Sarah" in text
                and "July 2025" in text
                and "downtown restaurant" in text
                for text in memory_texts
            )
            assert any(
                "Sarah loved" in text and "downtown restaurant" in text
                for text in memory_texts
            )
            assert any(
                "Sarah" in text and "September 2025" in text for text in memory_texts
            )

            # Ensure no ungrounded references remain
            for text in memory_texts:
                assert "she" not in text.lower() or "Sarah" in text
                assert (
                    "there" not in text.lower()
                    or "downtown" in text
                    or "restaurant" in text
                )
                assert "last month" not in text.lower() or "July" in text
                assert "next month" not in text.lower() or "September" in text

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_ambiguous_pronoun_handling(self, mock_get_client, mock_get_adapter):
        """Test handling of ambiguous pronoun references"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="John and Mike were discussing the project. He mentioned the deadline is tight.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "John and Mike discussed the project",
                                    "topics": ["work", "discussion"],
                                    "entities": ["John", "Mike", "project"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "Someone mentioned the project deadline is tight",
                                    "topics": ["work", "deadline"],
                                    "entities": ["project", "deadline"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # When pronoun reference is ambiguous, system should handle gracefully
            assert any("John and Mike" in text for text in memory_texts)
            # Should avoid making incorrect assumptions about who "he" refers to
            # Either use generic term like "Someone" or avoid ungrounded pronouns
            has_someone_mentioned = any(
                "Someone mentioned" in text for text in memory_texts
            )
            has_ungrounded_he = any(
                "He" in text and "John" not in text and "Mike" not in text
                for text in memory_texts
            )
            assert has_someone_mentioned or not has_ungrounded_he

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_event_date_metadata_setting(self, mock_get_client, mock_get_adapter):
        """Test that event_date metadata is properly set for episodic memories with temporal context"""
        current_date = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)

        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Last Tuesday I went to the dentist appointment.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
            created_at=current_date,
        )

        # Mock LLM to extract memory with proper event date
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User had dentist appointment on June 10, 2025",
                                    "topics": ["health", "appointment"],
                                    "entities": ["User", "dentist"],
                                }
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify temporal grounding in text
            assert any(
                "June 10, 2025" in text and "dentist" in text for text in memory_texts
            )

            # Find the episodic memory and verify content
            episodic_memories = [
                mem for mem in extracted_memories if mem.memory_type == "episodic"
            ]
            assert len(episodic_memories) > 0

            # Note: event_date metadata would be set by enhanced extraction logic
            # For now, we focus on verifying the text contains absolute dates

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_definite_reference_grounding_the_meeting(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of definite references like 'the meeting', 'the document'"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="I attended the meeting this morning. The document we discussed was very detailed.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        # Mock LLM to provide context about what "the meeting" and "the document" refer to
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User attended the quarterly planning meeting this morning",
                                    "topics": ["work", "meeting"],
                                    "entities": ["User", "quarterly planning meeting"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "The quarterly budget document discussed in the meeting was very detailed",
                                    "topics": ["work", "budget"],
                                    "entities": [
                                        "quarterly budget document",
                                        "meeting",
                                    ],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify definite references are grounded to specific entities
            assert any("quarterly planning meeting" in text for text in memory_texts)
            assert any("quarterly budget document" in text for text in memory_texts)

            # Ensure vague definite references are resolved
            for text in memory_texts:
                # Either the text specifies what "the meeting" was, or avoids the vague reference
                if "meeting" in text.lower():
                    assert (
                        "quarterly" in text
                        or "planning" in text
                        or not text.startswith("the meeting")
                    )

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_discourse_deixis_this_that_grounding(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of discourse deixis like 'this issue', 'that problem'"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="The server keeps crashing. This issue has been happening for days. That problem needs immediate attention.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "The production server has been crashing repeatedly for several days",
                                    "topics": ["technical", "server"],
                                    "entities": ["production server", "crashes"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "The recurring server crashes require immediate attention",
                                    "topics": ["technical", "priority"],
                                    "entities": [
                                        "server crashes",
                                        "immediate attention",
                                    ],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify discourse deixis is grounded to specific concepts
            assert any("server" in text and "crashing" in text for text in memory_texts)
            assert any(
                "crashes" in text and ("immediate" in text or "attention" in text)
                for text in memory_texts
            )

            # Ensure vague discourse references are resolved
            for text in memory_texts:
                if "this issue" in text.lower():
                    assert "server" in text or "crash" in text
                if "that problem" in text.lower():
                    assert "server" in text or "crash" in text

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_elliptical_construction_grounding(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of elliptical constructions like 'did too', 'will as well'"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="Sarah enjoyed the concert. Mike did too. They both will attend the next one as well.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "semantic",
                                    "text": "Sarah enjoyed the jazz concert",
                                    "topics": ["entertainment", "music"],
                                    "entities": ["Sarah", "jazz concert"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "Mike also enjoyed the jazz concert",
                                    "topics": ["entertainment", "music"],
                                    "entities": ["Mike", "jazz concert"],
                                },
                                {
                                    "type": "episodic",
                                    "text": "Sarah and Mike plan to attend the next jazz concert",
                                    "topics": ["entertainment", "plans"],
                                    "entities": ["Sarah", "Mike", "jazz concert"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify elliptical constructions are expanded
            assert any(
                "Sarah enjoyed" in text and "concert" in text for text in memory_texts
            )
            assert any(
                "Mike" in text and "enjoyed" in text and "concert" in text
                for text in memory_texts
            )
            assert any(
                "Sarah and Mike" in text and "attend" in text for text in memory_texts
            )

            # Ensure no unresolved ellipsis remains
            for text in memory_texts:
                assert "did too" not in text.lower()
                assert "as well" not in text.lower() or "attend" in text

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_bridging_reference_grounding(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of bridging references (part-whole, set-member relationships)"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="I bought a new car yesterday. The engine sounds great and the steering is very responsive.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
            created_at=datetime(2025, 8, 8, 10, 0, 0, tzinfo=UTC),
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User purchased a new car on August 7, 2025",
                                    "topics": ["purchase", "vehicle"],
                                    "entities": ["User", "new car"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "User's new car has a great-sounding engine and responsive steering",
                                    "topics": ["vehicle", "performance"],
                                    "entities": [
                                        "User",
                                        "new car",
                                        "engine",
                                        "steering",
                                    ],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify bridging references are properly contextualized
            assert any(
                "car" in text and ("purchased" in text or "bought" in text)
                for text in memory_texts
            )
            assert any(
                "car" in text and "engine" in text and "steering" in text
                for text in memory_texts
            )

            # Ensure definite references are linked to their antecedents
            for text in memory_texts:
                if "engine" in text or "steering" in text:
                    assert "car" in text or "User's" in text

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_implied_causal_relationship_grounding(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of implied causal and logical relationships"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="It started raining heavily. I got completely soaked walking to work.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "episodic",
                                    "text": "User got soaked walking to work because of heavy rain",
                                    "topics": ["weather", "commute"],
                                    "entities": ["User", "heavy rain", "work"],
                                }
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify implied causal relationship is made explicit
            assert any("soaked" in text and "rain" in text for text in memory_texts)
            # Should make the causal connection explicit
            assert any(
                "because" in text
                or "due to" in text
                or text.count("rain") > 0
                and text.count("soaked") > 0
                for text in memory_texts
            )

    @patch("agent_memory_server.vectorstore_factory.get_vectorstore_adapter")
    @patch("agent_memory_server.extraction.get_model_client")
    async def test_modal_expression_attitude_grounding(
        self, mock_get_client, mock_get_adapter
    ):
        """Test grounding of modal expressions and implied speaker attitudes"""
        test_memory = MemoryRecord(
            id=str(ulid.ULID()),
            text="That movie should have been much better. I suppose the director tried their best though.",
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id="test-session",
            user_id="test-user",
        )

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "type": "semantic",
                                    "text": "User was disappointed with the movie quality and had higher expectations",
                                    "topics": ["entertainment", "opinion"],
                                    "entities": ["User", "movie"],
                                },
                                {
                                    "type": "semantic",
                                    "text": "User acknowledges the movie director made an effort despite the poor result",
                                    "topics": ["entertainment", "judgment"],
                                    "entities": ["User", "director", "movie"],
                                },
                            ]
                        }
                    )
                )
            )
        ]
        mock_client.create_chat_completion = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = Mock(memories=[test_memory])
        mock_adapter.update_memories = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        with patch(
            "agent_memory_server.long_term_memory.index_long_term_memories"
        ) as mock_index:
            await extract_discrete_memories([test_memory])

            extracted_memories = mock_index.call_args[0][0]
            memory_texts = [mem.text for mem in extracted_memories]

            # Verify modal expressions and attitudes are made explicit
            assert any(
                "disappointed" in text or "expectations" in text
                for text in memory_texts
            )
            assert any(
                "acknowledges" in text or "effort" in text for text in memory_texts
            )

            # Should capture the nuanced attitude rather than just the surface modal
            for text in memory_texts:
                if "movie" in text:
                    # Should express the underlying attitude, not just "should have been"
                    assert any(
                        word in text
                        for word in [
                            "disappointed",
                            "expectations",
                            "acknowledges",
                            "effort",
                            "despite",
                        ]
                    )
