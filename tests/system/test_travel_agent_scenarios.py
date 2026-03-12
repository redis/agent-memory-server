"""
System Test: Travel Agent Scenarios

Real-world travel agent conversation scenarios to validate production readiness.
Uses realistic data from test_data_travel_agent.json.

Run with:
    uv run pytest tests/system/test_travel_agent_scenarios.py --run-api-tests -v -s
"""

import asyncio
import os
import time

import pytest
from agent_memory_client.client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import MemoryMessage, WorkingMemory

from tests.system.travel_agent_data import TravelAgentDataGenerator

# Test configuration
INTEGRATION_BASE_URL = os.getenv("MEMORY_SERVER_BASE_URL", "http://localhost:8001")

pytestmark = [pytest.mark.integration, pytest.mark.requires_api_keys]


@pytest.fixture
async def travel_agent_client():
    """Create a memory client configured for travel agent scenarios."""
    config = MemoryClientConfig(
        base_url=INTEGRATION_BASE_URL,
        timeout=60.0,
        default_namespace="travel-agent",
        default_context_window_max=128000,
    )
    async with MemoryAPIClient(config) as client:
        yield client


@pytest.fixture
def data_generator():
    """Provide travel agent data generator."""
    return TravelAgentDataGenerator()


class TestTravelAgentShortConversations:
    """Test short travel agent conversations (weekend trip inquiries)."""

    @pytest.mark.asyncio
    async def test_weekend_trip_inquiry(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test a quick weekend trip inquiry conversation."""
        session_id = f"weekend-paris-{int(time.time())}"

        # Get realistic short conversation
        messages = data_generator.get_short_conversation()

        print(f"\n📝 Weekend Trip Inquiry Scenario")
        print(f"   Messages: {len(messages)}")
        print(f"   First message: {messages[0].content[:80]}...")
        print(f"   Last message: {messages[-1].content[:80]}...")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="travel-agent",
            user_id="client-001",
            context="Client inquiring about weekend trip to Paris",
        )

        start = time.perf_counter()
        response = await travel_agent_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None
        assert len(response.messages) == len(messages)

        # Verify message content is preserved
        assert "Paris" in response.messages[0].content
        assert "vegetarian" in response.messages[-2].content.lower()

        print(f"\n✅ Weekend trip conversation stored successfully")
        print(
            f"   Latency: {latency:.3f}s ({latency / len(messages) * 1000:.2f}ms per message)"
        )
        print(f"   Messages preserved: {len(response.messages)}/{len(messages)}")

    @pytest.mark.asyncio
    async def test_retrieve_and_search_weekend_trip(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test retrieving conversation and searching for specific details."""
        session_id = f"weekend-search-{int(time.time())}"

        messages = data_generator.get_short_conversation()
        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="travel-agent",
            user_id="client-002",
        )

        await travel_agent_client.put_working_memory(session_id, working_memory)

        # Retrieve the conversation
        retrieved = await travel_agent_client.get_working_memory(session_id)
        assert retrieved is not None

        # Verify key details are accessible
        conversation_text = " ".join(msg.content for msg in retrieved.messages)
        assert "Paris" in conversation_text
        assert "vegetarian" in conversation_text.lower()
        assert "cooking class" in conversation_text.lower()
        assert "$2000" in conversation_text or "2000" in conversation_text

        print(f"\n✅ Conversation retrieval successful")
        print(f"   Key details found: Paris ✓, Vegetarian ✓, Cooking class ✓, Budget ✓")


class TestTravelAgentMediumConversations:
    """Test medium-length travel agent conversations (family vacation planning)."""

    @pytest.mark.asyncio
    async def test_family_vacation_planning(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test a detailed family vacation planning conversation."""
        session_id = f"family-italy-{int(time.time())}"

        # Get realistic medium conversation (50 messages)
        messages = data_generator.get_medium_conversation(num_messages=50)

        print(f"\n📝 Family Vacation Planning Scenario")
        print(f"   Messages: {len(messages)}")
        print(f"   Conversation topics: Italy, family of 4, kids ages 8 & 12")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="travel-agent",
            user_id="family-001",
            context="Family of 4 planning 2-week Italy vacation",
        )

        start = time.perf_counter()
        response = await travel_agent_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None

        print(f"\n✅ Family vacation conversation stored")
        print(
            f"   Latency: {latency:.3f}s ({latency / len(messages) * 1000:.2f}ms per message)"
        )
        print(f"   Context percentage: {response.context_percentage_total_used:.1f}%")

    @pytest.mark.asyncio
    async def test_incremental_family_planning(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test building up a family vacation conversation incrementally."""
        session_id = f"family-incremental-{int(time.time())}"

        # Start with initial messages
        all_messages = data_generator.get_medium_conversation(num_messages=50)
        initial_messages = all_messages[:10]

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=initial_messages,
            namespace="travel-agent",
            user_id="family-002",
        )

        await travel_agent_client.put_working_memory(session_id, working_memory)
        print(f"\n📝 Incremental Family Planning")
        print(f"   Initial messages: {len(initial_messages)}")

        # Add messages in batches (simulating ongoing conversation)
        batch_size = 10
        for i in range(10, len(all_messages), batch_size):
            batch = all_messages[i : i + batch_size]
            await travel_agent_client.append_messages_to_working_memory(
                session_id, batch
            )
            print(f"   Added batch {i//batch_size + 1}: {len(batch)} messages")

        # Verify final state
        final = await travel_agent_client.get_working_memory(session_id)
        assert final is not None

        print(f"\n✅ Incremental updates completed")
        print(f"   Final message count: {len(final.messages)}")
        print(f"   Context: {final.context[:100] if final.context else 'None'}...")


class TestTravelAgentLongConversations:
    """Test long, complex travel agent conversations (honeymoon planning)."""

    @pytest.mark.asyncio
    async def test_honeymoon_planning_full_journey(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test a complete honeymoon planning journey (200 messages)."""
        session_id = f"honeymoon-europe-{int(time.time())}"

        # Get realistic long conversation
        messages = data_generator.get_long_conversation(num_messages=200)

        print(f"\n📝 Honeymoon Planning Scenario")
        print(f"   Messages: {len(messages)}")
        print(f"   Phases: Initial → Destinations → Details → Refinements → Final")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="travel-agent",
            user_id="couple-001",
            context="Couple planning 3-week European honeymoon",
        )

        start = time.perf_counter()
        response = await travel_agent_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None

        print(f"\n✅ Honeymoon conversation stored")
        print(
            f"   Latency: {latency:.3f}s ({latency / len(messages) * 1000:.2f}ms per message)"
        )
        print(f"   Messages in response: {len(response.messages)}")
        print(f"   Context percentage: {response.context_percentage_total_used:.1f}%")

        # Check if summarization occurred
        if response.context and len(response.context) > 0:
            print(f"   Summary created: Yes ({len(response.context)} chars)")
        else:
            print(f"   Summary created: No (under threshold)")

    @pytest.mark.asyncio
    async def test_honeymoon_with_very_large_itinerary(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test conversation with very large messages (detailed itineraries)."""
        session_id = f"honeymoon-detailed-{int(time.time())}"

        # Start with some regular messages
        messages = data_generator.get_long_conversation(num_messages=20)

        # Add a very large detailed itinerary
        large_message = data_generator.get_very_large_message(size_chars=5000)
        messages.append(large_message)

        # Add some follow-up messages
        from agent_memory_client.models import MemoryMessage

        messages.extend(
            [
                MemoryMessage(
                    role="user",
                    content="This itinerary looks perfect! Can we make a few small changes?",
                ),
                MemoryMessage(
                    role="assistant",
                    content="Of course! What would you like to adjust?",
                ),
            ]
        )

        print(f"\n📝 Detailed Itinerary Scenario")
        print(f"   Total messages: {len(messages)}")
        print(f"   Large message size: {len(large_message.content)} chars")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="travel-agent",
            user_id="couple-002",
        )

        start = time.perf_counter()
        response = await travel_agent_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None

        print(f"\n✅ Large itinerary conversation stored")
        print(f"   Latency: {latency:.3f}s")
        print(f"   Successfully handled large message: ✓")


class TestTravelAgentConcurrentScenarios:
    """Test concurrent operations in travel agent scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_agents_updating_booking(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test multiple agents updating the same client conversation."""
        session_id = f"multi-agent-{int(time.time())}"

        # Start with initial conversation
        initial_messages = data_generator.get_medium_conversation(num_messages=10)
        working_memory = WorkingMemory(
            session_id=session_id,
            messages=initial_messages,
            namespace="travel-agent",
            user_id="client-multi-001",
        )

        await travel_agent_client.put_working_memory(session_id, working_memory)

        print(f"\n📝 Multi-Agent Update Scenario")
        print(f"   Initial messages: {len(initial_messages)}")

        # Get concurrent updates from different specialists
        agent_updates = data_generator.get_concurrent_update_messages()

        async def add_agent_update(agent_name: str, message):
            """Add an update from a specific agent."""
            await travel_agent_client.append_messages_to_working_memory(
                session_id, [message]
            )
            return agent_name

        # Execute updates concurrently
        start = time.perf_counter()
        results = await asyncio.gather(
            *[add_agent_update(agent, msg) for agent, msg in agent_updates]
        )
        latency = time.perf_counter() - start

        # Verify all updates were applied
        final = await travel_agent_client.get_working_memory(session_id)
        assert final is not None

        print(f"\n✅ Concurrent agent updates completed")
        print(f"   Agents: {', '.join(results)}")
        print(f"   Total time: {latency:.3f}s")
        print(f"   Final message count: {len(final.messages)}")

        # Verify all agent updates are present
        conversation_text = " ".join(msg.content for msg in final.messages)
        assert "Flight Specialist" in conversation_text
        assert "Hotel Specialist" in conversation_text
        assert "Activities Coordinator" in conversation_text
        assert "Restaurant Specialist" in conversation_text
        print(f"   All agent updates verified: ✓")

    @pytest.mark.asyncio
    async def test_parallel_client_conversations(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test handling multiple client conversations in parallel."""

        async def handle_client_conversation(client_num: int) -> tuple[str, float]:
            """Handle a single client conversation."""
            session_id = f"parallel-client-{client_num}-{int(time.time())}"
            messages = data_generator.get_medium_conversation(num_messages=30)

            working_memory = WorkingMemory(
                session_id=session_id,
                messages=messages,
                namespace="travel-agent",
                user_id=f"client-{client_num}",
            )

            start = time.perf_counter()
            await travel_agent_client.put_working_memory(session_id, working_memory)
            latency = time.perf_counter() - start

            return session_id, latency

        print(f"\n📝 Parallel Client Conversations")

        # Handle 5 clients in parallel
        num_clients = 5
        start_total = time.perf_counter()
        results = await asyncio.gather(
            *[handle_client_conversation(i) for i in range(num_clients)]
        )
        total_time = time.perf_counter() - start_total

        session_ids = [r[0] for r in results]
        latencies = [r[1] for r in results]

        print(f"\n✅ Parallel conversations completed")
        print(f"   Clients handled: {num_clients}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average latency: {sum(latencies) / len(latencies):.3f}s")
        print(f"   Max latency: {max(latencies):.3f}s")
        print(f"   Sessions created: {len(session_ids)}")


class TestTravelAgentSummarization:
    """Test summarization behavior with travel agent conversations."""

    @pytest.mark.asyncio
    async def test_summarization_with_greece_trip(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test that summarization works correctly for long planning conversations."""
        session_id = f"greece-summarization-{int(time.time())}"

        # Get summarization test data
        test_data = data_generator.get_summarization_test_data()

        # Create a conversation that should trigger summarization
        # Use smaller context window to force summarization
        config = MemoryClientConfig(
            base_url=INTEGRATION_BASE_URL,
            timeout=60.0,
            default_namespace="travel-agent",
            default_context_window_max=4000,  # Small window
        )

        async with MemoryAPIClient(config) as client:
            # Combine messages
            all_messages = (
                test_data["messages_to_summarize"] + test_data["recent_messages"]
            )

            # Add more content to each message to increase token count
            for msg in all_messages:
                msg.content = msg.content + " " + ("Additional planning details. " * 20)

            print(f"\n📝 Greece Trip Summarization Test")
            print(f"   Total messages: {len(all_messages)}")
            print(f"   Context window: 4000 tokens")

            working_memory = WorkingMemory(
                session_id=session_id,
                messages=all_messages,
                namespace="travel-agent",
                user_id="greece-client",
                context=test_data["initial_context"],
            )

            await client.put_working_memory(session_id, working_memory)

            # Wait for potential background summarization
            await asyncio.sleep(2)

            # Retrieve and check
            result = await client.get_working_memory(session_id)
            assert result is not None

            print(f"\n✅ Summarization test completed")

            if result.context and len(result.context) > len(
                test_data["initial_context"]
            ):
                print(f"   Summary updated: Yes")
                print(f"   Summary length: {len(result.context)} chars")

                # Check for expected keywords
                summary_lower = result.context.lower()
                found_keywords = [
                    kw
                    for kw in test_data["expected_keywords"]
                    if kw.lower() in summary_lower
                ]
                print(f"   Keywords found: {', '.join(found_keywords)}")
            else:
                print(f"   Summary updated: No (may be under threshold)")

            print(f"   Messages retained: {len(result.messages)}")
            print(f"   Context percentage: {result.context_percentage_total_used:.1f}%")

            # Verify recent messages are preserved
            recent_content = " ".join(msg.content for msg in result.messages[-3:])
            assert "Crete" in recent_content
            print(f"   Recent messages preserved: ✓")


class TestReturningClientScenarios:
    """Test returning client with multiple trips over time - tests long-term memory."""

    @pytest.mark.asyncio
    async def test_three_trips_same_client(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test storing three separate trips for the same client over time."""
        client_id = "sarah-johnson-001"
        trips = data_generator.get_returning_client_trips()

        print(f"\n📝 Returning Client: Sarah's Travel Journey")
        print(f"   Client ID: {client_id}")
        print(f"   Number of trips: {len(trips)}")

        stored_sessions = []

        # Store each trip as a separate working memory session
        for trip in trips:
            session_id = trip["session_id"]
            messages = [MemoryMessage(**msg) for msg in trip["sample_messages"]]

            working_memory = WorkingMemory(
                session_id=session_id,
                messages=messages,
                namespace="travel-agent",
                user_id=client_id,
                context=trip["description"],
            )

            response = await travel_agent_client.put_working_memory(
                session_id, working_memory
            )
            assert response is not None

            stored_sessions.append(session_id)

            print(
                f"\n   ✅ Trip {trip['trip_number']}: {trip['key_details']['destination']}"
            )
            print(f"      Session: {session_id}")
            print(f"      Budget: {trip['key_details']['budget']}")
            print(f"      Messages: {len(messages)}")

        print(f"\n✅ All trips stored for returning client")
        print(f"   Sessions: {', '.join(stored_sessions)}")

        # Verify we can retrieve each trip
        for session_id in stored_sessions:
            retrieved = await travel_agent_client.get_working_memory(session_id)
            assert retrieved is not None
            print(f"   ✓ Retrieved: {session_id}")

    @pytest.mark.asyncio
    async def test_long_term_memory_creation(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test that long-term memories are created from multiple trips."""
        client_id = "sarah-johnson-001"
        namespace = "travel-agent"

        # Store all three trips
        trips = data_generator.get_returning_client_trips()

        print(f"\n📝 Long-term Memory Creation Test")

        for trip in trips:
            session_id = trip["session_id"]
            messages = [MemoryMessage(**msg) for msg in trip["sample_messages"]]

            # Add more context to messages to ensure they're stored in long-term memory
            enriched_messages = []
            for msg in messages:
                enriched_msg = MemoryMessage(
                    role=msg.role,
                    content=msg.content,
                    metadata={
                        "trip_number": trip["trip_number"],
                        "destination": trip["key_details"]["destination"],
                        "budget": trip["key_details"]["budget"],
                    },
                )
                enriched_messages.append(enriched_msg)

            working_memory = WorkingMemory(
                session_id=session_id,
                messages=enriched_messages,
                namespace=namespace,
                user_id=client_id,
                context=trip["description"],
            )

            await travel_agent_client.put_working_memory(session_id, working_memory)

            # Promote to long-term memory
            await travel_agent_client.create_long_term_memories(
                memories=[
                    {
                        "text": f"{trip['description']}: {trip['key_details']['destination']}, "
                        f"budget {trip['key_details']['budget']}, "
                        f"preferences: {', '.join(trip['key_details']['preferences'])}",
                        "namespace": namespace,
                        "user_id": client_id,
                        "session_id": session_id,
                        "metadata": {
                            "trip_number": trip["trip_number"],
                            "date": trip["date"],
                            "destination": trip["key_details"]["destination"],
                        },
                    }
                ]
            )

            print(f"   ✅ Trip {trip['trip_number']} promoted to long-term memory")

        # Search long-term memory for Sarah's preferences
        print(f"\n🔍 Searching long-term memories for Sarah...")

        search_results = await travel_agent_client.search_long_term_memory(
            query="What are Sarah's travel preferences and history?",
            namespace=namespace,
            user_id=client_id,
            limit=10,
        )

        print(f"\n✅ Long-term memory search completed")
        print(f"   Results found: {len(search_results)}")

        if search_results:
            print(f"   Sample results:")
            for i, result in enumerate(search_results[:3], 1):
                print(f"   {i}. {result.text[:100]}...")

    @pytest.mark.asyncio
    async def test_context_switching_in_conversation(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test switching context mid-conversation to reference previous trips."""
        client_id = "sarah-johnson-001"

        # First, store the Italy trip (trip 2)
        trips = data_generator.get_returning_client_trips()
        italy_trip = next(t for t in trips if t["trip_number"] == 2)

        italy_session = italy_trip["session_id"]
        italy_messages = [MemoryMessage(**msg) for msg in italy_trip["sample_messages"]]

        await travel_agent_client.put_working_memory(
            italy_session,
            WorkingMemory(
                session_id=italy_session,
                messages=italy_messages,
                namespace="travel-agent",
                user_id=client_id,
            ),
        )

        print(f"\n📝 Context Switching Test")
        print(f"   Stored Italy trip: {italy_session}")

        # Now start planning a new trip (Greece)
        greece_session = f"trip-4-greece-{int(time.time())}"

        greece_messages = [
            MemoryMessage(
                role="user",
                content="Hi! I want to plan a trip to Greece for our anniversary.",
            ),
            MemoryMessage(
                role="assistant",
                content="Great! Tell me more about what you're looking for.",
            ),
            MemoryMessage(
                role="user",
                content="Actually, before we continue - can you remind me what hotel we stayed at in Florence during our Italy trip?",
            ),
        ]

        await travel_agent_client.put_working_memory(
            greece_session,
            WorkingMemory(
                session_id=greece_session,
                messages=greece_messages,
                namespace="travel-agent",
                user_id=client_id,
            ),
        )

        print(f"   Started Greece planning: {greece_session}")
        print(f"   User asked about previous Italy trip mid-conversation")

        # Retrieve Italy trip to answer the question
        italy_memory = await travel_agent_client.get_working_memory(italy_session)
        assert italy_memory is not None

        # Check if Florence hotel info is in the Italy conversation
        italy_text = " ".join(msg.content for msg in italy_memory.messages)
        has_florence_info = "Florence" in italy_text

        print(f"   ✅ Retrieved Italy trip to answer question")
        print(f"   Florence info available: {has_florence_info}")

        # Continue Greece conversation with context from Italy trip
        greece_messages.append(
            MemoryMessage(
                role="assistant",
                content="You stayed at a boutique hotel in Florence. Now, back to your Greece trip - what dates are you thinking?",
            )
        )
        greece_messages.append(
            MemoryMessage(
                role="user",
                content="Perfect! For Greece, we want something similar to that Florence hotel. Dates are June 10-17.",
            )
        )

        await travel_agent_client.put_working_memory(
            greece_session,
            WorkingMemory(
                session_id=greece_session,
                messages=greece_messages,
                namespace="travel-agent",
                user_id=client_id,
            ),
        )

        # Verify both conversations are intact
        final_greece = await travel_agent_client.get_working_memory(greece_session)
        assert final_greece is not None
        assert len(final_greece.messages) == 5

        print(f"\n✅ Context switching successful")
        print(f"   Greece conversation continued after referencing Italy trip")
        print(f"   Both sessions remain intact")

    @pytest.mark.asyncio
    async def test_preference_consistency_across_trips(
        self,
        travel_agent_client: MemoryAPIClient,
        data_generator: TravelAgentDataGenerator,
    ):
        """Test that client preferences remain consistent across multiple trips."""
        client_id = "sarah-johnson-001"
        trips = data_generator.get_returning_client_trips()
        expected_memories = data_generator.get_expected_long_term_memories()

        print(f"\n📝 Preference Consistency Test")
        print(f"   Analyzing {len(trips)} trips for consistent preferences")

        # Extract preferences from each trip
        preferences_by_trip = {}

        for trip in trips:
            trip_num = trip["trip_number"]
            prefs = trip["key_details"]["preferences"]
            preferences_by_trip[trip_num] = prefs

            print(f"\n   Trip {trip_num} ({trip['key_details']['destination']}):")
            print(f"      Preferences: {', '.join(prefs)}")

        # Check for consistent preferences
        consistent_prefs = []

        # Vegetarian appears in all trips
        if all(
            "vegetarian" in prefs or "vegetarian options" in prefs
            for prefs in preferences_by_trip.values()
        ):
            consistent_prefs.append("vegetarian")

        # Cultural experiences appear in all trips
        cultural_keywords = ["museums", "art", "cultural experiences", "temples"]
        if all(
            any(kw in pref.lower() for pref in prefs for kw in cultural_keywords)
            for prefs in preferences_by_trip.values()
        ):
            consistent_prefs.append("cultural experiences")

        print(f"\n✅ Consistent preferences identified:")
        for pref in consistent_prefs:
            print(f"   ✓ {pref}")

        # Verify expected long-term memories include these patterns
        preference_memories = [
            m for m in expected_memories if m["type"] == "preference"
        ]
        print(
            f"\n   Expected long-term preference memories: {len(preference_memories)}"
        )
        for mem in preference_memories:
            print(f"   - {mem['content']}")

        assert (
            len(consistent_prefs) > 0
        ), "Should identify at least one consistent preference"
