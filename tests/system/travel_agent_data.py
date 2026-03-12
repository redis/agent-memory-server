"""
Travel Agent Test Data Generator

Provides realistic conversation data for system tests based on travel planning scenarios.
"""

import json
from pathlib import Path
from typing import Any

from agent_memory_client.models import MemoryMessage


class TravelAgentDataGenerator:
    """Generate realistic travel agent conversation data for testing."""

    def __init__(self):
        """Load test data from JSON file."""
        data_file = Path(__file__).parent / "test_data_travel_agent.json"
        with open(data_file) as f:
            self.data = json.load(f)

    def get_short_conversation(self) -> list[MemoryMessage]:
        """Get a short conversation (10 messages) - weekend trip inquiry."""
        messages = self.data["short_conversation"]["messages"]
        return [MemoryMessage(**msg) for msg in messages]

    def get_medium_conversation(self, num_messages: int = 50) -> list[MemoryMessage]:
        """
        Generate a medium conversation (50 messages) - family vacation planning.

        Uses the conversation flow to create realistic back-and-forth dialogue.
        """
        flow = self.data["medium_conversation"]["conversation_flow"]
        sample_msgs = self.data["medium_conversation"]["sample_messages"]

        messages = []

        # Start with sample messages
        for msg in sample_msgs:
            messages.append(MemoryMessage(**msg))

        # Generate additional messages based on conversation flow
        current_idx = len(messages)
        for i, topic in enumerate(flow[current_idx:], start=current_idx):
            if len(messages) >= num_messages:
                break

            # Alternate between user and assistant
            role = "user" if i % 2 == 0 else "assistant"

            # Create contextual message based on topic
            if role == "user":
                content = (
                    f"I have a question about {topic.lower()}. What do you recommend?"
                )
            else:
                content = f"Great question! For {topic.lower()}, I suggest we consider several options. Let me provide some recommendations based on your family's needs."

            messages.append(MemoryMessage(role=role, content=content))

        return messages[:num_messages]

    def get_long_conversation(self, num_messages: int = 200) -> list[MemoryMessage]:
        """
        Generate a long conversation (200 messages) - complex honeymoon planning.

        Simulates a multi-phase planning process with increasing detail.
        """
        flow = self.data["long_conversation"]["conversation_flow"]
        sample_msgs = self.data["long_conversation"]["sample_messages"]

        messages = []

        # Start with sample messages
        for msg in sample_msgs:
            messages.append(MemoryMessage(**msg))

        # Generate messages for each phase
        phases = [
            ("Initial Planning", 40),
            ("Destination Selection", 40),
            ("Detailed Planning", 60),
            ("Refinements", 40),
            ("Final Details", 20),
        ]

        current_count = len(messages)

        for phase_name, target_count in phases:
            phase_messages = target_count - (len(messages) - current_count)

            for i in range(phase_messages):
                if len(messages) >= num_messages:
                    break

                role = "user" if i % 2 == 0 else "assistant"

                # Create phase-appropriate content
                if phase_name == "Initial Planning":
                    topics = [
                        "budget",
                        "dates",
                        "destinations",
                        "interests",
                        "accommodation style",
                    ]
                elif phase_name == "Destination Selection":
                    topics = [
                        "London activities",
                        "Paris hotels",
                        "French Riviera beaches",
                        "Tuscany wineries",
                        "Amalfi coast",
                    ]
                elif phase_name == "Detailed Planning":
                    topics = [
                        "flight bookings",
                        "hotel confirmations",
                        "restaurant reservations",
                        "wine tours",
                        "private transfers",
                    ]
                elif phase_name == "Refinements":
                    topics = [
                        "dietary preferences",
                        "special occasions",
                        "photography",
                        "surprises",
                        "upgrades",
                    ]
                else:  # Final Details
                    topics = [
                        "travel insurance",
                        "packing",
                        "currency",
                        "emergency contacts",
                        "final checklist",
                    ]

                topic = topics[i % len(topics)]

                if role == "user":
                    content = f"[{phase_name}] Can we discuss {topic}? I want to make sure we get this right."
                else:
                    content = f"[{phase_name}] Absolutely! For {topic}, here's what I recommend based on your honeymoon plans..."

                messages.append(MemoryMessage(role=role, content=content))

            current_count = len(messages)

        return messages[:num_messages]

    def get_very_large_message(self, size_chars: int = 5000) -> MemoryMessage:
        """
        Get a very large message (detailed itinerary or preferences).

        Uses realistic travel planning content.
        """
        examples = self.data["very_large_messages"]["examples"]
        base_content = examples[0]["content"]

        # Repeat and expand content to reach desired size
        content = base_content
        while len(content) < size_chars:
            content += "\n\n" + base_content

        return MemoryMessage(role="assistant", content=content[:size_chars])

    def get_concurrent_update_messages(self) -> list[tuple[str, MemoryMessage]]:
        """
        Get messages for concurrent update testing.

        Returns list of (agent_name, message) tuples representing different
        agents updating the same conversation.
        """
        updates = self.data["concurrent_updates_scenario"]["updates"]

        result = []
        for update in updates:
            agent = update["agent"]
            message = MemoryMessage(
                role="assistant", content=f"[{agent}] {update['message']}"
            )
            result.append((agent, message))

        return result

    def get_summarization_test_data(self) -> dict[str, Any]:
        """
        Get data specifically designed to test summarization.

        Returns dict with:
        - initial_context: Starting context
        - messages_to_summarize: Messages that should be summarized
        - recent_messages: Messages that should be preserved
        - expected_summary_keywords: Keywords expected in summary
        """
        data = self.data["summarization_test_data"]

        # Generate messages from topics
        messages_to_summarize = []
        for i, topic in enumerate(data["messages_to_summarize"]):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Discussing {topic} for the Greece trip."
            messages_to_summarize.append(MemoryMessage(role=role, content=content))

        # Recent messages to preserve
        recent_messages = [
            MemoryMessage(**msg) for msg in data["recent_messages_to_preserve"]
        ]

        return {
            "initial_context": data["initial_context"],
            "messages_to_summarize": messages_to_summarize,
            "recent_messages": recent_messages,
            "expected_keywords": [
                "anniversary",
                "Greece",
                "islands",
                "Santorini",
                "budget",
                "hotels",
            ],
        }

    def get_returning_client_trips(self) -> list[dict[str, Any]]:
        """
        Get data for returning client scenario (multiple trips over time).

        Returns list of trip dictionaries, each containing:
        - trip_number: Sequential trip number
        - session_id: Unique session ID for this trip
        - date: When the trip was planned
        - description: Trip description
        - key_details: Important details about the trip
        - sample_messages: Conversation messages
        - references_to_previous_trips: How this trip references earlier ones
        """
        return self.data["returning_client_scenario"]["trips"]

    def get_trip_messages(self, trip_number: int) -> list[MemoryMessage]:
        """
        Get conversation messages for a specific trip.

        Args:
            trip_number: Which trip (1, 2, or 3)

        Returns:
            List of MemoryMessage objects for that trip
        """
        trips = self.get_returning_client_trips()
        trip = next((t for t in trips if t["trip_number"] == trip_number), None)

        if not trip:
            raise ValueError(f"Trip {trip_number} not found")

        return [MemoryMessage(**msg) for msg in trip["sample_messages"]]

    def get_expected_long_term_memories(self) -> list[dict[str, str]]:
        """
        Get expected long-term memories that should be extracted from multiple trips.

        Returns list of dicts with:
        - type: Type of memory (preference, history, relationship, pattern)
        - content: The memory content
        """
        return self.data["returning_client_scenario"]["expected_long_term_memories"]

    def get_returning_client_test_scenarios(self) -> list[dict[str, Any]]:
        """
        Get test scenarios for returning client behavior.

        Returns scenarios that test:
        - Fourth trip planning (referencing all previous trips)
        - Context switching mid-conversation
        """
        return self.data["returning_client_scenario"]["test_scenarios"]
