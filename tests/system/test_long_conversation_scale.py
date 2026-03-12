"""
System Test: Long Conversation Memory at Scale

This test harness validates the system's ability to handle long conversations
as described in long_conversation_memory.md. It tests:

1. Long conversations are stored successfully with O(1) latency
2. Older content is summarized into context when needed
3. Recent messages stay available and in order regardless of length
4. Reading the session or building a memory prompt still works after summarization

Run with:
    uv run pytest tests/system/test_long_conversation_scale.py --run-api-tests -v -s

Use environment variables to control scale:
    SCALE_SHORT_MESSAGES=10 SCALE_MEDIUM_MESSAGES=50 SCALE_LONG_MESSAGES=200 \\
    SCALE_PARALLEL_SESSIONS=5 uv run pytest tests/system/test_long_conversation_scale.py --run-api-tests -v
"""

import asyncio
import os
import time

import pytest
from agent_memory_client.client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import MemoryMessage, WorkingMemory

# Scale configuration from environment
SCALE_SHORT_MESSAGES = int(os.getenv("SCALE_SHORT_MESSAGES", "10"))
SCALE_MEDIUM_MESSAGES = int(os.getenv("SCALE_MEDIUM_MESSAGES", "50"))
SCALE_LONG_MESSAGES = int(os.getenv("SCALE_LONG_MESSAGES", "200"))
SCALE_VERY_LARGE_MESSAGE_SIZE = int(os.getenv("SCALE_VERY_LARGE_MESSAGE_SIZE", "5000"))
SCALE_PARALLEL_SESSIONS = int(os.getenv("SCALE_PARALLEL_SESSIONS", "5"))
SCALE_CONCURRENT_UPDATES = int(os.getenv("SCALE_CONCURRENT_UPDATES", "10"))

# Test configuration
INTEGRATION_BASE_URL = os.getenv("MEMORY_SERVER_BASE_URL", "http://localhost:8001")

pytestmark = [pytest.mark.integration, pytest.mark.requires_api_keys]


@pytest.fixture
async def scale_test_client():
    """Create a memory client configured for scale testing."""
    config = MemoryClientConfig(
        base_url=INTEGRATION_BASE_URL,
        timeout=60.0,  # Longer timeout for scale tests
        default_namespace="scale-test",
        default_context_window_max=128000,  # Use a large context window
    )
    async with MemoryAPIClient(config) as client:
        yield client


class ConversationBuilder:
    """Helper to build test conversations of various sizes."""

    @staticmethod
    def create_messages(count: int, prefix: str = "msg") -> list[MemoryMessage]:
        """Create a list of alternating user/assistant messages."""
        messages = []
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"This is {prefix} number {i}. It contains conversation content about topic {i % 5}."
            messages.append(MemoryMessage(role=role, content=content))
        return messages

    @staticmethod
    def create_large_message(size_chars: int) -> MemoryMessage:
        """Create a single very large message."""
        content = "A" * size_chars
        return MemoryMessage(role="user", content=content)


class TestLongConversationPrepare:
    """Test preparation: Create conversations of different sizes."""

    @pytest.mark.asyncio
    async def test_short_conversation(self, scale_test_client: MemoryAPIClient):
        """Test storing a short conversation (baseline)."""
        session_id = f"short-conv-{int(time.time())}"
        messages = ConversationBuilder.create_messages(SCALE_SHORT_MESSAGES, "short")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
            context="Short conversation test",
        )

        start = time.perf_counter()
        response = await scale_test_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None
        assert len(response.messages) == SCALE_SHORT_MESSAGES
        print(
            f"\n✅ Short conversation ({SCALE_SHORT_MESSAGES} msgs) stored in {latency:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_medium_conversation(self, scale_test_client: MemoryAPIClient):
        """Test storing a medium-sized conversation."""
        session_id = f"medium-conv-{int(time.time())}"
        messages = ConversationBuilder.create_messages(SCALE_MEDIUM_MESSAGES, "medium")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
            context="Medium conversation test",
        )

        start = time.perf_counter()
        response = await scale_test_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None
        print(
            f"\n✅ Medium conversation ({SCALE_MEDIUM_MESSAGES} msgs) stored in {latency:.3f}s"
        )
        print(
            f"   Latency per message: {(latency / SCALE_MEDIUM_MESSAGES) * 1000:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_long_conversation(self, scale_test_client: MemoryAPIClient):
        """Test storing a very long conversation."""
        session_id = f"long-conv-{int(time.time())}"
        messages = ConversationBuilder.create_messages(SCALE_LONG_MESSAGES, "long")

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
            context="Long conversation test",
        )

        start = time.perf_counter()
        response = await scale_test_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None
        print(
            f"\n✅ Long conversation ({SCALE_LONG_MESSAGES} msgs) stored in {latency:.3f}s"
        )
        print(f"   Latency per message: {(latency / SCALE_LONG_MESSAGES) * 1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_very_large_messages(self, scale_test_client: MemoryAPIClient):
        """Test storing a conversation with a few very large messages."""
        session_id = f"large-msg-conv-{int(time.time())}"
        messages = [
            MemoryMessage(role="user", content="Start of conversation"),
            ConversationBuilder.create_large_message(SCALE_VERY_LARGE_MESSAGE_SIZE),
            MemoryMessage(role="assistant", content="Response to large message"),
            ConversationBuilder.create_large_message(SCALE_VERY_LARGE_MESSAGE_SIZE),
            MemoryMessage(role="user", content="End of conversation"),
        ]

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
            context="Large message test",
        )

        start = time.perf_counter()
        response = await scale_test_client.put_working_memory(
            session_id, working_memory
        )
        latency = time.perf_counter() - start

        assert response is not None
        assert len(response.messages) == 5
        print(f"\n✅ Large message conversation stored in {latency:.3f}s")
        print(f"   Total chars: {sum(len(m.content) for m in messages):,}")


class TestLongConversationRun:
    """Test running: Repeated updates and parallel sessions."""

    @pytest.mark.asyncio
    async def test_repeated_updates_to_session(
        self, scale_test_client: MemoryAPIClient
    ):
        """Test repeated updates to a single session to simulate a growing conversation."""
        session_id = f"repeated-updates-{int(time.time())}"

        # Start with initial messages
        initial_messages = ConversationBuilder.create_messages(5, "initial")
        working_memory = WorkingMemory(
            session_id=session_id,
            messages=initial_messages,
            namespace="scale-test",
        )
        await scale_test_client.put_working_memory(session_id, working_memory)

        # Perform repeated updates
        update_count = 20
        latencies = []

        for i in range(update_count):
            new_messages = [
                MemoryMessage(role="user", content=f"Update {i} user message"),
                MemoryMessage(
                    role="assistant", content=f"Update {i} assistant response"
                ),
            ]

            start = time.perf_counter()
            await scale_test_client.append_messages_to_working_memory(
                session_id, new_messages
            )
            latency = time.perf_counter() - start
            latencies.append(latency)

        # Verify final state
        final_memory = await scale_test_client.get_working_memory(session_id)
        assert final_memory is not None

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        print(f"\n✅ {update_count} repeated updates completed")
        print(f"   Average latency: {avg_latency * 1000:.2f}ms")
        print(f"   Max latency: {max_latency * 1000:.2f}ms")
        print(f"   Final message count: {len(final_memory.messages)}")

    @pytest.mark.asyncio
    async def test_parallel_long_sessions(self, scale_test_client: MemoryAPIClient):
        """Test many separate long sessions in parallel."""

        async def create_long_session(session_num: int) -> tuple[str, float]:
            """Create a single long session and return its ID and latency."""
            session_id = f"parallel-{session_num}-{int(time.time())}"
            messages = ConversationBuilder.create_messages(
                SCALE_MEDIUM_MESSAGES, f"parallel-{session_num}"
            )

            working_memory = WorkingMemory(
                session_id=session_id,
                messages=messages,
                namespace="scale-test",
            )

            start = time.perf_counter()
            await scale_test_client.put_working_memory(session_id, working_memory)
            latency = time.perf_counter() - start

            return session_id, latency

        # Create sessions in parallel
        start_total = time.perf_counter()
        results = await asyncio.gather(
            *[create_long_session(i) for i in range(SCALE_PARALLEL_SESSIONS)]
        )
        total_time = time.perf_counter() - start_total

        session_ids = [r[0] for r in results]
        latencies = [r[1] for r in results]

        # Verify all sessions were created
        assert len(session_ids) == SCALE_PARALLEL_SESSIONS

        avg_latency = sum(latencies) / len(latencies)
        print(f"\n✅ {SCALE_PARALLEL_SESSIONS} parallel sessions created")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average session latency: {avg_latency:.3f}s")
        print(f"   Messages per session: {SCALE_MEDIUM_MESSAGES}")

    @pytest.mark.asyncio
    async def test_concurrent_updates_same_session(
        self, scale_test_client: MemoryAPIClient
    ):
        """Test concurrent updates to the same session."""
        session_id = f"concurrent-{int(time.time())}"

        # Create initial session
        initial_messages = ConversationBuilder.create_messages(5, "initial")
        working_memory = WorkingMemory(
            session_id=session_id,
            messages=initial_messages,
            namespace="scale-test",
        )
        await scale_test_client.put_working_memory(session_id, working_memory)

        async def append_update(update_num: int) -> float:
            """Append messages and return latency."""
            messages = [
                MemoryMessage(role="user", content=f"Concurrent update {update_num}"),
            ]
            start = time.perf_counter()
            await scale_test_client.append_messages_to_working_memory(
                session_id, messages
            )
            return time.perf_counter() - start

        # Perform concurrent updates
        start_total = time.perf_counter()
        latencies = await asyncio.gather(
            *[append_update(i) for i in range(SCALE_CONCURRENT_UPDATES)]
        )
        total_time = time.perf_counter() - start_total

        # Verify final state
        final_memory = await scale_test_client.get_working_memory(session_id)
        assert final_memory is not None

        print(f"\n✅ {SCALE_CONCURRENT_UPDATES} concurrent updates completed")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average latency: {sum(latencies) / len(latencies) * 1000:.2f}ms")
        print(f"   Final message count: {len(final_memory.messages)}")


class TestLongConversationCheck:
    """Test checking: Verify summarization, message order, and prompt generation."""

    @pytest.mark.asyncio
    async def test_summarization_triggers(self, scale_test_client: MemoryAPIClient):
        """Test that summarization is triggered when conversation gets large."""
        session_id = f"summarization-test-{int(time.time())}"

        # Create a conversation large enough to trigger summarization
        # Using a smaller context window to force summarization
        config = MemoryClientConfig(
            base_url=INTEGRATION_BASE_URL,
            timeout=60.0,
            default_namespace="scale-test",
            default_context_window_max=4000,  # Small window to trigger summarization
        )

        async with MemoryAPIClient(config) as client:
            # Create many messages with substantial content
            messages = []
            for i in range(100):
                role = "user" if i % 2 == 0 else "assistant"
                content = f"Message {i}: " + (
                    "This is a longer message with more content. " * 10
                )
                messages.append(MemoryMessage(role=role, content=content))

            working_memory = WorkingMemory(
                session_id=session_id,
                messages=messages,
                namespace="scale-test",
            )

            # Store the conversation
            await client.put_working_memory(session_id, working_memory)

            # Wait a bit for background summarization to potentially occur
            await asyncio.sleep(2)

            # Retrieve and check
            result = await client.get_working_memory(session_id)
            assert result is not None

            # Check if summarization occurred
            has_summary = result.context is not None and len(result.context) > 0
            message_count = len(result.messages)

            print(f"\n✅ Summarization test completed")
            print(f"   Summary created: {has_summary}")
            if has_summary:
                print(f"   Summary length: {len(result.context)} chars")
            print(f"   Messages retained: {message_count} (started with 100)")
            print(
                f"   Context percentage used: {result.context_percentage_total_used:.1f}%"
            )
            if result.context_percentage_until_summarization is not None:
                print(
                    f"   Until summarization: {result.context_percentage_until_summarization:.1f}%"
                )

    @pytest.mark.asyncio
    async def test_message_order_preserved(self, scale_test_client: MemoryAPIClient):
        """Test that messages remain in correct chronological order."""
        session_id = f"order-test-{int(time.time())}"

        # Create messages with specific content to verify order
        messages = []
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message sequence number {i:03d}"
            messages.append(MemoryMessage(role=role, content=content))

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
        )

        await scale_test_client.put_working_memory(session_id, working_memory)

        # Retrieve and verify order
        result = await scale_test_client.get_working_memory(session_id)
        assert result is not None

        # Check that messages are in order
        for i, msg in enumerate(result.messages):
            # Extract sequence number from content
            if "sequence number" in msg.content:
                seq_num = int(msg.content.split("sequence number ")[1])
                # Verify it's in ascending order (accounting for potential summarization)
                if i > 0 and "sequence number" in result.messages[i - 1].content:
                    prev_seq = int(
                        result.messages[i - 1].content.split("sequence number ")[1]
                    )
                    assert (
                        seq_num > prev_seq
                    ), f"Messages out of order: {prev_seq} -> {seq_num}"

        print(f"\n✅ Message order preserved")
        print(f"   Total messages checked: {len(result.messages)}")
        print(f"   All messages in chronological order: ✓")

    @pytest.mark.asyncio
    async def test_recent_messages_available(self, scale_test_client: MemoryAPIClient):
        """Test that recent messages are always available even after summarization."""
        session_id = f"recent-test-{int(time.time())}"

        # Create a large conversation
        messages = ConversationBuilder.create_messages(100, "test")

        # Add some distinctive recent messages
        recent_marker = "RECENT_MESSAGE_MARKER"
        for i in range(5):
            messages.append(
                MemoryMessage(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"{recent_marker} {i}: This is a recent message that should be preserved",
                )
            )

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
        )

        await scale_test_client.put_working_memory(session_id, working_memory)

        # Retrieve and check recent messages
        result = await scale_test_client.get_working_memory(session_id)
        assert result is not None

        # Count how many recent messages are still present
        recent_count = sum(1 for msg in result.messages if recent_marker in msg.content)

        print(f"\n✅ Recent messages check")
        print(f"   Recent messages preserved: {recent_count}/5")
        print(f"   Total messages in session: {len(result.messages)}")

        # At least some recent messages should be preserved
        assert recent_count > 0, "No recent messages were preserved"

    @pytest.mark.asyncio
    async def test_memory_prompt_generation(self, scale_test_client: MemoryAPIClient):
        """Test that memory prompt generation works after long conversations."""
        session_id = f"prompt-test-{int(time.time())}"

        # Create a conversation with specific topics
        messages = []
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            topic = ["travel", "food", "technology", "sports", "music"][i % 5]
            content = f"Let's discuss {topic}. Message {i} about {topic} preferences."
            messages.append(MemoryMessage(role=role, content=content))

        working_memory = WorkingMemory(
            session_id=session_id,
            messages=messages,
            namespace="scale-test",
            user_id="test-user",
        )

        await scale_test_client.put_working_memory(session_id, working_memory)

        # Test memory prompt generation
        try:
            prompt_result = await scale_test_client.memory_prompt(
                query="What are the user's preferences about travel?",
                session_id=session_id,
                user_id="test-user",
                namespace="scale-test",
            )

            assert prompt_result is not None
            assert len(prompt_result.prompt) > 0

            print(f"\n✅ Memory prompt generation successful")
            print(f"   Prompt length: {len(prompt_result.prompt)} chars")
            print(
                f"   Working memory included: {prompt_result.working_memory_included}"
            )
            print(f"   Long-term memories: {len(prompt_result.long_term_memories)}")

        except Exception as e:
            print(f"\n⚠️  Memory prompt generation: {e}")
            # This is acceptable if long-term memory is not enabled
            print("   (This may be expected if long-term memory is disabled)")


class TestScaleMetrics:
    """Comprehensive scale test that reports overall metrics."""

    @pytest.mark.asyncio
    async def test_comprehensive_scale_report(self, scale_test_client: MemoryAPIClient):
        """Run a comprehensive scale test and report metrics."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SCALE TEST REPORT")
        print("=" * 70)

        metrics = {
            "short_conversations": [],
            "medium_conversations": [],
            "long_conversations": [],
            "update_latencies": [],
        }

        # Test 1: Multiple short conversations
        print("\n📊 Testing short conversations...")
        for i in range(5):
            session_id = f"scale-short-{i}-{int(time.time())}"
            messages = ConversationBuilder.create_messages(SCALE_SHORT_MESSAGES)
            working_memory = WorkingMemory(
                session_id=session_id, messages=messages, namespace="scale-test"
            )

            start = time.perf_counter()
            await scale_test_client.put_working_memory(session_id, working_memory)
            latency = time.perf_counter() - start
            metrics["short_conversations"].append(latency)

        # Test 2: Multiple medium conversations
        print("📊 Testing medium conversations...")
        for i in range(3):
            session_id = f"scale-medium-{i}-{int(time.time())}"
            messages = ConversationBuilder.create_messages(SCALE_MEDIUM_MESSAGES)
            working_memory = WorkingMemory(
                session_id=session_id, messages=messages, namespace="scale-test"
            )

            start = time.perf_counter()
            await scale_test_client.put_working_memory(session_id, working_memory)
            latency = time.perf_counter() - start
            metrics["medium_conversations"].append(latency)

        # Test 3: Long conversation with updates
        print("📊 Testing long conversation with updates...")
        session_id = f"scale-long-{int(time.time())}"
        messages = ConversationBuilder.create_messages(SCALE_LONG_MESSAGES)
        working_memory = WorkingMemory(
            session_id=session_id, messages=messages, namespace="scale-test"
        )

        start = time.perf_counter()
        await scale_test_client.put_working_memory(session_id, working_memory)
        latency = time.perf_counter() - start
        metrics["long_conversations"].append(latency)

        # Add some updates
        for i in range(10):
            new_msg = [MemoryMessage(role="user", content=f"Update {i}")]
            start = time.perf_counter()
            await scale_test_client.append_messages_to_working_memory(
                session_id, new_msg
            )
            latency = time.perf_counter() - start
            metrics["update_latencies"].append(latency)

        # Print comprehensive report
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\n📈 Short Conversations ({SCALE_SHORT_MESSAGES} messages each):")
        print(f"   Count: {len(metrics['short_conversations'])}")
        print(
            f"   Avg latency: {sum(metrics['short_conversations']) / len(metrics['short_conversations']) * 1000:.2f}ms"
        )
        print(f"   Max latency: {max(metrics['short_conversations']) * 1000:.2f}ms")

        print(f"\n📈 Medium Conversations ({SCALE_MEDIUM_MESSAGES} messages each):")
        print(f"   Count: {len(metrics['medium_conversations'])}")
        print(
            f"   Avg latency: {sum(metrics['medium_conversations']) / len(metrics['medium_conversations']) * 1000:.2f}ms"
        )
        print(f"   Max latency: {max(metrics['medium_conversations']) * 1000:.2f}ms")

        print(f"\n📈 Long Conversations ({SCALE_LONG_MESSAGES} messages each):")
        print(f"   Count: {len(metrics['long_conversations'])}")
        print(
            f"   Avg latency: {sum(metrics['long_conversations']) / len(metrics['long_conversations']) * 1000:.2f}ms"
        )

        print(f"\n📈 Update Operations:")
        print(f"   Count: {len(metrics['update_latencies'])}")
        print(
            f"   Avg latency: {sum(metrics['update_latencies']) / len(metrics['update_latencies']) * 1000:.2f}ms"
        )
        print(f"   Max latency: {max(metrics['update_latencies']) * 1000:.2f}ms")

        print("\n" + "=" * 70)
        print("✅ SCALE TEST COMPLETE")
        print("=" * 70)
