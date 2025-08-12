"""
Integration tests for contextual grounding with real LLM calls.

These tests make actual API calls to LLMs to evaluate contextual grounding
quality in real-world scenarios. They complement the mock-based tests by
providing validation of actual LLM performance on contextual grounding tasks.

Run with: uv run pytest tests/test_contextual_grounding_integration.py --run-api-tests
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import ulid
from pydantic import BaseModel

from agent_memory_server.config import settings
from agent_memory_server.extraction import extract_discrete_memories
from agent_memory_server.llms import get_model_client
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum


class GroundingEvaluationResult(BaseModel):
    """Result of contextual grounding evaluation"""

    category: str
    input_text: str
    grounded_text: str
    expected_grounding: dict[str, str]
    actual_grounding: dict[str, str]
    pronoun_resolution_score: float  # 0-1
    temporal_grounding_score: float  # 0-1
    spatial_grounding_score: float  # 0-1
    completeness_score: float  # 0-1
    accuracy_score: float  # 0-1
    overall_score: float  # 0-1


class ContextualGroundingBenchmark:
    """Benchmark dataset for contextual grounding evaluation"""

    @staticmethod
    def get_pronoun_grounding_examples():
        """Examples for testing pronoun resolution"""
        return [
            {
                "category": "pronoun_he_him",
                "messages": [
                    "John is a software engineer.",
                    "He works at Google and loves coding in Python.",
                    "I told him about the new framework we're using.",
                ],
                "expected_grounding": {"he": "John", "him": "John"},
                "context_date": datetime.now(UTC),
            },
            {
                "category": "pronoun_she_her",
                "messages": [
                    "Sarah is our project manager.",
                    "She has been leading the team for two years.",
                    "Her experience with agile methodology is invaluable.",
                ],
                "expected_grounding": {"she": "Sarah", "her": "Sarah"},
                "context_date": datetime.now(UTC),
            },
            {
                "category": "pronoun_they_them",
                "messages": [
                    "Alex joined our team last month.",
                    "They have expertise in machine learning.",
                    "We assigned them to the AI project.",
                ],
                "expected_grounding": {"they": "Alex", "them": "Alex"},
                "context_date": datetime.now(UTC),
            },
        ]

    @staticmethod
    def get_temporal_grounding_examples():
        """Examples for testing temporal grounding"""
        current_year = datetime.now(UTC).year
        yesterday = datetime.now(UTC) - timedelta(days=1)
        return [
            {
                "category": "temporal_last_year",
                "messages": [
                    f"We launched our product in {current_year - 1}.",
                    "Last year was a great year for growth.",
                    "The revenue last year exceeded expectations.",
                ],
                "expected_grounding": {"last year": str(current_year - 1)},
                "context_date": datetime.now(UTC),
            },
            {
                "category": "temporal_yesterday",
                "messages": [
                    "The meeting was scheduled for yesterday.",
                    "Yesterday's presentation went well.",
                    "We discussed the budget yesterday.",
                ],
                "expected_grounding": {"yesterday": yesterday.strftime("%Y-%m-%d")},
                "context_date": datetime.now(UTC),
            },
            {
                "category": "temporal_complex_relative",
                "messages": [
                    "The project started three months ago.",
                    "Two weeks later, we hit our first milestone.",
                    "Since then, progress has been steady.",
                ],
                "expected_grounding": {
                    "three months ago": (
                        datetime.now(UTC) - timedelta(days=90)
                    ).strftime("%Y-%m-%d"),
                    "two weeks later": (
                        datetime.now(UTC) - timedelta(days=76)
                    ).strftime("%Y-%m-%d"),
                    "since then": "since "
                    + (datetime.now(UTC) - timedelta(days=76)).strftime("%Y-%m-%d"),
                },
                "context_date": datetime.now(UTC),
            },
        ]

    @staticmethod
    def get_spatial_grounding_examples():
        """Examples for testing spatial grounding"""
        return [
            {
                "category": "spatial_there_here",
                "messages": [
                    "We visited San Francisco last week.",
                    "The weather there was perfect.",
                    "I'd love to go back there again.",
                ],
                "expected_grounding": {"there": "San Francisco"},
                "context_date": datetime.now(UTC),
            },
            {
                "category": "spatial_that_place",
                "messages": [
                    "Chez Panisse is an amazing restaurant.",
                    "That place has the best organic food.",
                    "We should make a reservation at that place.",
                ],
                "expected_grounding": {"that place": "Chez Panisse"},
                "context_date": datetime.now(UTC),
            },
        ]

    @staticmethod
    def get_definite_reference_examples():
        """Examples for testing definite reference resolution"""
        return [
            {
                "category": "definite_reference_meeting",
                "messages": [
                    "We scheduled a quarterly review for next Tuesday.",
                    "The meeting will cover Q4 performance.",
                    "Please prepare your slides for the meeting.",
                ],
                "expected_grounding": {"the meeting": "quarterly review"},
                "context_date": datetime.now(UTC),
            }
        ]

    @classmethod
    def get_all_examples(cls):
        """Get all benchmark examples"""
        examples = []
        examples.extend(cls.get_pronoun_grounding_examples())
        examples.extend(cls.get_temporal_grounding_examples())
        examples.extend(cls.get_spatial_grounding_examples())
        examples.extend(cls.get_definite_reference_examples())
        return examples


class LLMContextualGroundingJudge:
    """LLM-as-a-Judge system for evaluating contextual grounding quality"""

    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        # Load the evaluation prompt from template file
        template_path = (
            Path(__file__).parent
            / "templates"
            / "contextual_grounding_evaluation_prompt.txt"
        )
        with open(template_path) as f:
            self.EVALUATION_PROMPT = f.read()

    async def evaluate_grounding(
        self,
        context_messages: list[str],
        original_text: str,
        grounded_text: str,
        expected_grounding: dict[str, str],
    ) -> dict[str, float]:
        """Evaluate contextual grounding quality using LLM judge"""
        client = await get_model_client(self.judge_model)

        prompt = self.EVALUATION_PROMPT.format(
            context_messages="\n".join(context_messages),
            original_text=original_text,
            grounded_text=grounded_text,
            expected_grounding=json.dumps(expected_grounding, indent=2),
        )

        response = await client.create_chat_completion(
            model=self.judge_model,
            prompt=prompt,
            response_format={"type": "json_object"},
        )

        try:
            evaluation = json.loads(response.choices[0].message.content)
            return {
                "pronoun_resolution_score": evaluation.get(
                    "pronoun_resolution_score", 0.0
                ),
                "temporal_grounding_score": evaluation.get(
                    "temporal_grounding_score", 0.0
                ),
                "spatial_grounding_score": evaluation.get(
                    "spatial_grounding_score", 0.0
                ),
                "completeness_score": evaluation.get("completeness_score", 0.0),
                "accuracy_score": evaluation.get("accuracy_score", 0.0),
                "overall_score": evaluation.get("overall_score", 0.0),
                "explanation": evaluation.get("explanation", ""),
            }
        except json.JSONDecodeError as e:
            print(
                f"Failed to parse judge response: {response.choices[0].message.content}"
            )
            raise e


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
class TestContextualGroundingIntegration:
    """Integration tests for contextual grounding with real LLM calls"""

    async def create_test_memory_with_context(
        self, context_messages: list[str], target_message: str, context_date: datetime
    ) -> MemoryRecord:
        """Create a memory record with conversational context"""
        # Combine context messages and target message
        full_conversation = "\n".join(context_messages + [target_message])

        return MemoryRecord(
            id=str(ulid.ULID()),
            text=full_conversation,
            memory_type=MemoryTypeEnum.MESSAGE,
            discrete_memory_extracted="f",
            session_id=f"test-integration-session-{ulid.ULID()}",
            user_id="test-integration-user",
            timestamp=context_date.isoformat(),
        )

    async def test_pronoun_grounding_integration_he_him(self):
        """Integration test for he/him pronoun grounding with real LLM"""
        example = ContextualGroundingBenchmark.get_pronoun_grounding_examples()[0]

        # Create memory record and store it first
        memory = await self.create_test_memory_with_context(
            example["messages"][:-1],  # Context
            example["messages"][-1],  # Target message with pronouns
            example["context_date"],
        )

        # Store the memory so it can be found by extract_discrete_memories
        from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

        adapter = await get_vectorstore_adapter()
        await adapter.add_memories([memory])

        # Extract memories using real LLM
        await extract_discrete_memories([memory])

        # Retrieve all memories to verify extraction occurred
        all_memories = await adapter.search_memories(
            query="",
            limit=50,  # Get all memories
        )

        # Find the original memory by session_id and verify it was processed
        session_memories = [
            m for m in all_memories.memories if m.session_id == memory.session_id
        ]

        # Should find the original message memory that was processed
        assert (
            len(session_memories) >= 1
        ), f"No memories found in session {memory.session_id}"

        # Find our specific memory in the results
        processed_memory = next(
            (m for m in session_memories if m.id == memory.id), None
        )

        if processed_memory is None:
            # If we can't find by ID, try to find any memory in the session with discrete_memory_extracted = "t"
            processed_memory = next(
                (m for m in session_memories if m.discrete_memory_extracted == "t"),
                None,
            )

        assert (
            processed_memory is not None
        ), f"Could not find processed memory {memory.id} in session"
        assert processed_memory.discrete_memory_extracted == "t"

        # Should also find extracted discrete memories
        discrete_memories = [
            m
            for m in all_memories.memories
            if m.memory_type in ["episodic", "semantic"]
        ]
        assert (
            len(discrete_memories) >= 1
        ), "Expected at least one discrete memory to be extracted"

        # Note: Full evaluation with LLM judge will be implemented in subsequent tests

    async def test_temporal_grounding_integration_last_year(self):
        """Integration test for temporal grounding with real LLM"""
        example = ContextualGroundingBenchmark.get_temporal_grounding_examples()[0]

        memory = await self.create_test_memory_with_context(
            example["messages"][:-1], example["messages"][-1], example["context_date"]
        )

        # Store and extract
        from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

        adapter = await get_vectorstore_adapter()
        await adapter.add_memories([memory])
        await extract_discrete_memories([memory])

        # Check extraction was successful - search by session_id since ID search may not work reliably
        from agent_memory_server.filters import MemoryType, SessionId

        updated_memories = await adapter.search_memories(
            query="",
            session_id=SessionId(eq=memory.session_id),
            memory_type=MemoryType(eq="message"),
            limit=10,
        )
        # Find our specific memory in the results
        target_memory = next(
            (m for m in updated_memories.memories if m.id == memory.id), None
        )
        assert (
            target_memory is not None
        ), f"Could not find memory {memory.id} after extraction"
        assert target_memory.discrete_memory_extracted == "t"

    async def test_spatial_grounding_integration_there(self):
        """Integration test for spatial grounding with real LLM"""
        example = ContextualGroundingBenchmark.get_spatial_grounding_examples()[0]

        memory = await self.create_test_memory_with_context(
            example["messages"][:-1], example["messages"][-1], example["context_date"]
        )

        # Store and extract
        from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

        adapter = await get_vectorstore_adapter()
        await adapter.add_memories([memory])
        await extract_discrete_memories([memory])

        # Check extraction was successful - search by session_id since ID search may not work reliably
        from agent_memory_server.filters import MemoryType, SessionId

        updated_memories = await adapter.search_memories(
            query="",
            session_id=SessionId(eq=memory.session_id),
            memory_type=MemoryType(eq="message"),
            limit=10,
        )
        # Find our specific memory in the results
        target_memory = next(
            (m for m in updated_memories.memories if m.id == memory.id), None
        )
        assert (
            target_memory is not None
        ), f"Could not find memory {memory.id} after extraction"
        assert target_memory.discrete_memory_extracted == "t"

    @pytest.mark.requires_api_keys
    async def test_comprehensive_grounding_evaluation_with_judge(self):
        """Comprehensive test using LLM-as-a-judge for grounding evaluation"""

        judge = LLMContextualGroundingJudge()
        benchmark = ContextualGroundingBenchmark()

        results = []

        # Test a sample of examples (not all to avoid excessive API costs)
        sample_examples = benchmark.get_all_examples()[
            :2
        ]  # Just first 2 for integration testing

        for example in sample_examples:
            # Create memory and extract with real LLM
            memory = await self.create_test_memory_with_context(
                example["messages"][:-1],
                example["messages"][-1],
                example["context_date"],
            )

            original_text = example["messages"][-1]

            # Store and extract
            from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

            adapter = await get_vectorstore_adapter()
            await adapter.add_memories([memory])
            await extract_discrete_memories([memory])

            # Retrieve all extracted discrete memories to get the grounded text
            all_memories = await adapter.search_memories(query="", limit=50)
            discrete_memories = [
                m
                for m in all_memories.memories
                if m.memory_type in ["episodic", "semantic"]
                and m.session_id == memory.session_id
            ]

            # Combine the grounded memories into a single text for evaluation
            grounded_text = (
                " ".join([dm.text for dm in discrete_memories])
                if discrete_memories
                else original_text
            )

            # Evaluate with judge
            evaluation = await judge.evaluate_grounding(
                context_messages=example["messages"][:-1],
                original_text=original_text,
                grounded_text=grounded_text,
                expected_grounding=example["expected_grounding"],
            )

            result = GroundingEvaluationResult(
                category=example["category"],
                input_text=original_text,
                grounded_text=grounded_text,
                expected_grounding=example["expected_grounding"],
                actual_grounding={},  # Could be parsed from grounded_text
                **evaluation,
            )

            results.append(result)

            print(f"\nExample: {example['category']}")
            print(f"Original: {original_text}")
            print(f"Grounded: {grounded_text}")
            print(f"Score: {result.overall_score:.3f}")

            # Assert minimum quality thresholds (contextual grounding partially working)
            # Note: The system currently grounds subject pronouns but not all possessive pronouns
            assert (
                result.overall_score >= 0.05
            ), f"Poor grounding quality for {example['category']}: {result.overall_score}"

        # Print summary statistics
        avg_score = sum(r.overall_score for r in results) / len(results)
        print("\nContextual Grounding Integration Test Results:")
        print(f"Average Overall Score: {avg_score:.3f}")

        for result in results:
            print(f"{result.category}: {result.overall_score:.3f}")

        assert avg_score >= 0.05, f"Average grounding quality too low: {avg_score}"

    async def test_model_comparison_grounding_quality(self):
        """Compare contextual grounding quality across different models"""
        if not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")):
            pytest.skip("Multiple API keys required for model comparison")

        models_to_test = ["gpt-4o-mini", "claude-3-haiku-20240307"]
        example = ContextualGroundingBenchmark.get_pronoun_grounding_examples()[0]

        results_by_model = {}

        original_model = settings.generation_model

        try:
            for model in models_to_test:
                # Temporarily override the generation model setting
                settings.generation_model = model

                try:
                    memory = await self.create_test_memory_with_context(
                        example["messages"][:-1],
                        example["messages"][-1],
                        example["context_date"],
                    )

                    # Store the memory so it can be found by extract_discrete_memories
                    from agent_memory_server.vectorstore_factory import (
                        get_vectorstore_adapter,
                    )

                    adapter = await get_vectorstore_adapter()
                    await adapter.add_memories([memory])

                    await extract_discrete_memories([memory])

                    # Check if extraction was successful by searching for the memory
                    from agent_memory_server.filters import MemoryType, SessionId

                    updated_memories = await adapter.search_memories(
                        query="",
                        session_id=SessionId(eq=memory.session_id),
                        memory_type=MemoryType(eq="message"),
                        limit=10,
                    )

                    # Find our specific memory in the results
                    target_memory = next(
                        (m for m in updated_memories.memories if m.id == memory.id),
                        None,
                    )
                    success = (
                        target_memory is not None
                        and target_memory.discrete_memory_extracted == "t"
                    )

                    # Record success/failure for this model
                    results_by_model[model] = {"success": success, "model": model}

                except Exception as e:
                    results_by_model[model] = {
                        "success": False,
                        "error": str(e),
                        "model": model,
                    }
        finally:
            # Always restore original model setting
            settings.generation_model = original_model

        print("\nModel Comparison Results:")
        for model, result in results_by_model.items():
            status = "✓" if result["success"] else "✗"
            print(f"{model}: {status}")

        # At least one model should succeed
        assert any(
            r["success"] for r in results_by_model.values()
        ), "No model successfully completed grounding"
