import json
import os
from typing import TYPE_CHECKING, Any

import ulid
from tenacity.asyncio import AsyncRetrying
from tenacity.stop import stop_after_attempt

# Lazy-import transformers in get_ner_model to avoid heavy deps at startup
from agent_memory_server.config import settings
from agent_memory_server.filters import DiscreteMemoryExtracted, MemoryType
from agent_memory_server.llm import LLMClient
from agent_memory_server.logging import get_logger
from agent_memory_server.models import MemoryRecord


if TYPE_CHECKING:
    from bertopic import BERTopic


logger = get_logger(__name__)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model instances
_topic_model: "BERTopic | None" = None
_ner_model: Any | None = None
_ner_tokenizer: Any | None = None


def get_topic_model() -> "BERTopic":
    """
    Get or initialize the BERTopic model.

    Returns:
        The BERTopic model instance
    """
    from bertopic import BERTopic

    global _topic_model
    if _topic_model is None:
        # TODO: Expose this as a config option
        _topic_model = BERTopic.load(
            settings.topic_model, embedding_model="all-MiniLM-L6-v2"
        )
    return _topic_model  # type: ignore


def get_ner_model() -> Any:
    """
    Get or initialize the NER model and tokenizer.

    Returns:
        The NER pipeline instance
    """
    global _ner_model, _ner_tokenizer
    if _ner_model is None:
        # Lazy import to avoid importing heavy ML frameworks at process startup
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline as hf_pipeline,
            )
        except Exception as e:
            logger.warning(
                "Transformers not available or failed to import; NER disabled: %s", e
            )
            raise

        _ner_tokenizer = AutoTokenizer.from_pretrained(settings.ner_model)
        _ner_model = AutoModelForTokenClassification.from_pretrained(settings.ner_model)
        return hf_pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)

    # If already initialized, import the lightweight symbol and return a new pipeline
    from transformers import pipeline as hf_pipeline  # type: ignore

    return hf_pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)


def extract_entities(text: str) -> list[str]:
    """
    Extract named entities from text using the NER model.

    TODO: Cache this output.

    Args:
        text: The text to extract entities from

    Returns:
        List of unique entity names
    """
    try:
        ner = get_ner_model()
        results = ner(text)

        # Group tokens by entity
        current_entity = []
        entities = []

        for result in results:
            if result["word"].startswith("##"):
                # This is a continuation of the previous entity
                current_entity.append(result["word"][2:])
            else:
                # This is a new entity
                if current_entity:
                    entities.append("".join(current_entity))
                current_entity = [result["word"]]

        # Add the last entity if exists
        if current_entity:
            entities.append("".join(current_entity))

        return list(set(entities))  # Remove duplicates

    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return []


async def extract_topics_llm(
    text: str,
    num_topics: int | None = None,
) -> list[str]:
    """
    Extract topics from text using the LLM model.
    """
    _num_topics = num_topics if num_topics is not None else settings.top_k_topics

    prompt = f"""
    Extract the topic {_num_topics} topics from the following text:
    {text}

    Return a list of topics in JSON format, for example:
    {{
        "topics": ["topic1", "topic2", "topic3"]
    }}
    """
    topics = []

    async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
        with attempt:
            response = await LLMClient.create_chat_completion(
                model=settings.generation_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            try:
                topics = json.loads(response.content)["topics"]
            except (json.JSONDecodeError, KeyError):
                logger.error(f"Error decoding JSON: {response.content}")
                topics = []
            if topics:
                topics = topics[:_num_topics]

    return topics


def extract_topics_bertopic(text: str, num_topics: int | None = None) -> list[str]:
    """
    Extract topics from text using the BERTopic model.

    TODO: Cache this output.

    Args:
        text: The text to extract topics from

    Returns:
        List of topic labels
    """
    # Get model instance
    model = get_topic_model()

    _num_topics = num_topics if num_topics is not None else settings.top_k_topics

    # Get topic indices and probabilities
    topic_indices, _ = model.transform([text])

    topics = []
    for i, topic_idx in enumerate(topic_indices):
        if _num_topics and i >= _num_topics:
            break
        # Convert possible numpy integer to Python int
        topic_idx_int = int(topic_idx)
        if topic_idx_int != -1:  # Skip outlier topic (-1)
            topic_info: list[tuple[str, float]] = model.get_topic(topic_idx_int)  # type: ignore
            if topic_info:
                topics.extend([info[0] for info in topic_info])

    return topics


async def handle_extraction(text: str) -> tuple[list[str], list[str]]:
    """
    Handle topic and entity extraction for a message.

    Args:
        text: The text to process

    Returns:
        Tuple of extracted topics and entities
    """
    # Extract topics if enabled
    topics = []
    if settings.enable_topic_extraction:
        if settings.topic_model_source == "BERTopic":
            topics = extract_topics_bertopic(text)
        else:
            topics = await extract_topics_llm(text)

    # Extract entities if enabled
    entities = []
    if settings.enable_ner:
        entities = extract_entities(text)

    # Merge with existing topics and entities
    if topics:
        topics = list(set(topics))
    if entities:
        entities = list(set(entities))

    return topics, entities


async def extract_memories_with_strategy(
    memories: list[MemoryRecord] | None = None,
    deduplicate: bool = True,
):
    """
    Extract memories using their configured strategies.

    This function replaces extract_discrete_memories for strategy-aware extraction.
    Each memory record contains its extraction strategy configuration.
    """
    # Local imports to avoid circular dependencies:
    # long_term_memory imports from extraction, so we import locally here
    from agent_memory_server.long_term_memory import index_long_term_memories
    from agent_memory_server.memory_strategies import get_memory_strategy
    from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

    adapter = await get_vectorstore_adapter()

    if not memories:
        # If no memories are provided, search for any messages in long-term memory
        # that haven't been processed for extraction using filter-only query
        # (no embedding required)
        memories = []
        offset = 0
        while True:
            search_result = await adapter.list_memories(
                memory_type=MemoryType(eq="message"),
                discrete_memory_extracted=DiscreteMemoryExtracted(eq="f"),
                limit=25,
                offset=offset,
            )

            logger.info(
                f"Found {len(search_result.memories)} memories to extract: {[m.id for m in search_result.memories]}"
            )

            memories += search_result.memories

            if len(search_result.memories) < 25:
                break

            offset += 25

    # Group memories by extraction strategy for batch processing
    strategy_groups = {}
    for memory in memories:
        if not memory or not memory.text:
            logger.info(f"Deleting memory with no text: {memory}")
            await adapter.delete_memories([memory.id])
            continue

        strategy_key = (
            memory.extraction_strategy,
            tuple(sorted(memory.extraction_strategy_config.items())),
        )
        if strategy_key not in strategy_groups:
            strategy_groups[strategy_key] = []
        strategy_groups[strategy_key].append(memory)

    all_new_memories = []
    all_updated_memories = []

    # Process each strategy group
    for (strategy_name, config_items), strategy_memories in strategy_groups.items():
        logger.info(
            f"Processing {len(strategy_memories)} memories with strategy: {strategy_name}"
        )

        # Get strategy instance
        config_dict = dict(config_items)
        try:
            strategy = get_memory_strategy(strategy_name, **config_dict)
        except ValueError as e:
            logger.error(f"Unknown strategy {strategy_name}: {e}")
            # Fall back to discrete strategy
            strategy = get_memory_strategy("discrete")

        # Process memories with this strategy
        for memory in strategy_memories:
            try:
                extracted_memories = await strategy.extract_memories(memory.text)
                all_new_memories.extend(extracted_memories)

                # Update the memory to mark it as processed
                updated_memory = memory.model_copy(
                    update={"discrete_memory_extracted": "t"}
                )
                all_updated_memories.append(updated_memory)

            except Exception as e:
                logger.error(
                    f"Error extracting memory {memory.id} with strategy {strategy_name}: {e}"
                )
                # Still mark as processed to avoid infinite retry
                updated_memory = memory.model_copy(
                    update={"discrete_memory_extracted": "t"}
                )
                all_updated_memories.append(updated_memory)

    # Update processed memories
    if all_updated_memories:
        await adapter.update_memories(all_updated_memories)

    # Index new extracted memories
    if all_new_memories:
        long_term_memories = [
            MemoryRecord(
                id=str(ulid.ULID()),
                text=new_memory["text"],
                memory_type=new_memory.get("type", "episodic"),
                topics=new_memory.get("topics", []),
                entities=new_memory.get("entities", []),
                discrete_memory_extracted="t",
                extraction_strategy="discrete",  # These are already extracted
                extraction_strategy_config={},
            )
            for new_memory in all_new_memories
        ]

        await index_long_term_memories(
            long_term_memories,
            deduplicate=deduplicate,
        )
