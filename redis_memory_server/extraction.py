import os
from typing import Any

from bertopic import BERTopic
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from redis_memory_server.config import settings
from redis_memory_server.logging import get_logger
from redis_memory_server.models import (
    MemoryMessage,
)


logger = get_logger(__name__)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model instances
_topic_model: BERTopic | None = None
_ner_model: Any | None = None
_ner_tokenizer: Any | None = None


def get_topic_model() -> BERTopic:
    """
    Get or initialize the BERTopic model.

    Returns:
        The BERTopic model instance
    """
    global _topic_model
    if _topic_model is None:
        _topic_model = BERTopic.load(settings.topic_model)
    return _topic_model  # type: ignore


def get_ner_model() -> Any:
    """
    Get or initialize the NER model and tokenizer.

    Returns:
        The NER pipeline instance
    """
    global _ner_model, _ner_tokenizer
    if _ner_model is None:
        _ner_tokenizer = AutoTokenizer.from_pretrained(settings.ner_model)
        _ner_model = AutoModelForTokenClassification.from_pretrained(settings.ner_model)
    return pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)


def extract_entities(text: str) -> list[str]:
    """
    Extract named entities from text using the NER model.

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


def extract_topics(text: str) -> list[str]:
    """
    Extract topics from text using the BERTopic model.

    Args:
        text: The text to extract topics from

    Returns:
        List of topic labels
    """
    # Get model instance
    model = get_topic_model()

    # Get topic indices and probabilities
    topic_indices, _ = model.transform([text])

    topics = []
    for idx in topic_indices:
        # Convert possible numpy integer to Python int
        idx_int = int(idx)
        if idx_int != -1:  # Skip outlier topic (-1)
            topic_info = model.get_topic(idx_int)
            if topic_info:
                top_topics = [t[0] for t in topic_info[0:2]]  # type: ignore
                topics.extend(top_topics)

    return topics


async def handle_extraction(
    message: MemoryMessage,
) -> MemoryMessage:
    """
    Handle topic and entity extraction for a message.

    Args:
        message: The message to process

    Returns:
        Updated message with extracted topics and entities
    """
    # Extract topics if enabled
    topics = []
    if settings.enable_topic_extraction:
        topics = extract_topics(message.content)

    # Extract entities if enabled
    entities = []
    if settings.enable_ner:
        entities = extract_entities(message.content)

    # Merge with existing topics and entities
    if topics:
        message.topics = (
            list(set(message.topics + topics)) if message.topics else topics
        )
    if entities:
        message.entities = (
            list(set(message.entities + entities)) if message.entities else entities
        )

    return message
