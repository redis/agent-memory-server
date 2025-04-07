import os
from typing import Any

from bertopic import BERTopic
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from redis_memory_server.config import settings
from redis_memory_server.logging import get_logger


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
        _ner_tokenizer = AutoTokenizer.from_pretrained(settings.ner_model)
        _ner_model = AutoModelForTokenClassification.from_pretrained(settings.ner_model)
    return pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)


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


def extract_topics(text: str, num_topics: int | None = None) -> list[str]:
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

    # Get topic indices and probabilities
    topic_indices, _ = model.transform([text])

    topics = []
    for i, topic_idx in enumerate(topic_indices):
        if num_topics and i >= num_topics:
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
        topics = extract_topics(text)

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
