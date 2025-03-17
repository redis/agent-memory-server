import json
import logging

from redis_memory_server.models import (
    AnthropicClientWrapper,
    MemoryMessage,
    OpenAIClientWrapper,
)


logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analyze the following message and extract:
1. Key topics (as single words or short phrases)
2. Named entities (people, places, organizations, etc.)

Message: {message}

Respond in JSON format:
{{
    "topics": ["topic1", "topic2", ...],
    "entities": ["entity1", "entity2", ...]
}}

Keep topics and entities concise and relevant."""


async def extract_topics_and_entities(
    message: str,
    model_client: OpenAIClientWrapper | AnthropicClientWrapper,
) -> tuple[list[str], list[str]]:
    """
    Extract topics and entities from a message using the LLM.

    Args:
        message: The message to analyze
        model_client: The LLM client to use

    Returns:
        Tuple of (topics, entities) lists
    """
    try:
        # Get LLM response
        response = await model_client.create_chat_completion(
            "gpt-4o-mini",  # TODO: Make configurable
            EXTRACTION_PROMPT.format(message=message),
        )

        # Parse JSON response from content field
        content = response.choices[0]["message"]["content"].strip()
        result = json.loads(content)

        # Extract and validate topics and entities
        topics = result.get("topics", [])
        entities = result.get("entities", [])

        # Ensure we have lists
        if not isinstance(topics, list) or not isinstance(entities, list):
            logger.error("Invalid extraction response format")
            return [], []

        return topics, entities

    except Exception as e:
        logger.error(f"Error in topic/entity extraction: {e}")
        return [], []


async def handle_extraction(
    message: MemoryMessage,
    model_client: OpenAIClientWrapper | AnthropicClientWrapper,
) -> MemoryMessage:
    """
    Handle topic and entity extraction for a message.

    Args:
        message: The message to process
        model_client: The LLM client to use

    Returns:
        Updated message with extracted topics and entities
    """
    # Skip if message already has both topics and entities
    if message.topics and message.entities:
        return message

    # Extract topics and entities
    extracted_topics, extracted_entities = await extract_topics_and_entities(
        message.content, model_client
    )

    # Merge with existing topics and entities
    message.topics = (
        list(set(message.topics + extracted_topics))
        if message.topics
        else extracted_topics
    )
    message.entities = (
        list(set(message.entities + extracted_entities))
        if message.entities
        else extracted_entities
    )

    return message
