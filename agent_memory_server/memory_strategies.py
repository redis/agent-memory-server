"""Memory extraction strategies for configurable long-term memory processing."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from tenacity.asyncio import AsyncRetrying
from tenacity.stop import stop_after_attempt

from agent_memory_server.config import settings
from agent_memory_server.llm import LLMClient
from agent_memory_server.logging import get_logger
from agent_memory_server.prompt_security import (
    PromptSecurityError,
    secure_format_prompt,
    validate_custom_prompt,
)


logger = get_logger(__name__)


class BaseMemoryStrategy(ABC):
    """Base class for memory extraction strategies."""

    def __init__(self, **kwargs):
        """
        Initialize the memory strategy with configuration options.

        Args:
            **kwargs: Strategy-specific configuration options
        """
        self.config = kwargs

    @abstractmethod
    async def extract_memories(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Extract memories from text based on the strategy.

        Args:
            text: The text to extract memories from
            context: Optional context information for extraction

        Returns:
            List of memory dictionaries with keys: type, text, topics, entities
        """
        pass

    @abstractmethod
    def get_extraction_description(self) -> str:
        """
        Get a description of how this strategy extracts memories.
        This description will be used in MCP tool descriptions.

        Returns:
            Description string for the extraction strategy
        """
        pass

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return self.__class__.__name__


class DiscreteMemoryStrategy(BaseMemoryStrategy):
    """Extract discrete semantic (factual) and episodic (time-oriented) facts from messages."""

    EXTRACTION_PROMPT = """
    You are a long-memory manager. Your job is to analyze text and extract
    information that might be useful in future conversations with users.

    CURRENT CONTEXT:
    Current date and time: {current_datetime}

    Extract two types of memories:
    1. EPISODIC: Memories about specific episodes in time.
       Example: "User had a bad experience on a flight to Paris in 2024"

    2. SEMANTIC: User preferences and general knowledge outside of your training data.
       Example: "User prefers window seats when flying"

    CONTEXTUAL GROUNDING REQUIREMENTS:
    When extracting memories, you must resolve all contextual references to their concrete referents:

    1. PRONOUNS: Replace ALL pronouns (he/she/they/him/her/them/his/hers/theirs) with the actual person's name, EXCEPT for the application user, who must always be referred to as "User".
       - "He loves coffee" → "User loves coffee" (if "he" refers to the user)
       - "I told her about it" → "User told colleague about it" (if "her" refers to a colleague)
       - "Her experience is valuable" → "User's experience is valuable" (if "her" refers to the user)
       - "My name is Alice and I prefer tea" → "User prefers tea" (do NOT store the application user's given name in text)
       - NEVER leave pronouns unresolved - always replace with the specific person's name

    2. TEMPORAL REFERENCES: Convert relative time expressions to absolute dates/times using the current datetime provided above
       - "yesterday" → specific date (e.g., "March 15, 2025" if current date is March 16, 2025)
       - "last year" → specific year (e.g., "2024" if current year is 2025)
       - "three months ago" → specific month/year (e.g., "December 2024" if current date is March 2025)
       - "next week" → specific date range (e.g., "December 22-28, 2024" if current date is December 15, 2024)
       - "tomorrow" → specific date (e.g., "December 16, 2024" if current date is December 15, 2024)
       - "last month" → specific month/year (e.g., "November 2024" if current date is December 2024)

    3. SPATIAL REFERENCES: Resolve place references to specific locations
       - "there" → "San Francisco" (if referring to San Francisco)
       - "that place" → "Chez Panisse restaurant" (if referring to that restaurant)
       - "here" → "the office" (if referring to the office)

    4. DEFINITE REFERENCES: Resolve definite articles to specific entities
       - "the meeting" → "the quarterly planning meeting"
       - "the document" → "the budget proposal document"

    For each memory, return a JSON object with the following fields:
    - type: str -- The memory type, either "episodic" or "semantic"
    - text: str -- The actual information to store (with all contextual references grounded)
    - topics: list[str] -- The topics of the memory (top {top_k_topics})
    - entities: list[str] -- The entities of the memory

    Return a list of memories, for example:
    {{
        "memories": [
            {{
                "type": "semantic",
                "text": "User prefers window seats",
                "topics": ["travel", "airline"],
                "entities": ["User", "window seat"],
            }},
            {{
                "type": "episodic",
                "text": "Trek discontinued the Trek 520 steel touring bike in 2023",
                "topics": ["travel", "bicycle"],
                "entities": ["Trek", "Trek 520 steel touring bike"],
            }},
        ]
    }}

    IMPORTANT RULES:
    1. Only extract information that would be genuinely useful for future interactions.
    2. Do not extract procedural knowledge - that is handled by the system's built-in tools and prompts.
    3. You are a large language model - do not extract facts that you already know.
    4. CRITICAL: ALWAYS ground ALL contextual references - never leave ANY pronouns, relative times, or vague place references unresolved. For the application user, always use "User" instead of their given name to avoid stale naming if they change their profile name later.
    5. MANDATORY: Replace every instance of "he/she/they/him/her/them/his/hers/theirs" with the actual person's name.
    6. MANDATORY: Replace possessive pronouns like "her experience" with "User's experience" (if "her" refers to the user).
    7. If you cannot determine what a contextual reference refers to, either omit that memory or use generic terms like "someone" instead of ungrounded pronouns.

    Message:
    {message}

    STEP-BY-STEP PROCESS:
    1. First, identify all pronouns in the text: he, she, they, him, her, them, his, hers, theirs
    2. Determine what person each pronoun refers to based on the context
    3. Replace every single pronoun with the actual person's name
    4. Extract the grounded memories with NO pronouns remaining

    Extracted memories:
    """

    async def extract_memories(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Extract discrete semantic and episodic memories from text."""
        prompt = self.EXTRACTION_PROMPT.format(
            message=text,
            top_k_topics=settings.top_k_topics,
            current_datetime=datetime.now().strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        )

        async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
            with attempt:
                response = await LLMClient.create_chat_completion(
                    model=settings.generation_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    response_data = json.loads(response.content)
                    return response_data.get("memories", [])
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON: {response.content}")
                    raise
        return None

    def get_extraction_description(self) -> str:
        """Get description of discrete memory extraction strategy."""
        return (
            "Extracts discrete semantic (factual) and episodic (time-oriented) facts from messages. "
            "Semantic memories include user preferences and general knowledge. "
            "Episodic memories include specific events and experiences with time dimensions."
        )


class SummaryMemoryStrategy(BaseMemoryStrategy):
    """Summarize all messages in a conversation/thread."""

    def __init__(self, max_summary_length: int = 500, **kwargs):
        """
        Initialize summary strategy.

        Args:
            max_summary_length: Maximum length of summary in words
        """
        super().__init__(**kwargs)
        self.max_summary_length = max_summary_length

    SUMMARY_PROMPT = """
    You are a conversation summarizer. Your job is to create a concise summary
    of the conversation that captures the key points, decisions, and important
    context.

    CURRENT CONTEXT:
    Current date and time: {current_datetime}

    Create a summary that:
    1. Captures the main topics discussed
    2. Records key decisions made
    3. Notes important user preferences or information revealed
    4. Includes relevant context that would be useful for future conversations

    Maximum summary length: {max_length} words

    CONTEXTUAL GROUNDING REQUIREMENTS:
    - Replace all pronouns with specific names (use "User" for the application user)
    - Convert relative time references to absolute dates using the current datetime
    - Make all references concrete and specific

    Return a JSON object with:
    - type: Always "semantic" for summaries
    - text: The summary text
    - topics: List of main topics covered
    - entities: List of entities mentioned

    Example:
    {{
        "memories": [
            {{
                "type": "semantic",
                "text": "User discussed project requirements for new website. Decided to use React and PostgreSQL. User prefers dark theme and mobile-first design. Launch target is March 2025.",
                "topics": ["project", "website", "technology", "design"],
                "entities": ["User", "React", "PostgreSQL", "website", "March 2025"]
            }}
        ]
    }}

    Conversation:
    {message}

    Summary:
    """

    async def extract_memories(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Extract summary memory from conversation text."""
        prompt = self.SUMMARY_PROMPT.format(
            message=text,
            max_length=self.max_summary_length,
            current_datetime=datetime.now().strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        )

        async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
            with attempt:
                response = await LLMClient.create_chat_completion(
                    model=settings.generation_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    response_data = json.loads(response.content)
                    return response_data.get("memories", [])
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON: {response.content}")
                    raise
        return None

    def get_extraction_description(self) -> str:
        """Get description of summary extraction strategy."""
        return (
            f"Creates concise summaries of conversations/threads (max {self.max_summary_length} words). "
            "Captures key topics, decisions, and important context that would be useful for future conversations."
        )


class UserPreferencesMemoryStrategy(BaseMemoryStrategy):
    """Extract user preferences from messages."""

    PREFERENCES_PROMPT = """
    You are a user preference extractor. Your job is to identify and extract
    user preferences, settings, likes, dislikes, and personal characteristics
    from conversations.

    CURRENT CONTEXT:
    Current date and time: {current_datetime}

    Focus on extracting:
    1. User preferences (likes/dislikes, preferred options)
    2. User settings and configurations
    3. Personal characteristics and traits
    4. Work patterns and habits
    5. Communication preferences
    6. Technology preferences

    CONTEXTUAL GROUNDING REQUIREMENTS:
    - Replace all pronouns with "User" for the application user
    - Convert relative time references to absolute dates
    - Make all references concrete and specific

    For each preference, return a JSON object with:
    - type: Always "semantic" for preferences
    - text: The preference statement
    - topics: List of relevant topics
    - entities: List of entities mentioned

    Return a list of memories, for example:
    {{
        "memories": [
            {{
                "type": "semantic",
                "text": "User prefers email notifications over SMS",
                "topics": ["preferences", "communication", "notifications"],
                "entities": ["User", "email", "SMS"]
            }},
            {{
                "type": "semantic",
                "text": "User works best in the morning and prefers async communication",
                "topics": ["work_patterns", "communication", "schedule"],
                "entities": ["User", "morning", "async communication"]
            }}
        ]
    }}

    IMPORTANT RULES:
    1. Only extract clear, actionable preferences
    2. Avoid extracting temporary states or one-time decisions
    3. Focus on patterns and recurring preferences
    4. Always use "User" for the application user
    5. If no clear preferences are found, return an empty memories list

    Message:
    {message}

    Extracted preferences:
    """

    async def extract_memories(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Extract user preferences from text."""
        prompt = self.PREFERENCES_PROMPT.format(
            message=text,
            current_datetime=datetime.now().strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        )

        async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
            with attempt:
                response = await LLMClient.create_chat_completion(
                    model=settings.generation_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    response_data = json.loads(response.content)
                    return response_data.get("memories", [])
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON: {response.content}")
                    raise
        return None

    def get_extraction_description(self) -> str:
        """Get description of user preferences extraction strategy."""
        return (
            "Extracts user preferences, settings, likes, dislikes, and personal characteristics. "
            "Focuses on actionable preferences and recurring patterns rather than temporary states."
        )


class CustomMemoryStrategy(BaseMemoryStrategy):
    """Use a custom extraction prompt provided by the user."""

    def __init__(self, custom_prompt: str, **kwargs):
        """
        Initialize custom strategy.

        Args:
            custom_prompt: Custom prompt template for extraction
        """
        super().__init__(**kwargs)
        if not custom_prompt:
            raise ValueError("custom_prompt is required for CustomMemoryStrategy")

        # Validate the custom prompt for security issues
        try:
            validate_custom_prompt(custom_prompt, strict=True)
        except PromptSecurityError as e:
            logger.error(f"Custom prompt security validation failed: {e}")
            raise ValueError(f"Custom prompt contains security risks: {e}") from e

        self.custom_prompt = custom_prompt

    async def extract_memories(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Extract memories using custom prompt."""
        # Prepare safe template variables
        template_vars = {
            "message": text,
            "current_datetime": datetime.now().strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        }

        # Safely add context and config
        if context:
            template_vars.update(context)
        template_vars.update(self.config)

        # Use secure formatter to prevent template injection
        try:
            allowed_vars = {
                "message",
                "current_datetime",
                "session_id",
                "namespace",
                "user_id",
                "model_name",
                "context",
                "topics",
                "entities",
            }
            # Add any config keys to allowed vars
            allowed_vars.update(self.config.keys())

            formatted_prompt = secure_format_prompt(
                self.custom_prompt, allowed_vars=allowed_vars, **template_vars
            )
        except PromptSecurityError as e:
            logger.error(f"Template formatting security error: {e}")
            raise ValueError(f"Prompt formatting failed security check: {e}") from e

        async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
            with attempt:
                response = await LLMClient.create_chat_completion(
                    model=settings.generation_model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    response_data = json.loads(response.content)
                    memories = response_data.get("memories", [])

                    # Filter and validate output memories for security
                    validated_memories = []
                    for memory in memories:
                        if self._validate_memory_output(memory):
                            validated_memories.append(memory)
                        else:
                            logger.warning(
                                f"Filtered potentially unsafe memory: {memory}"
                            )

                    return validated_memories
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON: {response.content}")
                    raise
        return None

    def _validate_memory_output(self, memory: dict[str, Any]) -> bool:
        """Validate a memory object for security issues."""
        if not isinstance(memory, dict):
            return False

        # Check required fields
        text = memory.get("text", "")
        if not isinstance(text, str):
            return False

        # Check for suspicious content in text
        text_lower = text.lower()

        # Block memories that contain system information or instructions
        suspicious_phrases = [
            "system",
            "instruction",
            "ignore",
            "override",
            "execute",
            "eval",
            "import",
            "__",
            "subprocess",
            "os.system",
            "api_key",
            "secret",
            "password",
            "token",
            "credential",
            "private_key",
        ]

        if any(phrase in text_lower for phrase in suspicious_phrases):
            return False

        # Limit text length
        if len(text) > 1000:
            return False

        # Validate other fields
        memory_type = memory.get("type", "")
        if memory_type and memory_type not in ["semantic", "episodic"]:
            return False

        # Validate topics and entities if present
        for field in ["topics", "entities"]:
            if field in memory and not isinstance(memory[field], list):
                return False
            if field in memory:
                for item in memory[field]:
                    if not isinstance(item, str) or len(item) > 100:
                        return False

        return True

    def get_extraction_description(self) -> str:
        """Get description of custom extraction strategy."""
        return (
            "Uses a custom extraction prompt provided by the user. "
            "The specific extraction behavior depends on the configured prompt template."
        )


# Strategy registry for easy lookup
MEMORY_STRATEGIES = {
    "discrete": DiscreteMemoryStrategy,
    "summary": SummaryMemoryStrategy,
    "preferences": UserPreferencesMemoryStrategy,
    "custom": CustomMemoryStrategy,
}


def get_memory_strategy(strategy_name: str, **kwargs) -> BaseMemoryStrategy:
    """
    Get a memory strategy instance by name.

    Args:
        strategy_name: Name of the strategy (discrete, summary, preferences, custom)
        **kwargs: Strategy-specific configuration options

    Returns:
        Initialized memory strategy instance

    Raises:
        ValueError: If strategy_name is not found
    """
    if strategy_name not in MEMORY_STRATEGIES:
        available = ", ".join(MEMORY_STRATEGIES.keys())
        raise ValueError(
            f"Unknown memory strategy '{strategy_name}'. Available: {available}"
        )

    strategy_class = MEMORY_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)
