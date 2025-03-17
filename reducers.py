import logging

import tiktoken
from redis.asyncio import Redis

from models import (
    AnthropicClientWrapper,
    OpenAIClientWrapper,
    get_model_config,
)


logger = logging.getLogger(__name__)


async def _incremental_summary(
    model: str,
    client: OpenAIClientWrapper | AnthropicClientWrapper,
    context: str | None,
    messages: list[str],
) -> tuple[str, int]:
    """
    Incrementally summarize messages, building upon a previous summary.

    Args:
        model: The model to use (OpenAI or Anthropic)
        client: The client wrapper (OpenAI or Anthropic)
        context: Previous summary, if any
        messages: New messages to summarize

    Returns:
        Tuple of (summary, tokens_used)
    """
    # Reverse messages to put the most recent ones last
    messages.reverse()
    messages_joined = "\n".join(messages)
    prev_summary = context or ""

    # Prompt template for progressive summarization (from langchain)
    progressive_prompt = f"""
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. If the lines are meaningless just return NONE

EXAMPLE
Current summary:
The human asks who is the lead singer of Motorhead. The AI responds Lemmy Kilmister.
New lines of conversation:
Human: What are the other members of Motorhead?
AI: The original members included Lemmy Kilmister (vocals, bass), Larry Wallis (guitar), and Lucas Fox (drums), with notable members throughout the years including "Fast" Eddie Clarke (guitar), Phil "Philthy Animal" Taylor (drums), and Mikkey Dee (drums).
New summary:
The human asks who is the lead singer and other members of Motorhead. The AI responds Lemmy Kilmister is the lead singer and other original members include Larry Wallis, and Lucas Fox, with notable past members including "Fast" Eddie Clarke, Phil "Philthy Animal" Taylor, and Mikkey Dee.
END OF EXAMPLE

Current summary:
{prev_summary}
New lines of conversation:
{messages_joined}
New summary:
"""

    try:
        # Get completion from client
        response = await client.create_chat_completion(model, progressive_prompt)

        # Extract completion text
        completion = response.choices[0]["message"]["content"]

        # Get token usage
        tokens_used = response.total_tokens

        logger.info(f"Summarization complete, using {tokens_used} tokens")
        return completion, tokens_used
    except Exception as e:
        logger.error(f"Error in incremental summarization: {e}")
        raise


async def handle_compaction(
    session_id: str,
    model: str,
    window_size: int,
    client: OpenAIClientWrapper | AnthropicClientWrapper,
    redis_conn: Redis,
) -> None:
    """
    Handle compaction of messages when they exceed the window size.

    This function:
    1. Gets the second half of messages and current context
    2. Generates a new summary that includes these messages
    3. Removes older messages and updates the context

    Args:
        session_id: The session ID
        model: The model to use
        window_size: Maximum number of messages to keep
        client: The client wrapper (OpenAI or Anthropic)
        redis_conn: Redis connection
    """
    try:
        # Calculate half the window size
        half = window_size // 2

        # Get keys
        session_key = f"session:{session_id}"
        context_key = f"context:{session_id}"

        # Get messages and context from Redis
        pipe = redis_conn.pipeline()
        pipe.lrange(session_key, half, window_size)
        pipe.get(context_key)
        results = await pipe.execute()

        messages = results[0]
        context = results[1]

        # Get model configuration for token limits
        model_config = get_model_config(model)

        # Set up token limits based on model configuration
        max_tokens = model_config.max_tokens
        summary_max_tokens = max(
            512, max_tokens // 8
        )  # Use at least 512 tokens or 12.5% of model's context
        buffer_tokens = 230
        max_message_tokens = max_tokens - summary_max_tokens - buffer_tokens

        # Initialize encoding (currently uses OpenAI's tokenizer, but could be extended for different models)
        encoding = tiktoken.get_encoding("cl100k_base")

        # Check token count of messages
        total_tokens = 0
        messages_to_summarize = []

        for msg in messages:
            # Decode message if needed
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")

            msg_tokens = len(encoding.encode(msg))
            if total_tokens + msg_tokens <= max_message_tokens:
                total_tokens += msg_tokens
                messages_to_summarize.append(msg)

        # Skip if no messages to summarize
        if not messages_to_summarize:
            logger.info(f"No messages to summarize for session {session_id}")
            return

        # Generate new summary
        summary, _ = await _incremental_summary(
            model,
            client,
            context.decode("utf-8") if isinstance(context, bytes) else context,
            messages_to_summarize,
        )

        # Update Redis
        pipe = redis_conn.pipeline()
        pipe.set(context_key, summary)

        # Remove messages that were summarized (up to half the window)
        for _ in range(half):
            pipe.lpop(session_key)

        await pipe.execute()
        logger.info(f"Compaction complete for session {session_id}")

    except Exception as e:
        logger.error(f"Error in handle_compaction: {e}")
        raise
