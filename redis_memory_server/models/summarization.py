import json
import logging

import tiktoken
from redis.asyncio import Redis

from redis_memory_server.llms import (
    AnthropicClientWrapper,
    OpenAIClientWrapper,
    get_model_config,
)
from redis_memory_server.models.messages import MemoryMessage
from redis_memory_server.utils import Keys


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

    # Prompt template for progressive summarization
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
    Handle compaction of messages in a session when they exceed the window size.

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
        messages_key = Keys.messages_key(session_id)
        context_key = Keys.context_key(session_id)

        # Get messages and context from Redis
        pipe = redis_conn.pipeline()
        pipe.lrange(messages_key, half, window_size)
        pipe.get(context_key)
        results = await pipe.execute()

        messages_raw = results[0]
        context = results[1]

        # Parse messages
        messages = []
        for msg_raw in messages_raw:
            if isinstance(msg_raw, bytes):
                msg_raw = msg_raw.decode("utf-8")
            msg_dict = json.loads(msg_raw)
            messages.append(MemoryMessage(**msg_dict))

        # Get context string
        context_str = context.decode("utf-8") if isinstance(context, bytes) else context

        # Get model configuration for token limits
        model_config = get_model_config(model)

        # Set up token limits based on model configuration
        max_tokens = model_config.max_tokens

        # More nuanced summary token allocation:
        # - For small context (<10k): use 12.5% (min 512)
        # - For medium context (10k-50k): use 10% (min 1024)
        # - For large context (>50k): use 5% (min 2048)
        if max_tokens < 10000:
            summary_max_tokens = max(512, max_tokens // 8)  # 12.5%
        elif max_tokens < 50000:
            summary_max_tokens = max(1024, max_tokens // 10)  # 10%
        else:
            summary_max_tokens = max(2048, max_tokens // 20)  # 5%

        # Scale buffer tokens with context size, but keep reasonable bounds
        buffer_tokens = min(max(230, max_tokens // 100), 1000)

        max_message_tokens = max_tokens - summary_max_tokens - buffer_tokens

        # Initialize encoding
        encoding = tiktoken.get_encoding("cl100k_base")

        # Check token count of messages
        total_tokens = 0
        messages_to_summarize = []

        for msg in messages:
            msg_str = json.dumps(msg.model_dump())
            msg_tokens = len(encoding.encode(msg_str))
            if total_tokens + msg_tokens <= max_message_tokens:
                total_tokens += msg_tokens
                messages_to_summarize.append(msg_str)

        # Skip if no messages to summarize
        if not messages_to_summarize:
            logger.info(f"No messages to summarize for session {session_id}")
            return

        # Generate new summary
        summary, _ = await _incremental_summary(
            model,
            client,
            context_str,
            messages_to_summarize,
        )

        # Update Redis
        pipe = redis_conn.pipeline()
        pipe.set(context_key, summary)

        # Remove messages that were summarized (up to half the window)
        for _ in range(half):
            pipe.lpop(messages_key)

        await pipe.execute()
        logger.info(f"Compaction complete for session {session_id}")

    except Exception as e:
        logger.error(f"Error in handle_compaction: {e}")
        raise
