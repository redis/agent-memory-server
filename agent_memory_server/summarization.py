import json
import logging

import tiktoken
from redis import WatchError

from agent_memory_server.config import settings
from agent_memory_server.llm import LLMClient, get_model_config
from agent_memory_server.models import MemoryMessage
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import get_redis_conn


logger = logging.getLogger(__name__)


async def _incremental_summary(
    model: str,
    context: str | None,
    messages: list[str],
) -> tuple[str, int]:
    """
    Incrementally summarize messages, building upon a previous summary.

    Args:
        model: The model to use for summarization
        context: Previous summary, if any
        messages: New messages to summarize

    Returns:
        Tuple of (summary, tokens_used)
    """
    # Reverse messages to put the most recent ones last
    messages.reverse()
    messages_joined = "\n".join(messages)
    prev_summary = context or ""

    # Use configurable prompt template for progressive summarization
    progressive_prompt = settings.progressive_summarization_prompt.format(
        prev_summary=prev_summary, messages_joined=messages_joined
    )

    try:
        # Get completion from LLMClient
        response = await LLMClient.create_chat_completion(
            model=model,
            messages=[{"role": "user", "content": progressive_prompt}],
        )

        # Extract completion text
        completion = response.content

        # Get token usage
        tokens_used = response.total_tokens

        logger.info(f"Summarization complete, using {tokens_used} tokens")
        return completion, tokens_used
    except Exception as e:
        logger.error(f"Error in incremental summarization: {e}")
        raise


async def summarize_session(
    session_id: str,
    model: str,
    max_context_tokens: int | None = None,
) -> None:
    """
    Summarize messages in a session when they exceed the token limit.

    This function:
    1. Gets messages and current context
    2. Generates a new summary that includes older messages
    3. Removes older, summarized messages and updates the context

    Args:
        session_id: The session ID
        model: The model to use for summarization
        max_context_tokens: Maximum context tokens to keep (defaults to model's context window * summarization_threshold)
    """
    logger.debug(f"Summarizing session {session_id}")
    redis = await get_redis_conn()

    messages_key = Keys.messages_key(session_id)
    metadata_key = Keys.metadata_key(session_id)

    async with redis.pipeline(transaction=False) as pipe:
        await pipe.watch(messages_key, metadata_key)

        num_messages = await pipe.llen(messages_key)  # type: ignore
        logger.debug(f"[summarization] Number of messages: {num_messages}")
        if num_messages < 2:  # Need at least 2 messages to summarize
            logger.info(f"Not enough messages to summarize for session {session_id}")
            return

        messages_raw = await pipe.lrange(messages_key, 0, -1)  # Get all messages
        metadata = await pipe.hgetall(metadata_key)  # type: ignore
        pipe.multi()

        while True:
            try:
                messages = []
                for msg_raw in messages_raw:
                    if isinstance(msg_raw, bytes):
                        msg_raw = msg_raw.decode("utf-8")
                    msg_dict = json.loads(msg_raw)
                    messages.append(MemoryMessage(**msg_dict))

                logger.debug(f"[summarization] Messages: {messages}")

                model_config = get_model_config(model)
                full_context_tokens = model_config.max_tokens

                # Use provided max_context_tokens or calculate from model context window
                if max_context_tokens is None:
                    max_context_tokens = int(
                        full_context_tokens * settings.summarization_threshold
                    )

                # Calculate current token usage
                encoding = tiktoken.get_encoding("cl100k_base")
                current_tokens = sum(
                    len(encoding.encode(f"{msg.role}: {msg.content}"))
                    for msg in messages
                )

                # If we're under the limit, no need to summarize
                if current_tokens <= max_context_tokens:
                    logger.info(
                        f"Messages under token limit ({current_tokens} <= {max_context_tokens}) for session {session_id}"
                    )
                    return

                # Token allocation for summarization
                if full_context_tokens < 10000:
                    summary_max_tokens = max(512, full_context_tokens // 8)  # 12.5%
                elif full_context_tokens < 50000:
                    summary_max_tokens = max(1024, full_context_tokens // 10)  # 10%
                else:
                    summary_max_tokens = max(2048, full_context_tokens // 20)  # 5%

                logger.debug(
                    f"[summarization] Summary max tokens: {summary_max_tokens}"
                )

                # Scale buffer tokens with context size, but keep reasonable bounds
                buffer_tokens = min(max(230, full_context_tokens // 100), 1000)

                logger.debug(f"[summarization] Buffer tokens: {buffer_tokens}")

                max_message_tokens = (
                    full_context_tokens - summary_max_tokens - buffer_tokens
                )

                # Determine how many messages to keep (target ~40% of max_context_tokens for recent messages)
                target_remaining_tokens = int(max_context_tokens * 0.4)

                # Work backwards to find recent messages to keep
                recent_messages_tokens = 0
                keep_count = 0

                for i in range(len(messages) - 1, -1, -1):
                    msg = messages[i]
                    msg_str = f"{msg.role}: {msg.content}"
                    msg_tokens = len(encoding.encode(msg_str))

                    if recent_messages_tokens + msg_tokens <= target_remaining_tokens:
                        recent_messages_tokens += msg_tokens
                        keep_count += 1
                    else:
                        break

                # Messages to summarize are the ones we're not keeping
                messages_to_check = (
                    messages[:-keep_count] if keep_count > 0 else messages[:-1]
                )

                total_tokens = 0
                messages_to_summarize = []

                for msg in messages_to_check:
                    msg_str = f"{msg.role}: {msg.content}"
                    msg_tokens = len(encoding.encode(msg_str))

                    # Handle oversized messages
                    if msg_tokens > max_message_tokens:
                        msg_str = msg_str[: max_message_tokens // 2]
                        msg_tokens = len(encoding.encode(msg_str))

                    if total_tokens + msg_tokens <= max_message_tokens:
                        total_tokens += msg_tokens
                        messages_to_summarize.append(msg_str)
                    else:
                        break

                if not messages_to_summarize:
                    logger.info(f"No messages to summarize for session {session_id}")
                    return

                context = metadata.get("context", "")

                summary, summary_tokens_used = await _incremental_summary(
                    model,
                    context,
                    messages_to_summarize,
                )
                total_tokens += summary_tokens_used

                metadata["context"] = summary
                metadata["tokens"] = str(total_tokens)

                pipe.hmset(metadata_key, mapping=metadata)
                logger.debug(f"[summarization] Metadata: {metadata_key} {metadata}")

                # Keep only the most recent messages that fit in our token budget
                if keep_count > 0:
                    # Keep the last keep_count messages
                    pipe.ltrim(messages_key, -keep_count, -1)
                else:
                    # Keep at least the last message
                    pipe.ltrim(messages_key, -1, -1)

                await pipe.execute()
                break
            except WatchError:
                continue

    logger.info(f"Summarization complete for session {session_id}")
