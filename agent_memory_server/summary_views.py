"""Helpers for SummaryView configs, stored results, and summarization logic.

This module implements the execution logic for summarizing long-term memory
sources using LLMs, including Redis JSON storage, key conventions, and
partitioned summary management so the API surface is wired end-to-end.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote, unquote

import tiktoken
from docket import Perpetual, Timeout

from agent_memory_server import long_term_memory
from agent_memory_server.config import settings
from agent_memory_server.filters import (
    CreatedAt,
    MemoryType,
    Namespace,
    SessionId,
    UserId,
)
from agent_memory_server.models import (
    MemoryRecord,
    SummaryView,
    SummaryViewPartitionResult,
    TaskStatusEnum,
)
from agent_memory_server.tasks import update_task_status
from agent_memory_server.utils.redis import get_redis_conn


logger = logging.getLogger(__name__)


_SUMMARY_VIEW_INDEX_KEY = "summary_view:index"

# Conservative cap on how many memories we inline into a single LLM prompt.
# We still report the full memory_count separately.
_MAX_MEMORIES_FOR_LLM_PROMPT = 200


def _config_key(view_id: str) -> str:
    return f"summary_view:{view_id}:config"


def _summary_key(view_id: str, partition_key: str) -> str:
    return f"summary_view:{view_id}:summary:{partition_key}"


def encode_partition_key(group: dict[str, str]) -> str:
    """Create a stable key representation from group_by values.

    Keys are sorted alphabetically so the same group always produces the
    same identifier. Both keys and values are URL-encoded to handle any
    special characters (including '|' and '=') that could otherwise cause
    ambiguous or colliding partition keys.

    The resulting string is treated as an opaque key; use
    decode_partition_key() if you need to reverse the encoding.
    """

    parts: list[str] = []
    for key in sorted(group.keys()):
        encoded_key = quote(key, safe="")
        encoded_value = quote(group[key], safe="")
        parts.append(f"{encoded_key}={encoded_value}")
    return "|".join(parts)


def decode_partition_key(partition_key: str) -> dict[str, str]:
    """Decode a partition key back into a group dictionary.

    This reverses the URL-encoding applied by encode_partition_key().
    """

    if not partition_key:
        return {}

    result: dict[str, str] = {}
    for part in partition_key.split("|"):
        if "=" not in part:
            continue
        encoded_key, encoded_value = part.split("=", 1)
        result[unquote(encoded_key)] = unquote(encoded_value)
    return result


def _matches_group_filter(group: dict[str, str], group_filter: dict[str, str]) -> bool:
    return all(group.get(key) == value for key, value in group_filter.items())


async def save_summary_view(view: SummaryView) -> None:
    """Persist a SummaryView definition in Redis as JSON and index it."""

    redis = await get_redis_conn()
    await redis.set(_config_key(view.id), view.model_dump_json())
    await redis.sadd(_SUMMARY_VIEW_INDEX_KEY, view.id)


async def get_summary_view(view_id: str) -> SummaryView | None:
    """Load a SummaryView by ID from Redis JSON storage."""

    redis = await get_redis_conn()
    raw = await redis.get(_config_key(view_id))
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        return SummaryView.model_validate_json(raw)
    except Exception:
        logger.exception("Failed to decode SummaryView JSON for %s", view_id)
        return None


async def list_summary_views() -> list[SummaryView]:
    """Return all SummaryViews registered in the index.

    This performs one GET per view ID; acceptable for the current scale.
    """

    redis = await get_redis_conn()
    ids: Iterable[bytes] = await redis.smembers(_SUMMARY_VIEW_INDEX_KEY)
    views: list[SummaryView] = []

    for raw_id in ids:
        view_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        view = await get_summary_view(view_id)
        if view is not None:
            views.append(view)

    return views


async def delete_summary_view(view_id: str) -> None:
    """Delete a SummaryView config and remove it from the index.

    Stored partition summaries are left as-is for now; they can be cleaned
    up in a later pass if needed.
    """

    redis = await get_redis_conn()
    await redis.delete(_config_key(view_id))
    await redis.srem(_SUMMARY_VIEW_INDEX_KEY, view_id)


async def save_partition_result(result: SummaryViewPartitionResult) -> None:
    """Persist a single partition result for a SummaryView."""

    redis = await get_redis_conn()
    partition_key = encode_partition_key(result.group)
    await redis.set(
        _summary_key(result.view_id, partition_key), result.model_dump_json()
    )


async def list_partition_results(
    view_id: str, group_filter: dict[str, str] | None = None
) -> list[SummaryViewPartitionResult]:
    """List stored partition results for a view, optionally filtered by group.

    This reads whatever has been materialized so far; it does not trigger
    recomputation.
    """

    redis = await get_redis_conn()
    pattern = _summary_key(view_id, "*")
    results: list[SummaryViewPartitionResult] = []

    async for key in redis.scan_iter(match=pattern):
        raw = await redis.get(key)
        if raw is None:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            result = SummaryViewPartitionResult.model_validate_json(raw)
        except Exception:
            logger.exception(
                "Failed to decode SummaryViewPartitionResult for key %s", key
            )
            continue

        if group_filter and not _matches_group_filter(result.group, group_filter):
            continue

        results.append(result)

    return results


def _build_long_term_filters_for_view(
    view: SummaryView,
    extra_group: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for search_long_term_memories.

    Maps SummaryView.filters and optionally a concrete group dict into
    typed filter objects used by long_term_memory.search_long_term_memories.
    """

    filters: dict[str, Any] = {}

    def _apply_filter(key: str, value: str | Any) -> None:
        """Apply a single filter mapping from a raw key/value pair.

        Both static view.filters and extra_group values are coerced to str
        for consistency.
        """

        if key == "user_id":
            filters["user_id"] = UserId(eq=str(value))
        elif key == "namespace":
            filters["namespace"] = Namespace(eq=str(value))
        elif key == "session_id":
            filters["session_id"] = SessionId(eq=str(value))
        elif key == "memory_type":
            filters["memory_type"] = MemoryType(eq=str(value))

    # Static filters from the view config
    for key, value in view.filters.items():
        _apply_filter(key, value)

    # Group-specific filters
    if extra_group:
        for key, value in extra_group.items():
            _apply_filter(key, value)

    # Time window: apply to created_at for now
    if view.time_window_days is not None and view.time_window_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=view.time_window_days)
        filters["created_at"] = CreatedAt(gte=cutoff)

    return filters


async def _fetch_long_term_memories_for_view(
    view: SummaryView,
    extra_group: dict[str, str] | None = None,
    page_size: int = 1000,
    overall_limit: int | None = None,
) -> list[MemoryRecord]:
    """Fetch long-term memories matching a SummaryView and optional group.

    Uses the filter-only listing path of search_long_term_memories by
    providing an empty text query and paginating through results.

    If overall_limit is provided, it serves as an upper bound on the total
    number of memories returned; otherwise, all available pages are fetched.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    filters = _build_long_term_filters_for_view(view, extra_group)

    memories: list[MemoryRecord] = []
    offset = 0

    while True:
        # Respect an overall cap if provided
        if overall_limit is not None:
            remaining = overall_limit - len(memories)
            if remaining <= 0:
                break
            current_limit = min(page_size, remaining)
        else:
            current_limit = page_size

        results = await long_term_memory.search_long_term_memories(
            text="",
            limit=current_limit,
            offset=offset,
            **filters,
        )
        batch = list(results.memories)
        if not batch:
            break

        memories.extend(batch)

        # If fewer results than requested were returned, we've reached the
        # end of the result set.
        if len(batch) < current_limit:
            break

        offset += len(batch)

    # If we applied an overall limit, enforce it defensively here too.
    if overall_limit is not None and len(memories) > overall_limit:
        memories = memories[:overall_limit]

    return memories


def _partition_memories_by_group(
    view: SummaryView, memories: list[MemoryRecord]
) -> dict[tuple[tuple[str, str], ...], list[MemoryRecord]]:
    """Group memories into partitions based on view.group_by fields.

    Returns a mapping from a stable tuple key to a list of MemoryRecord.
    The key is a sorted tuple of (field, value) pairs.
    """

    partitions: dict[tuple[tuple[str, str], ...], list[MemoryRecord]] = {}
    for mem in memories:
        group_dict: dict[str, str] = {}
        for field in view.group_by:
            value = getattr(mem, field, None)
            if value is None:
                # If a grouping field is missing for this memory, skip it
                break
            group_dict[field] = str(value)
        else:
            # Only executed if the inner loop did not break
            key = tuple(sorted(group_dict.items()))
            partitions.setdefault(key, []).append(mem)

    return partitions


def _build_long_term_summary_prompt(
    view: SummaryView,
    group: dict[str, str],
    memories: list[MemoryRecord],
    model_name: str,
    instructions: str,
) -> str:
    """Build a token-aware prompt for long-term memory summarization.

    Uses tiktoken and the model's configured context window to truncate the
    inlined memories so the prompt stays within a safe fraction of the
    context limit while still leaving room for the model's response.
    """

    # Import here to avoid circular imports at module load time.
    from agent_memory_server.llm import get_model_config

    encoding = tiktoken.get_encoding("cl100k_base")

    model_config = get_model_config(model_name)
    full_context_tokens = max(model_config.max_tokens, 1)

    # Use the same summarization_threshold knob as working-memory
    # summarization to control how much of the context window we devote
    # to the prompt itself.
    prompt_budget = int(full_context_tokens * settings.summarization_threshold)

    # Reserve some space for the model's response and any overhead.
    reserved_completion_tokens = min(4096, full_context_tokens // 10)
    max_prompt_tokens = max(prompt_budget - reserved_completion_tokens, 1024)

    base_prefix = (
        f"{instructions}\n\nGROUP: {json.dumps(group, sort_keys=True)}\n\nMEMORIES:\n"
    )
    base_tokens = len(encoding.encode(base_prefix))

    remaining_tokens = max_prompt_tokens - base_tokens
    if remaining_tokens <= 0:
        return (
            base_prefix
            + "[Memories omitted due to token budget constraints.]\n\nSUMMARY:"
        )

    # Cap the size of each individual memory text we inline so that a
    # single extremely long memory cannot dominate the prompt.
    max_bullet_tokens = min(1024, full_context_tokens // 20)

    bullet_lines: list[str] = []
    for mem in memories[:_MAX_MEMORIES_FOR_LLM_PROMPT]:
        text = mem.text or ""
        bullet = f"- {text}"
        bullet_tokens = len(encoding.encode(bullet))

        if bullet_tokens > max_bullet_tokens:
            # Roughly truncate very long memories by characters, then
            # recompute tokens. This mirrors the approach used in
            # agent_memory_server.summarization.
            approx_chars = max_bullet_tokens * 4
            text = text[:approx_chars]
            bullet = f"- {text}"
            bullet_tokens = len(encoding.encode(bullet))

        if bullet_tokens > remaining_tokens:
            break

        bullet_lines.append(bullet)
        remaining_tokens -= bullet_tokens

    memories_text = "\n".join(bullet_lines)
    total_memories = len(memories)
    used_memories = len(bullet_lines)
    if total_memories > used_memories:
        memories_text += (
            f"\n\n[Memories truncated to fit token budget: used {used_memories} "
            f"of {total_memories} entries]"
        )

    return f"{base_prefix}{memories_text}\n\nSUMMARY:"


async def summarize_partition_long_term(
    view: SummaryView,
    group: dict[str, str],
    memories: list[MemoryRecord],
) -> SummaryViewPartitionResult:
    """Summarize a partition of long-term memories.

    For now we keep the prompt simple and use a single chat completion
    call with a textual join of memory texts.
    """

    if not memories:
        summary_text = f"No memories found for group {group!r}."
        return SummaryViewPartitionResult(
            view_id=view.id,
            group=group,
            summary=summary_text,
            memory_count=0,
            computed_at=datetime.now(UTC),
        )

    # If no LLM credentials are configured, fall back to a simple
    # deterministic summary that just concatenates memory texts.
    if not (
        settings.openai_api_key
        or settings.anthropic_api_key
        or settings.aws_access_key_id
    ):
        joined = "\n".join(f"- {m.text}" for m in memories[:50])
        summary_text = (
            "LLM summarization disabled (no API keys configured). "
            "Concatenated up to 50 memories:\n" + joined
        )
    else:
        from agent_memory_server.llm import LLMClient

        model_name = view.model_name or settings.fast_model

        # Build a prompt using either the view's prompt or a default, then
        # construct a token-aware memories section based on the model's
        # configured context window.
        default_instructions = (
            "You are a summarization assistant. Given a set of long-term "
            "memories, produce a concise summary that highlights key facts, "
            "stable preferences, and important events relevant to the group."
        )
        instructions = view.prompt or default_instructions

        prompt = _build_long_term_summary_prompt(
            view=view,
            group=group,
            memories=memories,
            model_name=model_name,
            instructions=instructions,
        )

        # We use the same interface pattern as other summarization helpers,
        # but add minimal defensive checks around the response structure.
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await LLMClient.create_chat_completion(model_name, messages)
            content = response.content
        except Exception:
            logger.exception(
                "Error calling summarization model %s for SummaryView %s group %r",
                model_name,
                view.id,
                group,
            )
            content = None

        if not content:
            logger.warning(
                "Summarization model %s returned empty response for SummaryView %s "
                "group %r; using fallback text.",
                model_name,
                view.id,
                group,
            )
            summary_text = "No summary could be generated for this partition."
        else:
            summary_text = content

    return SummaryViewPartitionResult(
        view_id=view.id,
        group=group,
        summary=summary_text,
        memory_count=len(memories),
        computed_at=datetime.now(UTC),
    )


async def summarize_partition_placeholder(
    view: SummaryView, group: dict[str, str]
) -> SummaryViewPartitionResult:
    """Fallback placeholder summary for unsupported sources.

    Used currently for SummaryViews whose source is not yet implemented.
    """

    summary_text = f"Placeholder summary for view {view.id} with group {group!r}."
    return SummaryViewPartitionResult(
        view_id=view.id,
        group=group,
        summary=summary_text,
        memory_count=0,
        computed_at=datetime.now(UTC),
    )


async def summarize_partition_for_view(
    view: SummaryView, group: dict[str, str]
) -> SummaryViewPartitionResult:
    """High-level entry point to summarize a single partition for a view.

    Dispatches to the appropriate implementation based on view.source.
    """

    if view.source == "long_term":
        memories = await _fetch_long_term_memories_for_view(view, extra_group=group)
        return await summarize_partition_long_term(view, group, memories)

    # Fallback for sources we haven't implemented yet.
    return await summarize_partition_placeholder(view, group)


async def refresh_summary_view(
    view_id: str,
    task_id: str | None = None,
    timeout: Timeout = Timeout(timedelta(minutes=settings.llm_task_timeout_minutes)),
) -> None:
    """Docket task to recompute all partitions for a SummaryView.

    For long-term memory sources, this will fetch memories matching the
    view filters, partition them by group_by, summarize each partition,
    and store SummaryViewPartitionResult entries.

    Args:
        view_id: The ID of the SummaryView to refresh
        task_id: Optional task ID for status tracking
        timeout: Docket timeout for this task (defaults to llm_task_timeout_minutes from settings)
    """

    view = await get_summary_view(view_id)
    now = datetime.now(UTC)

    if task_id is not None:
        if view is None:
            await update_task_status(
                task_id,
                status=TaskStatusEnum.FAILED,
                completed_at=now,
                error_message=f"SummaryView {view_id} not found",
            )
            return

        await update_task_status(
            task_id,
            status=TaskStatusEnum.RUNNING,
            started_at=now,
        )

    if view is None:
        # Nothing to do; already handled task status above if needed.
        return

    # Threshold above which we log a warning about large memory sets.
    # This helps operators identify views that may benefit from tighter
    # filters or time windows.
    large_memory_threshold = 5000

    try:
        if view.source == "long_term":
            # Fetch all relevant memories and partition them.
            memories = await _fetch_long_term_memories_for_view(view)

            if len(memories) >= large_memory_threshold:
                logger.warning(
                    "refresh_summary_view: fetched %d memories for view %s; "
                    "consider adding filters or a time_window_days to reduce volume.",
                    len(memories),
                    view.id,
                )

            partitions = _partition_memories_by_group(view, memories)

            for key, mems in partitions.items():
                group = dict(key)
                result = await summarize_partition_long_term(view, group, mems)
                await save_partition_result(result)
        else:
            # For unsupported sources, we currently do nothing.
            logger.info(
                "refresh_summary_view: source %s not yet implemented for view %s",
                view.source,
                view.id,
            )

        if task_id is not None:
            await update_task_status(
                task_id,
                status=TaskStatusEnum.SUCCESS,
                completed_at=datetime.now(UTC),
            )
    except Exception as exc:  # noqa: BLE001
        # We deliberately catch all exceptions here so that background workers
        # never crash silently and any failure is reflected in the Task record
        # as FAILED. The original error is logged with traceback above.
        logger.exception("Error refreshing SummaryView %s", view_id)
        if task_id is not None:
            await update_task_status(
                task_id,
                status=TaskStatusEnum.FAILED,
                completed_at=datetime.now(UTC),
                error_message=str(exc),
            )


async def periodic_refresh_summary_views(
    perpetual: Perpetual = Perpetual(
        every=timedelta(minutes=60),
        automatic=True,
    ),
    timeout: Timeout = Timeout(
        timedelta(minutes=settings.llm_task_timeout_minutes * 5)
    ),
) -> None:
    """Periodic Docket task to refresh all continuous SummaryViews.

    Uses the same refresh_summary_view helper but without task tracking.

    Args:
        perpetual: Docket perpetual schedule configuration
        timeout: Docket timeout for this task (defaults to 5x llm_task_timeout_minutes
                 since this task may process multiple views)
    """

    if not settings.long_term_memory:
        # If long-term memory is entirely disabled, there may still be
        # working-memory backed views later, but for now we bail out.
        return

    views = await list_summary_views()
    for view in views:
        if not view.continuous:
            continue
        await refresh_summary_view(view.id, task_id=None)
