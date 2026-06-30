"""Migration helpers for Redis Cloud Agent Memory Service exports.

Redis Cloud Agent Memory Service stores long-term memories with a service/store
scoped key layout like::

    memory:<store_id>:ltm:<memory_id>

and fields such as ``id``, ``owner_id`` and ``text_vector``. The local
agent-memory-server RedisVL backend stores compatible records under the local
RedisVL index prefix, usually::

    memory_idx:<memory_id>

with canonical fields ``id_``, ``user_id`` and ``vector``.

This module copies cloud-exported hashes into the local schema without deleting
source keys. It is intentionally conservative: dry-run is the default at the CLI
layer, existing target keys are skipped unless overwrite is explicitly enabled,
and malformed source records are counted rather than partially migrated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

from redis.asyncio import Redis

from agent_memory_server.config import settings
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.tag_codec import decode_tag_values, encode_tag_values


logger = logging.getLogger(__name__)


@dataclass
class CloudLongTermMemoryMigrationStats:
    """Counters from a cloud long-term-memory schema migration."""

    scanned: int = 0
    eligible: int = 0
    migrated: int = 0
    skipped_existing: int = 0
    skipped_missing_id: int = 0
    skipped_missing_text: int = 0
    skipped_missing_vector: int = 0
    failed: int = 0
    sample_source_keys: list[str] = field(default_factory=list)
    sample_target_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "eligible": self.eligible,
            "migrated": self.migrated,
            "skipped_existing": self.skipped_existing,
            "skipped_missing_id": self.skipped_missing_id,
            "skipped_missing_text": self.skipped_missing_text,
            "skipped_missing_vector": self.skipped_missing_vector,
            "failed": self.failed,
            "sample_source_keys": self.sample_source_keys,
            "sample_target_keys": self.sample_target_keys,
        }


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _coerce_timestamp(value: Any) -> float | None:
    """Convert Redis Cloud timestamps to local AMS seconds.

    Cloud exports observed in Redis store millisecond epoch values. Local
    agent-memory-server stores seconds. If a value is already seconds, preserve
    it. Empty/unparseable values return None so callers can omit the field.
    """

    if value in (None, b"", ""):
        return None
    try:
        number = float(_decode(value))
    except (TypeError, ValueError):
        return None

    # 1e11 is safely above current epoch seconds and below epoch millis.
    if number > 100_000_000_000:
        number = number / 1000.0
    return number


def _coerce_vector(value: Any, *, dimensions: int | None = None) -> bytes | None:
    """Convert Cloud vector bytes to local RedisVL float32 bytes.

    Redis Cloud Agent Memory exports observed locally store 1536-dimensional
    vectors as float64 blobs (12288 bytes). Local agent-memory-server's RedisVL
    schema expects FLOAT32 blobs (1536 * 4 = 6144 bytes). If the vector is
    already float32-sized, preserve it. If it is float64-sized, downcast little
    endian doubles to little endian floats.
    """

    if not isinstance(value, bytes) or not value:
        return None

    dims = dimensions or int(settings.redisvl_vector_dimensions)
    float32_bytes = dims * 4
    float64_bytes = dims * 8

    if len(value) == float32_bytes:
        return value
    if len(value) != float64_bytes:
        return None

    floats = struct.unpack(f"<{dims}d", value)
    return struct.pack(f"<{dims}f", *floats)


def cloud_hash_to_local_hash(source: dict[Any, Any]) -> dict[str, Any] | None:
    """Map one Redis Cloud Agent Memory hash to local AMS RedisVL fields.

    Returns None when mandatory source fields are missing. The vector bytes are
    copied as-is; no embedding provider/API call is required.
    """

    text = _decode(source.get(b"text", source.get("text"))).strip()
    memory_id = _decode(source.get(b"id", source.get("id"))).strip()
    vector = _coerce_vector(source.get(b"text_vector", source.get("text_vector")))

    if not memory_id or not text or vector is None:
        return None

    user_id = _decode(source.get(b"user_id", source.get("user_id"))).strip()
    if not user_id:
        user_id = _decode(source.get(b"owner_id", source.get("owner_id"))).strip()

    created_at = _coerce_timestamp(source.get(b"created_at", source.get("created_at")))
    updated_at = _coerce_timestamp(source.get(b"updated_at", source.get("updated_at")))
    last_accessed = _coerce_timestamp(
        source.get(b"last_accessed", source.get("last_accessed"))
    )
    if last_accessed is None:
        last_accessed = updated_at or created_at

    memory_type = _decode(
        source.get(b"memory_type", source.get("memory_type", "episodic"))
    ).strip() or "episodic"

    target: dict[str, Any] = {
        "id_": memory_id,
        "text": text,
        "vector": vector,
        "session_id": _decode(source.get(b"session_id", source.get("session_id"))),
        "user_id": user_id,
        "namespace": _decode(source.get(b"namespace", source.get("namespace"))),
        "memory_type": memory_type,
        "topics": encode_tag_values(
            decode_tag_values(_decode(source.get(b"topics", source.get("topics"))))
        ),
        "entities": encode_tag_values(
            decode_tag_values(_decode(source.get(b"entities", source.get("entities"))))
        ),
        "extracted_from": encode_tag_values(
            decode_tag_values(
                _decode(source.get(b"extracted_from", source.get("extracted_from")))
            )
        ),
        "memory_hash": _decode(
            source.get(b"memory_hash", source.get("memory_hash"))
        ),
        "discrete_memory_extracted": _decode(
            source.get(
                b"discrete_memory_extracted",
                source.get("discrete_memory_extracted", "f"),
            )
        )
        or "f",
        "pinned": int(_decode(source.get(b"pinned", source.get("pinned", 0))) or 0),
        "access_count": int(
            _decode(source.get(b"access_count", source.get("access_count", 0))) or 0
        ),
    }

    if not target["memory_hash"]:
        content_fields = {
            "text": target["text"],
            "user_id": target["user_id"],
            "session_id": target["session_id"],
            "namespace": target["namespace"],
            "memory_type": target["memory_type"],
        }
        target["memory_hash"] = hashlib.sha256(
            json.dumps(content_fields, sort_keys=True).encode()
        ).hexdigest()

    if created_at is not None:
        target["created_at"] = created_at
    if updated_at is not None:
        target["updated_at"] = updated_at
    if last_accessed is not None:
        target["last_accessed"] = last_accessed

    persisted_at = _coerce_timestamp(
        source.get(b"persisted_at", source.get("persisted_at"))
    )
    if persisted_at is not None:
        target["persisted_at"] = persisted_at

    event_date = _coerce_timestamp(source.get(b"event_date", source.get("event_date")))
    if event_date is not None:
        target["event_date"] = event_date

    return target


async def migrate_cloud_long_term_memory(
    redis: Any,
    *,
    store_id: str | None = None,
    source_pattern: str | None = None,
    target_prefix: str | None = None,
    batch_size: int = 500,
    dry_run: bool = True,
    overwrite: bool = False,
) -> CloudLongTermMemoryMigrationStats:
    """Copy Redis Cloud Agent Memory Service hashes into local AMS schema."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if source_pattern is None:
        source_pattern = f"memory:{store_id}:ltm:*" if store_id else "memory:*:ltm:*"

    target_prefix = target_prefix or settings.redisvl_index_prefix
    stats = CloudLongTermMemoryMigrationStats()
    cursor = 0

    while True:
        cursor, keys = await redis.scan(
            cursor=cursor, match=source_pattern, count=batch_size
        )
        if not keys and cursor == 0:
            break

        for key in keys:
            key_str = _decode(key)
            stats.scanned += 1
            if len(stats.sample_source_keys) < 5:
                stats.sample_source_keys.append(key_str)

            source = await redis.hgetall(key)
            memory_id = _decode(source.get(b"id", source.get("id"))).strip()
            text = _decode(source.get(b"text", source.get("text"))).strip()
            vector = _coerce_vector(source.get(b"text_vector", source.get("text_vector")))

            if not memory_id:
                stats.skipped_missing_id += 1
                continue
            if not text:
                stats.skipped_missing_text += 1
                continue
            if vector is None:
                stats.skipped_missing_vector += 1
                continue

            target_key = f"{target_prefix}:{memory_id}"
            if len(stats.sample_target_keys) < 5:
                stats.sample_target_keys.append(target_key)

            if not overwrite and await redis.exists(target_key):
                stats.skipped_existing += 1
                continue

            stats.eligible += 1
            target = cloud_hash_to_local_hash(source)
            if target is None:
                stats.failed += 1
                continue

            if not dry_run:
                try:
                    await redis.hset(target_key, mapping=target)
                except Exception:
                    stats.failed += 1
                    logger.exception("Failed to migrate %s to %s", key_str, target_key)
                    continue

            stats.migrated += 1

        if cursor == 0:
            break

    return stats


async def count_local_long_term_memory(redis: Redis) -> int:
    """Count local AMS canonical long-term-memory keys for the current prefix."""

    count = 0
    cursor = 0
    pattern = Keys.memory_key("*")
    while True:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
        count += len(keys)
        if cursor == 0:
            break
    return count
