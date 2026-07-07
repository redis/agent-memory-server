import struct

import pytest

from agent_memory_server.redis_cloud_migration import (
    _coerce_timestamp,
    _coerce_vector,
    cloud_hash_to_local_hash,
)


def test_cloud_hash_to_local_hash_maps_cloud_fields_to_local_schema():
    cloud_vector = struct.pack("<1536d", 0.5, -0.25, *([0.0] * 1534))
    source = {
        b"id": b"abc123",
        b"text": b"remember the important thing",
        b"text_vector": cloud_vector,
        b"owner_id": b"john",
        b"session_id": b"session-1",
        b"namespace": b"hermes-dev-default",
        b"topics": b"alpha|beta",
        b"created_at": b"1780000803492",
        b"updated_at": b"1780000805492",
        b"memory_type": b"episodic",
    }

    target = cloud_hash_to_local_hash(source)

    assert target is not None
    assert target["id_"] == "abc123"
    assert target["text"] == "remember the important thing"
    assert len(target["vector"]) == 1536 * 4
    assert struct.unpack("<2f", target["vector"][:8]) == pytest.approx((0.5, -0.25))
    assert target["user_id"] == "john"
    assert target["session_id"] == "session-1"
    assert target["namespace"] == "hermes-dev-default"
    assert target["topics"] == "alpha,beta"
    assert target["memory_type"] == "episodic"
    assert target["discrete_memory_extracted"] == "f"
    assert target["pinned"] == 0
    assert target["access_count"] == 0
    assert target["created_at"] == pytest.approx(1780000803.492)
    assert target["updated_at"] == pytest.approx(1780000805.492)
    assert target["last_accessed"] == pytest.approx(1780000805.492)
    assert len(target["memory_hash"]) == 64


def test_cloud_hash_to_local_hash_requires_id_text_and_vector():
    base = {b"id": b"abc123", b"text": b"memory", b"text_vector": b"vector"}

    for missing in [b"id", b"text", b"text_vector"]:
        source = dict(base)
        source.pop(missing)
        assert cloud_hash_to_local_hash(source) is None


def test_coerce_vector_preserves_float32_and_downcasts_float64():
    float32_vector = struct.pack("<2f", 1.0, -2.0)
    float64_vector = struct.pack("<2d", 1.0, -2.0)

    assert _coerce_vector(float32_vector, dimensions=2) == float32_vector
    assert _coerce_vector(float64_vector, dimensions=2) == float32_vector
    assert _coerce_vector(b"bad", dimensions=2) is None


def test_coerce_timestamp_preserves_seconds_and_converts_milliseconds():
    assert _coerce_timestamp("1780000803.492") == pytest.approx(1780000803.492)
    assert _coerce_timestamp("1780000803492") == pytest.approx(1780000803.492)
    assert _coerce_timestamp("") is None
    assert _coerce_timestamp("not-a-number") is None
