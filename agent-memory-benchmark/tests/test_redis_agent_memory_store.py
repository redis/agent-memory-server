from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

import agent_memory_benchmark.memory.common as common
from agent_memory_benchmark.datasets.models import ContextMessage, Session
from agent_memory_benchmark.memory import QueryResult
from agent_memory_benchmark.memory.redis_agent_memory_store import (
    RedisAgentMemoryStore,
)


@pytest.mark.asyncio
async def test_redis_agent_memory_uses_public_rest_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    search_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal search_calls
        requests.append(request)
        path = request.url.path
        if request.method == "GET" and path.endswith("/session-memory"):
            return httpx.Response(200, json={"items": ["benchmark-run-s0"]})
        if request.method == "POST" and path.endswith("/session-memory/events"):
            return httpx.Response(201, json={"event": {}})
        if request.method == "POST" and path.endswith("/long-term-memory/search"):
            body = json.loads(request.content)
            if body.get("text") == "What does the user like?":
                return httpx.Response(
                    200, json={"items": [{"id": "m1", "text": "Likes Redis"}]}
                )
            search_calls += 1
            items = [{"id": "m1", "text": "Likes Redis"}] if search_calls == 1 else []
            return httpx.Response(200, json={"items": items})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": ["m1"]})
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    async def fake_generate(
        context: str,
        question: str,
        *,
        model: str,
        question_date: str | None,
    ) -> tuple[QueryResult, dict[str, int]]:
        assert context == "Likes Redis"
        return (
            QueryResult(answer="Redis", prompt=[]),
            {"prompt_tokens": 2, "completion_tokens": 1},
        )

    monkeypatch.setattr(common, "generate_answer", fake_generate)
    store = RedisAgentMemoryStore(
        base_url="https://memory.example",
        store_id="store/one",
        api_key="secret",
        user_id="benchmark-run",
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url="https://memory.example",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    )

    await store.ingest(
        [
            Session(
                label="2026-09-01",
                date=datetime(2026, 9, 1, tzinfo=timezone.utc),
                messages=[
                    ContextMessage("user", "I like Redis"),
                    ContextMessage("assistant", "Noted"),
                ],
            )
        ]
    )
    result = await store.query("What does the user like?")
    await store.reset()
    await store.close()

    assert result.answer == "Redis"
    event_requests = [
        request
        for request in requests
        if request.url.path.endswith("/session-memory/events")
    ]
    assert len(event_requests) == 3
    first_event = json.loads(event_requests[0].content)
    assert first_event == {
        "sessionId": "benchmark-run-s0",
        "actorId": "benchmark-run",
        "role": "USER",
        "content": [{"text": "Conversation date: 2026-09-01"}],
        "createdAt": "2026-09-01T00:00:00+00:00",
    }
    assert all(
        request.headers["authorization"] == "Bearer secret" for request in requests
    )
    assert any(
        request.method == "DELETE" and request.url.path.endswith("/long-term-memory")
        for request in requests
    )


@pytest.mark.asyncio
async def test_redis_agent_memory_reset_stops_when_search_does_not_shrink() -> None:
    searches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal searches
        if request.method == "GET" and request.url.path.endswith("/session-memory"):
            return httpx.Response(200, json={"items": []})
        if request.method == "POST" and request.url.path.endswith(
            "/long-term-memory/search"
        ):
            searches += 1
            return httpx.Response(200, json={"items": [{"id": "stuck"}]})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": []})
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    store = RedisAgentMemoryStore(
        base_url="https://memory.example",
        store_id="store/one",
        api_key="secret",
        user_id="benchmark-run",
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url="https://memory.example",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    )
    await store.reset()
    await store.close()
    assert searches == 2


@pytest.mark.asyncio
async def test_redis_agent_memory_reset_raises_when_memory_delete_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/session-memory"):
            return httpx.Response(200, json={"items": []})
        if request.method == "POST" and request.url.path.endswith(
            "/long-term-memory/search"
        ):
            return httpx.Response(200, json={"items": [{"id": "m1", "text": "left"}]})
        if request.method == "DELETE" and request.url.path.endswith(
            "/long-term-memory"
        ):
            return httpx.Response(500, json={"error": "failed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    store = RedisAgentMemoryStore(
        base_url="https://memory.example",
        store_id="store/one",
        api_key="secret",
        user_id="benchmark-run",
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url="https://memory.example",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await store.reset()
    await store.close()


@pytest.mark.asyncio
async def test_redis_agent_memory_reset_lists_owner_sessions_by_page() -> None:
    listed: list[str | None] = []
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/session-memory"):
            listed.append(request.url.params.get("filterOwnerId"))
            listed.append(request.url.params.get("pageToken"))
            if request.url.params.get("pageToken") is None:
                return httpx.Response(
                    200,
                    json={
                        "items": ["benchmark-run-s0"],
                        "nextPageToken": "page-2",
                    },
                )
            return httpx.Response(200, json={"items": ["benchmark-run-s1"]})
        if request.method == "POST" and request.url.path.endswith(
            "/long-term-memory/search"
        ):
            return httpx.Response(200, json={"items": []})
        if request.method == "DELETE" and "/session-memory/" in request.url.path:
            deleted.append(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": []})
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    store = RedisAgentMemoryStore(
        base_url="https://memory.example",
        store_id="store/one",
        api_key="secret",
        user_id="benchmark-run",
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url="https://memory.example",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    )
    await store.reset()
    await store.close()
    assert listed == ["benchmark-run", None, "benchmark-run", "page-2"]
    assert set(deleted) == {"benchmark-run-s0", "benchmark-run-s1"}


@pytest.mark.asyncio
async def test_redis_agent_memory_lists_all_search_pages() -> None:
    tokens: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tokens.append(body.get("pageToken"))
        if body.get("pageToken") is None:
            return httpx.Response(
                200,
                json={
                    "items": [{"id": "m1", "text": "one"}],
                    "nextPageToken": "page-2",
                },
            )
        return httpx.Response(200, json={"items": [{"id": "m2", "text": "two"}]})

    store = RedisAgentMemoryStore(
        base_url="https://memory.example",
        store_id="store/one",
        api_key="secret",
        user_id="benchmark-run",
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url="https://memory.example",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    )
    memories = await store.list_memories()
    await store.close()
    assert memories == ["one", "two"]
    assert tokens == [None, "page-2"]


def test_redis_agent_memory_requires_connection_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "REDIS_AGENT_MEMORY_URL",
        "REDIS_AGENT_MEMORY_STORE_ID",
        "REDIS_AGENT_MEMORY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="REDIS_AGENT_MEMORY_URL"):
        RedisAgentMemoryStore()
