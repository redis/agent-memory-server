from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx

from agent_memory_benchmark.memory.base import QueryResult, SessionLike, TokenUsage
from agent_memory_benchmark.memory.common import AnsweringStore


def _api_id(value: str, *, max_length: int = 48) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
    if normalized and len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    prefix = normalized[: max_length - len(digest) - 1].rstrip("-")
    return f"{prefix}-{digest}" if prefix else digest


class RedisAgentMemoryStore(AnsweringStore):
    """Redis Agent Memory adapter using the public Iris data-plane REST API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        store_id: str | None = None,
        api_key: str | None = None,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
        request_timeout: float = 120,
        extraction_timeout: float = 1800,
        extraction_poll_interval: float = 30,
        extraction_stable_seconds: float = 120,
    ) -> None:
        super().__init__(model=model)
        self._base_url = (
            base_url or os.environ.get("REDIS_AGENT_MEMORY_URL", "")
        ).rstrip("/")
        self._store_id = store_id or os.environ.get("REDIS_AGENT_MEMORY_STORE_ID", "")
        key = api_key or os.environ.get("REDIS_AGENT_MEMORY_API_KEY", "")
        missing = [
            name
            for name, value in (
                ("REDIS_AGENT_MEMORY_URL", self._base_url),
                ("REDIS_AGENT_MEMORY_STORE_ID", self._store_id),
                ("REDIS_AGENT_MEMORY_API_KEY", key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("Redis Agent Memory requires " + ", ".join(missing))
        self._owner_id = _api_id(user_id)
        self._search_limit = search_limit
        self._extraction_timeout = extraction_timeout
        self._extraction_poll_interval = extraction_poll_interval
        self._extraction_stable_seconds = extraction_stable_seconds
        self._session_ids: list[str] = []
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=request_timeout,
        )

    @property
    def _store_path(self) -> str:
        return f"/v1/stores/{quote(self._store_id, safe='')}"

    async def close(self) -> None:
        await self._client.aclose()

    async def _search(
        self, text: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        page_token: str | None = None
        remaining = limit
        for _ in range(50):
            page_size = 100 if remaining is None else min(remaining, 100)
            body: dict[str, Any] = {
                "filter": {"ownerId": {"eq": self._owner_id}},
                "limit": page_size,
            }
            if text.strip():
                body["text"] = text
            if page_token:
                body["pageToken"] = page_token
            response = await self._client.post(
                f"{self._store_path}/long-term-memory/search",
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("items") or payload.get("memories") or []
            collected.extend(batch)
            page_token = payload.get("nextPageToken") or payload.get("next_page_token")
            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break
            if not page_token or not batch:
                break
        return collected[:limit] if limit is not None else collected

    @staticmethod
    def _texts(items: list[dict[str, Any]]) -> list[str]:
        return [
            str(item.get("text") or item.get("content") or "")
            for item in items
            if item.get("text") or item.get("content")
        ]

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for session_index, session in enumerate(sessions):
            session_id = f"{self._owner_id}-s{session_index}"
            self._session_ids.append(session_id)
            created_at = session.date or datetime.now(timezone.utc)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            events: list[tuple[str, str]] = []
            if session.label:
                events.append(("USER", f"Conversation date: {session.label}"))
            events.extend(
                (
                    "USER"
                    if message.speaker.lower().strip()
                    in {"user", "human", "environment", "tool"}
                    else "ASSISTANT",
                    message.text.strip(),
                )
                for message in session.messages
                if message.text.strip()
            )
            for offset, (role, text) in enumerate(events):
                response = await self._client.post(
                    f"{self._store_path}/session-memory/events",
                    json={
                        "sessionId": session_id,
                        "actorId": self._owner_id,
                        "role": role,
                        "content": [{"text": text[:65_536]}],
                        "createdAt": (
                            created_at + timedelta(seconds=offset)
                        ).isoformat(),
                    },
                )
                response.raise_for_status()

    async def wait_for_extraction(
        self, *, timeout: float = 120, poll_interval: float = 2
    ) -> list[str]:
        effective_timeout = timeout if timeout != 120 else self._extraction_timeout
        effective_poll = (
            poll_interval if poll_interval != 2 else self._extraction_poll_interval
        )
        deadline = time.monotonic() + effective_timeout
        last_count: int | None = None
        last_change = time.monotonic()
        latest: list[str] = []
        while time.monotonic() < deadline:
            latest = await self.list_memories()
            count = len(latest)
            now = time.monotonic()
            if count != last_count:
                last_count = count
                last_change = now
            elif count and now - last_change >= self._extraction_stable_seconds:
                return latest
            await asyncio.sleep(effective_poll)
        raise TimeoutError(
            f"Memory extraction did not stabilize within {effective_timeout} seconds"
        )

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        started = time.perf_counter()
        items = await self._search(question, self._search_limit)
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)
        result = await self._answer(
            "\n".join(self._texts(items)), question, question_date
        )
        result.retrieval_latency_ms = retrieval_ms
        return result

    async def list_memories(self) -> list[str]:
        return self._texts(await self._search(""))

    async def reset(self) -> None:
        await self._delete_sessions()
        await self._delete_memories()
        self._session_ids.clear()
        self._token_usage = TokenUsage()

    async def _delete_sessions(self) -> None:
        session_ids = set(self._session_ids)
        page_token: str | None = None
        for _ in range(50):
            params: dict[str, str] = {
                "filterOwnerId": self._owner_id,
                "limit": "1000",
            }
            if page_token:
                params["pageToken"] = page_token
            response = await self._client.get(
                f"{self._store_path}/session-memory",
                params=params,
            )
            if response.status_code == 404:
                break
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("items") or payload.get("sessions") or []:
                session_id = item if isinstance(item, str) else item.get("sessionId")
                if session_id:
                    session_ids.add(session_id)
            page_token = payload.get("nextPageToken") or payload.get("next_page_token")
            if not page_token:
                break
        for session_id in session_ids:
            response = await self._client.delete(
                f"{self._store_path}/session-memory/{quote(session_id, safe='')}"
            )
            if response.status_code != 404:
                response.raise_for_status()

    async def _delete_memories(self) -> None:
        previous: set[str] | None = None
        for _ in range(10):
            items = await self._search("")
            memory_ids = [item.get("id") for item in items if item.get("id")]
            if not memory_ids:
                return
            current = set(memory_ids)
            if previous is not None and current == previous:
                return
            previous = current
            response = await self._client.request(
                "DELETE",
                f"{self._store_path}/long-term-memory",
                json={"memoryIds": memory_ids},
            )
            if response.status_code == 404:
                return
            response.raise_for_status()

    def get_store_metadata(self) -> dict[str, Any]:
        return {
            **super().get_store_metadata(),
            "endpoint": self._base_url,
            "store_id": self._store_id,
            "search_limit": self._search_limit,
        }
