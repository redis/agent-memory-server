from __future__ import annotations

import asyncio
from typing import Any

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class Mem0MemoryStore(AnsweringStore):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise missing_dependency("mem0ai", "mem0", "Mem0", exc) from exc
        self._memory = Memory.from_config(config or {})
        self._user_id = user_id
        self._search_limit = search_limit

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for session in sessions:
            messages = []
            if session.label:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Conversation date: {session.label}",
                    }
                )
            messages.extend(
                {"role": normalize_role(item.speaker), "content": item.text}
                for item in session.messages
                if item.text.strip()
            )
            await asyncio.to_thread(
                self._memory.add,
                messages,
                user_id=self._user_id,
                metadata={"date": session.label},
            )

    @staticmethod
    def _items(raw: Any) -> list[Any]:
        if isinstance(raw, dict):
            return raw.get("results", raw.get("memories", []))
        return raw

    @staticmethod
    def _memory_text(item: Any) -> str:
        if not isinstance(item, dict):
            return str(item)
        text = str(item.get("memory", item))
        metadata = item.get("metadata") or {}
        date = metadata.get("date") if isinstance(metadata, dict) else None
        if date and f"Conversation date: {date}" not in text:
            return f"Conversation date: {date}\n{text}"
        return text

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        raw = await asyncio.to_thread(
            self._memory.search,
            question,
            filters={"user_id": self._user_id},
            top_k=self._search_limit,
        )
        context = "\n".join(self._memory_text(item) for item in self._items(raw))
        return await self._answer(context, question, question_date)

    async def list_memories(self) -> list[str]:
        raw = await asyncio.to_thread(
            self._memory.get_all, filters={"user_id": self._user_id}
        )
        return [self._memory_text(item) for item in self._items(raw)]

    async def reset(self) -> None:
        await asyncio.to_thread(self._memory.delete_all, user_id=self._user_id)
        self._token_usage = TokenUsage()
