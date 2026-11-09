from __future__ import annotations

from typing import Any

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class LangMemStore(AnsweringStore):
    """LangMem manager plus LangGraph's provider-supported indexed store."""

    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            from langgraph.store.memory import InMemoryStore
            from langmem import create_memory_manager
        except ImportError as exc:
            raise missing_dependency("langmem", "langmem", "LangMem", exc) from exc
        self._manager = create_memory_manager(f"openai:{model}")
        self._store = InMemoryStore(
            index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        )
        self._namespace = ("memories", user_id)
        self._search_limit = search_limit
        self._existing: list[Any] = []
        self._keys: list[str] = []

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for session in sessions:
            messages = []
            if session.label:
                messages.append(
                    {"role": "user", "content": f"Conversation date: {session.label}"}
                )
            messages.extend(
                {"role": normalize_role(item.speaker), "content": item.text}
                for item in session.messages
                if item.text.strip()
            )
            self._existing = await self._manager.ainvoke(
                {"messages": messages, "existing": self._existing}
            )
        self._keys = []
        for index, memory in enumerate(self._existing):
            key = str(index)
            value = memory[1] if isinstance(memory, tuple) else memory
            text = getattr(value, "content", value)
            self._store.put(self._namespace, key, {"text": str(text)})
            self._keys.append(key)

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        results = self._store.search(
            self._namespace, query=question, limit=self._search_limit
        )
        context = "\n".join(str(item.value.get("text", "")) for item in results)
        return await self._answer(context, question, question_date)

    async def list_memories(self) -> list[str]:
        texts: list[str] = []
        for key in self._keys:
            item = self._store.get(self._namespace, key)
            if item is None:
                continue
            value = getattr(item, "value", item)
            texts.append(
                str(value.get("text", "") if isinstance(value, dict) else value)
            )
        return texts

    async def reset(self) -> None:
        for key in self._keys:
            self._store.delete(self._namespace, key)
        self._keys.clear()
        self._existing.clear()
        self._token_usage = TokenUsage()
