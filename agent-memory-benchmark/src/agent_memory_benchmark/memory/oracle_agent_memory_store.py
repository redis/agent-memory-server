from __future__ import annotations

import asyncio
import os
from typing import Any

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class OracleAgentMemoryStore(AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        embedder_model: str = "text-embedding-3-small",
        max_search_results: int = 10,
        connection: Any = None,
    ) -> None:
        super().__init__(model=model)
        try:
            import oracledb
            from oracleagentmemory.core import OracleAgentMemory
            from oracleagentmemory.core.embedders.embedder import Embedder
            from oracleagentmemory.core.llms.llm import Llm
        except ImportError as exc:
            raise missing_dependency(
                "oracleagentmemory", "oracle", "Oracle Agent Memory", exc
            ) from exc
        if connection is None:
            password = os.environ.get("ORACLE_MEMORY_DB_PASSWORD")
            if not password:
                raise RuntimeError(
                    "Oracle Agent Memory requires connection= or "
                    "ORACLE_MEMORY_DB_PASSWORD"
                )
            connection = oracledb.create_pool(
                user=os.environ.get("ORACLE_MEMORY_DB_USER"),
                password=password,
                dsn=os.environ.get("ORACLE_MEMORY_DB_CONNECT_STRING"),
            )
        self._connection = connection
        self._memory = OracleAgentMemory(
            connection=connection,
            llm=Llm(model=model),
            embedder=Embedder(model=embedder_model),
        )
        self._user_id = user_id
        self._max_search_results = max_search_results
        self._thread: Any = None
        self._thread_id: str | None = None

    async def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if close:
            await asyncio.to_thread(close)

    async def ingest(self, sessions: list[SessionLike]) -> None:
        from oracleagentmemory.apis.thread import Message

        try:
            self._thread = await asyncio.to_thread(
                self._memory.create_thread,
                thread_id=self._user_id,
                user_id=self._user_id,
            )
        except TypeError:
            self._thread = await asyncio.to_thread(
                self._memory.create_thread, user_id=self._user_id
            )
        self._thread_id = (
            getattr(self._thread, "thread_id", None)
            or getattr(self._thread, "id", None)
            or self._user_id
        )
        for session in sessions:
            messages = []
            if session.label:
                messages.append(
                    Message(
                        role="user",
                        content=f"Conversation date: {session.label}",
                        timestamp=(session.date.isoformat() if session.date else None),
                    )
                )
            messages.extend(
                Message(
                    role=normalize_role(item.speaker),
                    content=item.text,
                    timestamp=session.date.isoformat() if session.date else None,
                )
                for item in session.messages
                if item.text.strip()
            )
            await asyncio.to_thread(self._thread.add_messages, messages)

    async def _search(self, query: str, limit: int) -> list[str]:
        from oracleagentmemory.apis.searchscope import SearchScope

        results = await asyncio.to_thread(
            self._memory.search,
            query,
            scope=SearchScope(user_id=self._user_id),
            max_results=limit,
        )
        return [str(getattr(item, "content", item)) for item in results]

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        return await self._answer(
            "\n".join(await self._search(question, self._max_search_results)),
            question,
            question_date,
        )

    async def list_memories(self) -> list[str]:
        getter = getattr(self._memory, "get_all", None)
        if not getter:
            return await self._search("", 100)
        try:
            results = await asyncio.to_thread(getter, user_id=self._user_id)
        except TypeError:
            from oracleagentmemory.apis.searchscope import SearchScope

            results = await asyncio.to_thread(
                getter, scope=SearchScope(user_id=self._user_id)
            )
        return [str(getattr(item, "content", item)) for item in results or []]

    async def reset(self) -> None:
        thread_id = self._thread_id or getattr(self._thread, "thread_id", None)
        if thread_id:
            await asyncio.to_thread(self._memory.delete_thread, str(thread_id))
        delete_user = getattr(self._memory, "delete_user", None)
        if delete_user:
            try:
                await asyncio.to_thread(delete_user, self._user_id, cascade=True)
            except TypeError:
                await asyncio.to_thread(delete_user, self._user_id)
        self._thread = None
        self._thread_id = None
        self._token_usage = TokenUsage()
