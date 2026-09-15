from __future__ import annotations

import asyncio
import os

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class VertexMemoryBankStore(AnsweringStore):
    """Vertex Memory Bank adapter for an existing Agent Engine resource."""

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        agent_engine_name: str,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            import vertexai
        except ImportError as exc:
            raise missing_dependency(
                "google-cloud-aiplatform", "google", "Vertex Memory Bank", exc
            ) from exc
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "Vertex Memory Bank requires project= or GOOGLE_CLOUD_PROJECT"
            )
        self._client = vertexai.Client(
            project=project,
            location=location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        self._engine = agent_engine_name
        self._scope = {"user_id": user_id}
        self._search_limit = search_limit

    async def ingest(self, sessions: list[SessionLike]) -> None:
        memories = self._client.agent_engines.memories
        for session in sessions:
            events = []
            if session.label:
                events.append(
                    {
                        "content": {
                            "role": "user",
                            "parts": [{"text": f"Conversation date: {session.label}"}],
                        }
                    }
                )
            events.extend(
                {
                    "content": {
                        "role": normalize_role(item.speaker),
                        "parts": [{"text": item.text}],
                    }
                }
                for item in session.messages
                if item.text.strip()
            )
            await asyncio.to_thread(
                memories.generate,
                name=self._engine,
                direct_contents_source={"events": events},
                scope=self._scope,
                config={"wait_for_completion": True},
            )

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        results = await asyncio.to_thread(
            lambda: list(
                self._client.agent_engines.memories.retrieve(
                    name=self._engine,
                    scope=self._scope,
                    similarity_search_params={
                        "search_query": question,
                        "top_k": self._search_limit,
                    },
                )
            )
        )
        context = "\n".join(
            str(getattr(getattr(item, "memory", item), "fact", "")) for item in results
        )
        return await self._answer(context, question, question_date)

    async def _scoped(self, **kwargs: object) -> list[object]:
        return await asyncio.to_thread(
            lambda: list(
                self._client.agent_engines.memories.retrieve(
                    name=self._engine,
                    scope=self._scope,
                    **kwargs,
                )
            )
        )

    async def list_memories(self) -> list[str]:
        texts: list[str] = []
        for item in await self._scoped():
            fact = getattr(getattr(item, "memory", item), "fact", "")
            if fact:
                texts.append(str(fact))
        return texts

    async def reset(self) -> None:
        memories = self._client.agent_engines.memories
        purge = getattr(memories, "purge", None)
        if purge:
            kwargs: dict[str, object] = {
                "name": self._engine,
                "filter": f'scope.user_id="{self._scope["user_id"]}"',
                "force": True,
                "config": {"wait_for_completion": True},
            }
            try:
                await asyncio.to_thread(purge, **kwargs)
            except TypeError:
                kwargs.pop("config", None)
                await asyncio.to_thread(purge, **kwargs)
        else:
            for memory in await asyncio.to_thread(
                lambda: list(memories.list(name=self._engine))
            ):
                if _same_scope(getattr(memory, "scope", None), self._scope):
                    await asyncio.to_thread(memories.delete, name=memory.name)
        self._token_usage = TokenUsage()


def _same_scope(scope: object, expected: dict[str, str]) -> bool:
    if hasattr(scope, "model_dump"):
        scope = scope.model_dump()
    elif hasattr(scope, "to_dict"):
        scope = scope.to_dict()
    elif not isinstance(scope, dict):
        try:
            scope = dict(scope) if scope is not None else {}
        except TypeError:
            return False
    return scope == expected
