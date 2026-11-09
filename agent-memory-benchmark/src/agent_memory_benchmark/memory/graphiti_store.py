from __future__ import annotations

import os
from datetime import datetime, timezone

from agent_memory_benchmark.memory.base import QueryResult, SessionLike, TokenUsage
from agent_memory_benchmark.memory.common import (
    AnsweringStore,
    missing_dependency,
    session_text,
)


class GraphitiStore(AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
    ) -> None:
        super().__init__(model=model)
        try:
            from graphiti_core import Graphiti
        except ImportError as exc:
            raise missing_dependency(
                "graphiti-core", "graphiti", "Graphiti", exc
            ) from exc
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("Graphiti requires NEO4J_PASSWORD or neo4j_password=")
        self._graph = Graphiti(
            uri=neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=neo4j_user or os.environ.get("NEO4J_USER", "neo4j"),
            password=password,
        )
        self._group_id = user_id
        self._search_limit = search_limit

    async def __aenter__(self) -> GraphitiStore:
        await self._graph.build_indices_and_constraints()
        return self

    async def close(self) -> None:
        await self._graph.close()

    async def ingest(self, sessions: list[SessionLike]) -> None:
        from graphiti_core.nodes import EpisodeType

        for index, session in enumerate(sessions):
            reference_time = session.date or datetime.now(timezone.utc)
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
            await self._graph.add_episode(
                name=f"session-{index}",
                episode_body=session_text(session),
                source=EpisodeType.message,
                source_description="conversation session",
                reference_time=reference_time,
                group_id=self._group_id,
            )

    async def _facts(self, query: str, limit: int) -> list[str]:
        edges = await self._graph.search(
            query=query,
            group_ids=[self._group_id],
            num_results=limit,
        )
        return [getattr(edge, "fact", str(edge)) for edge in edges or []]

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        return await self._answer(
            "\n".join(await self._facts(question, self._search_limit)),
            question,
            question_date,
        )

    async def list_memories(self) -> list[str]:
        try:
            from graphiti_core.edges import EntityEdge
            from graphiti_core.errors import GroupsEdgesNotFoundError
        except ImportError:
            return await self._facts("*", 50)
        getter = getattr(EntityEdge, "get_by_group_ids", None)
        if not getter:
            return await self._facts("*", 50)
        try:
            edges = await getter(self._graph.driver, [self._group_id])
        except GroupsEdgesNotFoundError:
            return []
        return [getattr(edge, "fact", str(edge)) for edge in edges or []]

    async def reset(self) -> None:
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        await clear_data(self._graph.driver, group_ids=[self._group_id])
        self._token_usage = TokenUsage()
