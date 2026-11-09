from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

import agent_memory_benchmark.memory.common as common
from agent_memory_benchmark.datasets.models import ContextMessage, Session
from agent_memory_benchmark.memory import QueryResult, TokenUsage
from agent_memory_benchmark.memory.bedrock_agentcore_store import (
    BedrockAgentCoreStore,
)
from agent_memory_benchmark.memory.graphiti_store import GraphitiStore
from agent_memory_benchmark.memory.langmem_store import LangMemStore
from agent_memory_benchmark.memory.mem0_store import Mem0MemoryStore
from agent_memory_benchmark.memory.oracle_agent_memory_store import (
    OracleAgentMemoryStore,
)
from agent_memory_benchmark.memory.supermemory_store import SupermemoryStore
from agent_memory_benchmark.memory.vertex_memory_bank import VertexMemoryBankStore
from agent_memory_benchmark.memory.zep_store import ZepMemoryStore


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, module: Any) -> None:
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        dotted = ".".join(parts[:index])
        if dotted not in sys.modules:
            monkeypatch.setitem(sys.modules, dotted, types.ModuleType(dotted))
    monkeypatch.setitem(sys.modules, name, module)


async def _fake_answer(
    context: str,
    question: str,
    *,
    model: str,
    question_date: str | None,
) -> tuple[QueryResult, dict[str, int]]:
    return (
        QueryResult(answer=context or "none", prompt=[]),
        {"prompt_tokens": 1, "completion_tokens": 1},
    )


class _Mem0:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _Mem0:
        return cls()

    def search(self, query: str, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        self.calls.append(("search", (query,), kwargs))
        return {
            "results": [{"memory": "likes coffee", "metadata": {"date": "2026/01/02"}}]
        }

    def get_all(self, **kwargs: Any) -> dict[str, list[dict[str, str]]]:
        self.calls.append(("get_all", (), kwargs))
        return {"results": [{"memory": "likes coffee"}]}

    def add(self, messages: Any, *_args: Any, **kwargs: Any) -> None:
        self.calls.append(("add", (messages,), kwargs))

    def delete_all(self, **kwargs: Any) -> None:
        self.calls.append(("delete_all", (), kwargs))


@pytest.mark.asyncio
async def test_mem0_search_and_list_use_v2_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_module(monkeypatch, "mem0", SimpleNamespace(Memory=_Mem0))
    monkeypatch.setattr(common, "generate_answer", _fake_answer)
    store = Mem0MemoryStore(user_id="u1", search_limit=7)
    await store.ingest(
        [
            Session(
                label="2026/01/02",
                messages=[ContextMessage("user", "I like coffee")],
            )
        ]
    )
    memories = await store.list_memories()
    result = await store.query("drink?")
    assert store._memory.calls[0][0] == "add"
    assert store._memory.calls[0][1][0][0] == {
        "role": "user",
        "content": "Conversation date: 2026/01/02",
    }
    assert memories == ["likes coffee"]
    assert result.answer == "Conversation date: 2026/01/02\nlikes coffee"
    assert store._memory.calls[1] == ("get_all", (), {"filters": {"user_id": "u1"}})
    assert store._memory.calls[2][0] == "search"
    assert store._memory.calls[2][2] == {"filters": {"user_id": "u1"}, "top_k": 7}


@pytest.mark.asyncio
async def test_vertex_uses_memories_subclient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    generate_kwargs: list[dict[str, Any]] = []
    retrieve_kwargs: list[dict[str, Any]] = []
    purge_kwargs: list[dict[str, Any]] = []

    class Memories:
        def generate(self, **kwargs: Any) -> None:
            calls.append("generate")
            generate_kwargs.append(kwargs)

        def retrieve(self, **kwargs: Any) -> list[Any]:
            calls.append("retrieve")
            retrieve_kwargs.append(kwargs)
            return [SimpleNamespace(memory=SimpleNamespace(fact="likes coffee"))]

        def list(self, **_kwargs: Any) -> list[Any]:
            calls.append("list")
            return []

        def purge(self, **kwargs: Any) -> None:
            calls.append("purge")
            purge_kwargs.append(kwargs)

        def delete(self, **_kwargs: Any) -> None:
            calls.append("delete")

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.agent_engines = SimpleNamespace(memories=Memories())

    _install_module(monkeypatch, "vertexai", SimpleNamespace(Client=Client))
    monkeypatch.setattr(common, "generate_answer", _fake_answer)
    store = VertexMemoryBankStore(
        project="p", agent_engine_name="engines/e", user_id="u1"
    )
    await store.ingest(
        [Session(label="now", messages=[ContextMessage("user", "I like coffee")])]
    )
    await store.query("drink?")
    await store.list_memories()
    await store.reset()
    assert calls == ["generate", "retrieve", "retrieve", "purge"]
    events = generate_kwargs[0]["direct_contents_source"]["events"]
    assert events[0]["content"]["parts"][0]["text"] == "Conversation date: now"
    assert retrieve_kwargs[0]["scope"] == {"user_id": "u1"}
    assert retrieve_kwargs[1] == {"name": "engines/e", "scope": {"user_id": "u1"}}
    assert purge_kwargs[0]["filter"] == 'scope.user_id="u1"'
    assert purge_kwargs[0]["force"] is True


@pytest.mark.asyncio
async def test_vertex_reset_matches_non_dict_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[str] = []

    class Memories:
        def list(self, **_kwargs: Any) -> list[Any]:
            return [
                SimpleNamespace(
                    name="projects/p/memories/m1",
                    scope=SimpleNamespace(
                        model_dump=lambda: {"user_id": "u1"},
                    ),
                ),
                SimpleNamespace(
                    name="projects/p/memories/other",
                    scope=SimpleNamespace(
                        model_dump=lambda: {"user_id": "other"},
                    ),
                ),
            ]

        def delete(self, **kwargs: Any) -> None:
            deleted.append(kwargs["name"])

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.agent_engines = SimpleNamespace(memories=Memories())

    _install_module(monkeypatch, "vertexai", SimpleNamespace(Client=Client))
    store = VertexMemoryBankStore(
        project="p", agent_engine_name="engines/e", user_id="u1"
    )
    await store.reset()
    assert deleted == ["projects/p/memories/m1"]


@pytest.mark.asyncio
async def test_bedrock_reset_deletes_remote_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def list_events(self, **kwargs: Any) -> dict[str, list[dict[str, str]]]:
            calls.append(("list_events", kwargs))
            return {"events": [{"eventId": "e1"}]}

        def delete_event(self, **kwargs: Any) -> None:
            calls.append(("delete_event", kwargs))

        def retrieve_memories(self, **kwargs: Any) -> list[dict[str, Any]]:
            if "namespace_path" in kwargs or "namespacePath" in kwargs:
                raise TypeError("unexpected keyword argument 'namespace_path'")
            calls.append(("retrieve_memories", kwargs))
            return [{"content": {"text": "t"}}]

        def list_memory_records(self, **kwargs: Any) -> dict[str, list[dict[str, str]]]:
            if "namespace_path" in kwargs or "namespacePath" in kwargs:
                raise TypeError("unexpected keyword argument 'namespace_path'")
            calls.append(("list_memory_records", kwargs))
            return {
                "memoryRecords": [{"memoryRecordId": "m1", "content": {"text": "t"}}]
            }

        def delete_memory_record(self, **kwargs: Any) -> None:
            calls.append(("delete_memory_record", kwargs))

    _install_module(
        monkeypatch,
        "bedrock_agentcore.memory",
        SimpleNamespace(MemoryClient=Client),
    )
    store = BedrockAgentCoreStore(
        memory_id="mem-1",
        namespace_path="/ns/{user_id}/",
        user_id="u1",
    )
    store._sessions = ["u1-0"]
    monkeypatch.setattr(common, "generate_answer", _fake_answer)
    result = await store.query("drink?")
    memories = await store.list_memories()
    await store.reset()
    assert result.answer == "t"
    assert memories == ["t"]
    assert store._sessions == []
    assert calls[0][1]["namespace"] == "/ns/u1/"
    assert calls[1][1]["namespace"] == "/ns/u1/"
    assert [name for name, _kwargs in calls] == [
        "retrieve_memories",
        "list_memory_records",
        "list_events",
        "delete_event",
        "list_memory_records",
        "delete_memory_record",
    ]
    assert calls[-1] == (
        "delete_memory_record",
        {"memory_id": "mem-1", "memory_record_id": "m1"},
    )


@pytest.mark.asyncio
async def test_zep_reset_ignores_missing_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NotFoundError(Exception):
        status_code = 404

    class Users:
        async def delete(self, _user_id: str) -> None:
            raise NotFoundError("missing")

    class Client:
        def __init__(self) -> None:
            self.user = Users()

        async def close(self) -> None:
            return None

    _install_module(
        monkeypatch,
        "zep_cloud.client",
        SimpleNamespace(AsyncZep=lambda api_key: Client()),
    )
    store = ZepMemoryStore(api_key="k", user_id="missing-user")
    await store.reset()
    assert store._threads == []


@pytest.mark.asyncio
async def test_zep_ingest_stamps_created_at_and_lists_all_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    list_calls: list[dict[str, Any]] = []

    class Message:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    class Users:
        async def add(self, **_kwargs: Any) -> None:
            return None

        async def delete(self, _user_id: str) -> None:
            return None

    class Threads:
        async def create(self, **_kwargs: Any) -> None:
            return None

        async def add_messages(self, _thread_id: str, messages: list[Any]) -> None:
            return None

    class Edges:
        async def get_by_user_id(self, **kwargs: Any) -> list[Any]:
            list_calls.append(kwargs)
            if kwargs.get("uuid_cursor"):
                return [SimpleNamespace(fact="later", uuid="e2")]
            return [
                SimpleNamespace(fact=f"edge-{index}", uuid="e1") for index in range(100)
            ]

    class Client:
        def __init__(self) -> None:
            self.user = Users()
            self.thread = Threads()
            self.graph = SimpleNamespace(edge=Edges())

        async def close(self) -> None:
            return None

    _install_module(monkeypatch, "zep_cloud.types", SimpleNamespace(Message=Message))
    _install_module(
        monkeypatch,
        "zep_cloud.client",
        SimpleNamespace(AsyncZep=lambda api_key: Client()),
    )
    store = ZepMemoryStore(api_key="k", user_id="u1")
    await store.ingest(
        [
            Session(
                label="2026/01/02",
                date=datetime(2026, 1, 2, tzinfo=timezone.utc),
                messages=[ContextMessage("user", "hello")],
            )
        ]
    )
    memories = await store.list_memories()
    assert captured[0]["content"] == "Conversation date: 2026/01/02"
    assert captured[0]["created_at"].startswith("2026-01-02")
    assert captured[1]["content"] == "hello"
    assert memories[-1] == "later"
    assert list_calls[1]["uuid_cursor"] == "e1"


@pytest.mark.asyncio
async def test_supermemory_wait_pages_through_document_list() -> None:
    pages: list[int] = []

    class Documents:
        async def list(self, **kwargs: Any) -> Any:
            pages.append(int(kwargs["page"]))
            if kwargs["page"] == 1:
                return SimpleNamespace(
                    memories=[SimpleNamespace(id="a", status="done")] * 200,
                    pagination=SimpleNamespace(totalPages=2),
                )
            return SimpleNamespace(
                memories=[SimpleNamespace(id="b", status="done")],
                pagination=SimpleNamespace(totalPages=2),
            )

    store = SupermemoryStore.__new__(SupermemoryStore)
    store._client = SimpleNamespace(documents=Documents())
    store._user_id = "u1"
    store._document_ids = {"a", "b"}
    store._documents = ["one", "two"]
    store._wait_timeout = 1
    store._wait_poll_interval = 0.01
    memories = await store.wait_for_extraction(timeout=1, poll_interval=0.01)
    assert memories == ["one", "two"]
    assert pages == [1, 2]


@pytest.mark.asyncio
async def test_supermemory_prefers_singular_container_tag() -> None:
    seen: list[dict[str, Any]] = []

    async def add(**kwargs: Any) -> SimpleNamespace:
        seen.append(kwargs)
        if "container_tag" not in kwargs:
            raise TypeError("unexpected keyword argument 'container_tags'")
        return SimpleNamespace(id="d1")

    store = SupermemoryStore.__new__(SupermemoryStore)
    store._client = SimpleNamespace(add=add)
    store._user_id = "u1"
    store._documents = []
    store._document_ids = set()
    await store.ingest([Session(label="now", messages=[ContextMessage("user", "hi")])])
    assert seen[0]["container_tag"] == "u1"
    assert "container_tags" not in seen[0]


@pytest.mark.asyncio
async def test_langmem_list_and_reset_use_tracked_keys() -> None:
    class Store:
        def __init__(self) -> None:
            self.items: dict[str, SimpleNamespace] = {}

        def get(self, _ns: object, key: str) -> SimpleNamespace | None:
            return self.items.get(key)

        def delete(self, _ns: object, key: str) -> None:
            self.items.pop(key, None)

        def search(self, *_args: object, **_kwargs: object) -> list[object]:
            raise AssertionError("list/reset must not search")

    store = LangMemStore.__new__(LangMemStore)
    store._store = Store()
    store._namespace = ("memories", "u1")
    store._keys = []
    store._existing = []
    store._token_usage = TokenUsage()
    for index in range(120):
        key = str(index)
        store._store.items[key] = SimpleNamespace(value={"text": f"m{index}"})
        store._keys.append(key)
    memories = await store.list_memories()
    assert len(memories) == 120
    await store.reset()
    assert store._keys == []
    assert store._store.items == {}


@pytest.mark.asyncio
async def test_graphiti_reset_uses_clear_data_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    async def clear_data(driver: Any, group_ids: list[str] | None = None) -> None:
        calls.append((driver, group_ids))

    _install_module(
        monkeypatch,
        "graphiti_core.utils.maintenance.graph_data_operations",
        SimpleNamespace(clear_data=clear_data),
    )
    store = GraphitiStore.__new__(GraphitiStore)
    store._graph = SimpleNamespace(driver="driver")
    store._group_id = "group-1"
    store._token_usage = TokenUsage()
    await store.reset()
    assert calls == [("driver", ["group-1"])]


@pytest.mark.asyncio
async def test_graphiti_lists_edges_by_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GroupsEdgesNotFoundError(Exception):
        pass

    async def get_by_group_ids(driver: Any, group_ids: list[str]) -> list[Any]:
        assert driver == "driver"
        assert group_ids == ["group-1"]
        return [SimpleNamespace(fact="listed fact")]

    _install_module(
        monkeypatch,
        "graphiti_core.edges",
        SimpleNamespace(EntityEdge=SimpleNamespace(get_by_group_ids=get_by_group_ids)),
    )
    _install_module(
        monkeypatch,
        "graphiti_core.errors",
        SimpleNamespace(GroupsEdgesNotFoundError=GroupsEdgesNotFoundError),
    )
    store = GraphitiStore.__new__(GraphitiStore)
    store._graph = SimpleNamespace(driver="driver")
    store._group_id = "group-1"
    assert await store.list_memories() == ["listed fact"]


@pytest.mark.asyncio
async def test_graphiti_returns_empty_list_when_group_has_no_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GroupsEdgesNotFoundError(Exception):
        pass

    async def get_by_group_ids(_driver: Any, _group_ids: list[str]) -> list[Any]:
        raise GroupsEdgesNotFoundError

    _install_module(
        monkeypatch,
        "graphiti_core.edges",
        SimpleNamespace(EntityEdge=SimpleNamespace(get_by_group_ids=get_by_group_ids)),
    )
    _install_module(
        monkeypatch,
        "graphiti_core.errors",
        SimpleNamespace(GroupsEdgesNotFoundError=GroupsEdgesNotFoundError),
    )
    store = GraphitiStore.__new__(GraphitiStore)
    store._graph = SimpleNamespace(driver="driver")
    store._group_id = "group-1"
    assert await store.list_memories() == []


def test_oracle_store_passes_llm_and_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: dict[str, Any] = {}

    class OracleAgentMemory:
        def __init__(self, **kwargs: Any) -> None:
            constructed.update(kwargs)

    _install_module(monkeypatch, "oracledb", SimpleNamespace())
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.llms.llm",
        SimpleNamespace(Llm=lambda model: f"llm:{model}"),
    )
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.embedders.embedder",
        SimpleNamespace(Embedder=lambda model: f"embed:{model}"),
    )
    sys.modules["oracleagentmemory.core"].OracleAgentMemory = OracleAgentMemory
    OracleAgentMemoryStore(connection=object(), model="gpt-4o", embedder_model="emb")
    assert constructed["llm"] == "llm:gpt-4o"
    assert constructed["embedder"] == "embed:emb"


@pytest.mark.asyncio
async def test_bedrock_ingest_includes_session_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def create_event(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    _install_module(
        monkeypatch,
        "bedrock_agentcore.memory",
        SimpleNamespace(MemoryClient=Client),
    )
    store = BedrockAgentCoreStore(
        memory_id="mem-1",
        namespace_path="/ns/{user_id}/",
        user_id="u1",
    )
    await store.ingest(
        [Session(label="2026/01/02", messages=[ContextMessage("user", "hi")])]
    )
    assert calls[0]["messages"][0] == ("Conversation date: 2026/01/02", "USER")
    assert calls[0]["messages"][1] == ("hi", "USER")


@pytest.mark.asyncio
async def test_oracle_ingest_includes_session_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    added: list[Any] = []

    class Thread:
        def add_messages(self, messages: list[Any]) -> None:
            added.extend(messages)

    class OracleAgentMemory:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def create_thread(self, **_kwargs: Any) -> Thread:
            return Thread()

    class Message:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    _install_module(monkeypatch, "oracledb", SimpleNamespace())
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.llms.llm",
        SimpleNamespace(Llm=lambda model: f"llm:{model}"),
    )
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.embedders.embedder",
        SimpleNamespace(Embedder=lambda model: f"embed:{model}"),
    )
    _install_module(
        monkeypatch,
        "oracleagentmemory.apis.thread",
        SimpleNamespace(Message=Message),
    )
    sys.modules["oracleagentmemory.core"].OracleAgentMemory = OracleAgentMemory
    store = OracleAgentMemoryStore(connection=object(), model="gpt-4o")
    await store.ingest(
        [Session(label="2026/01/02", messages=[ContextMessage("user", "hi")])]
    )
    assert added[0].kwargs["content"] == "Conversation date: 2026/01/02"
    assert added[1].kwargs["content"] == "hi"


@pytest.mark.asyncio
async def test_oracle_reset_deletes_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[str] = []

    class Thread:
        thread_id = "u1"

        def add_messages(self, _messages: list[Any]) -> None:
            return None

    class OracleAgentMemory:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def create_thread(self, **_kwargs: Any) -> Thread:
            return Thread()

        def delete_thread(self, thread_id: str) -> int:
            deleted.append(thread_id)
            return 1

    class Message:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    _install_module(monkeypatch, "oracledb", SimpleNamespace())
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.llms.llm",
        SimpleNamespace(Llm=lambda model: f"llm:{model}"),
    )
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.embedders.embedder",
        SimpleNamespace(Embedder=lambda model: f"embed:{model}"),
    )
    _install_module(
        monkeypatch,
        "oracleagentmemory.apis.thread",
        SimpleNamespace(Message=Message),
    )
    sys.modules["oracleagentmemory.core"].OracleAgentMemory = OracleAgentMemory
    store = OracleAgentMemoryStore(connection=object(), model="gpt-4o", user_id="u1")
    await store.ingest([Session(label="now", messages=[ContextMessage("user", "hi")])])
    await store.reset()
    assert deleted == ["u1"]
    assert store._thread is None
