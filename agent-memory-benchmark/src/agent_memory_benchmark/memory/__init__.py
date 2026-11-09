"""Provider-neutral memory adapters.

Third-party SDKs are imported only when their adapter is instantiated, so
importing this package never requires provider extras or credentials.
"""

from agent_memory_benchmark.memory.base import (
    MemoryStore,
    QueryResult,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.bedrock_agentcore_store import BedrockAgentCoreStore
from agent_memory_benchmark.memory.graphiti_store import GraphitiStore
from agent_memory_benchmark.memory.langmem_store import LangMemStore
from agent_memory_benchmark.memory.mem0_store import Mem0MemoryStore
from agent_memory_benchmark.memory.oracle_agent_memory_store import (
    OracleAgentMemoryStore,
)
from agent_memory_benchmark.memory.redis_agent_memory_store import (
    RedisAgentMemoryStore,
)
from agent_memory_benchmark.memory.supermemory_store import SupermemoryStore
from agent_memory_benchmark.memory.vertex_memory_bank import VertexMemoryBankStore
from agent_memory_benchmark.memory.zep_store import ZepMemoryStore

STORES: dict[str, type[MemoryStore]] = {
    "redis-agent-memory": RedisAgentMemoryStore,
    "mem0": Mem0MemoryStore,
    "langmem": LangMemStore,
    "zep": ZepMemoryStore,
    "graphiti": GraphitiStore,
    "supermemory": SupermemoryStore,
    "vertex-memory-bank": VertexMemoryBankStore,
    "bedrock-agentcore": BedrockAgentCoreStore,
    "oracle-agent-memory": OracleAgentMemoryStore,
}


def get_store_class(name: str) -> type[MemoryStore]:
    """Resolve an adapter by public registry name."""
    try:
        return STORES[name]
    except KeyError as exc:
        available = ", ".join(sorted(STORES))
        raise KeyError(
            f"Unknown memory store {name!r}. Available: {available}"
        ) from exc


__all__ = [
    "BedrockAgentCoreStore",
    "GraphitiStore",
    "LangMemStore",
    "Mem0MemoryStore",
    "MemoryStore",
    "OracleAgentMemoryStore",
    "QueryResult",
    "RedisAgentMemoryStore",
    "STORES",
    "SupermemoryStore",
    "TokenUsage",
    "VertexMemoryBankStore",
    "ZepMemoryStore",
    "get_store_class",
    "normalize_role",
]
