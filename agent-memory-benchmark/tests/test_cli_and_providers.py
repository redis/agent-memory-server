from __future__ import annotations

import builtins
import inspect
import json
import sys

import pytest

from agent_memory_benchmark import cli
from agent_memory_benchmark.benchmark.runner import (
    parse_provider_params,
    redact_provider_params,
    store_kwargs,
)
from agent_memory_benchmark.memory import STORES, get_store_class
from agent_memory_benchmark.memory.mem0_store import Mem0MemoryStore


def test_cli_parser_exposes_current_commands_and_defaults() -> None:
    parser = cli.build_parser()

    run = parser.parse_args(["run", "--provider", "mem0"])
    assert run.command == "run"
    assert run.split == "small"
    assert run.results_root.name == "experiment_results"

    judge = parser.parse_args(
        ["judge", "--experiment", "trial", "--judge-model", "local-judge"]
    )
    assert judge.command == "judge"
    assert judge.experiment == "trial"
    assert judge.model == "local-judge"
    assert judge.results_root.name == "experiment_results"


@pytest.mark.parametrize("value", ["0", "-2"])
def test_cli_rejects_non_positive_numeric_options(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--provider", "mem0", "--retries", value])


def test_provider_listing_is_sorted_in_text_and_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = sorted(STORES)

    assert cli.main(["providers"]) == 0
    assert capsys.readouterr().out.splitlines() == expected

    assert cli.main(["providers", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_registry_imports_without_loading_provider_extras() -> None:
    optional_roots = {
        "bedrock_agentcore",
        "boto3",
        "graphiti_core",
        "langmem",
        "mem0",
        "oracleagentmemory",
        "oracledb",
        "supermemory",
        "vertexai",
        "zep_cloud",
    }
    assert optional_roots.isdisjoint(sys.modules)
    assert get_store_class("mem0") is Mem0MemoryStore
    with pytest.raises(KeyError, match="Available:"):
        get_store_class("not-a-provider")


def test_all_runnable_providers_accept_runner_user_id() -> None:
    for store_type in STORES.values():
        signature = inspect.signature(store_type.__init__)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        assert "user_id" in signature.parameters or accepts_kwargs


def test_missing_provider_extra_has_actionable_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mem0":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError) as error:
        Mem0MemoryStore()
    message = str(error.value)
    assert "mem0ai" in message
    assert "agent-memory-benchmark[mem0]" in message


def test_provider_param_coercion_and_last_value_wins() -> None:
    assert parse_provider_params(
        [
            "enabled=TRUE",
            "disabled=false",
            "missing=null",
            "count=7",
            "ratio=1.25",
            "label=007x",
            "empty=",
            "count=8",
            "store_id=123",
            "project=456789",
            "top_k=10",
        ]
    ) == {
        "enabled": True,
        "disabled": False,
        "missing": None,
        "count": 8,
        "ratio": 1.25,
        "label": "007x",
        "empty": "",
        "store_id": "123",
        "project": "456789",
        "top_k": 10,
    }


def test_sensitive_provider_params_are_redacted_from_metadata() -> None:
    assert redact_provider_params(
        {
            "api_key": "secret-value",
            "access_token": "secret-value",
            "neo4j_password": "secret-value",
            "region": "us-east-1",
        }
    ) == {
        "api_key": "<redacted>",
        "access_token": "<redacted>",
        "neo4j_password": "<redacted>",
        "region": "us-east-1",
    }


def test_store_kwargs_keep_isolated_user_id() -> None:
    assert store_kwargs({"model": "gpt-4o", "user_id": "shared"}, "bench-1") == {
        "model": "gpt-4o",
        "user_id": "bench-1",
    }


@pytest.mark.parametrize(
    ("item", "message"),
    [("broken", "Expected KEY=VALUE"), ("=value", "Empty parameter name")],
)
def test_provider_param_validation(item: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_provider_params([item])
