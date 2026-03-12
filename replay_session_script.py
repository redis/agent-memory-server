from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a conversation fixture against working memory one turn at a time "
            "and report PUT/GET latency."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a conversation JSON file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Agent Memory Server base URL.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id override. Defaults to payload.data.dataset_id.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional namespace override. Defaults to payload.namespace.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional user_id override. Defaults to payload.user_id.",
    )
    parser.add_argument(
        "--context-window-max",
        type=int,
        default=None,
        help="Optional context window max to encourage summarization during replay.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name for context-window calculations.",
    )
    parser.add_argument(
        "--recent-limit",
        type=int,
        default=8,
        help="Recent message limit used for GET readback. Use 0 to fetch all messages.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Optional maximum number of turns to replay.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between turns.",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Delete the target session before replaying.",
    )
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        default=None,
        help="Optional JSONL file for per-turn snapshots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn progress output.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_session_metadata(
    payload: dict[str, Any],
    *,
    session_id_override: str | None,
    namespace_override: str | None,
    user_id_override: str | None,
) -> tuple[str, str | None, str | None]:
    data = payload.get("data") or {}
    session_id = session_id_override or data.get("dataset_id")
    if not session_id:
        raise ValueError(
            "session_id is required. Provide --session-id or include data.dataset_id."
        )

    namespace = (
        namespace_override
        if namespace_override is not None
        else payload.get("namespace")
    )
    user_id = (
        user_id_override if user_id_override is not None else payload.get("user_id")
    )
    return session_id, namespace, user_id


def build_put_payload(payload: dict[str, Any], turn_index: int) -> dict[str, Any]:
    next_payload = dict(payload)
    next_payload["messages"] = list((payload.get("messages") or [])[:turn_index])
    return next_payload


def build_query_params(
    *,
    namespace: str | None,
    user_id: str | None,
    context_window_max: int | None,
    model_name: str | None,
    recent_limit: int | None = None,
) -> dict[str, str]:
    params: dict[str, str] = {}
    if namespace is not None:
        params["namespace"] = namespace
    if user_id is not None:
        params["user_id"] = user_id
    if context_window_max is not None:
        params["context_window_max"] = str(context_window_max)
    if model_name is not None:
        params["model_name"] = model_name
    if recent_limit is not None and recent_limit > 0:
        params["recent_messages_limit"] = str(recent_limit)
    return params


async def delete_session(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    namespace: str | None,
    user_id: str | None,
) -> None:
    params = build_query_params(
        namespace=namespace,
        user_id=user_id,
        context_window_max=None,
        model_name=None,
    )
    response = await client.delete(f"/v1/working-memory/{session_id}", params=params)
    if response.status_code not in (200, 404):
        response.raise_for_status()


async def timed_put(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    payload: dict[str, Any],
    namespace: str | None,
    user_id: str | None,
    context_window_max: int | None,
    model_name: str | None,
) -> tuple[dict[str, Any], float]:
    params = build_query_params(
        namespace=namespace,
        user_id=user_id,
        context_window_max=context_window_max,
        model_name=model_name,
    )
    start = perf_counter()
    response = await client.put(
        f"/v1/working-memory/{session_id}",
        json=payload,
        params=params,
    )
    elapsed_ms = (perf_counter() - start) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


async def timed_get(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    namespace: str | None,
    user_id: str | None,
    context_window_max: int | None,
    model_name: str | None,
    recent_limit: int,
) -> tuple[dict[str, Any], float]:
    params = build_query_params(
        namespace=namespace,
        user_id=user_id,
        context_window_max=context_window_max,
        model_name=model_name,
        recent_limit=recent_limit if recent_limit > 0 else None,
    )
    start = perf_counter()
    response = await client.get(f"/v1/working-memory/{session_id}", params=params)
    elapsed_ms = (perf_counter() - start) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": float(len(values)),
        "min_ms": min(values),
        "avg_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def build_snapshot(
    *,
    turn_index: int,
    total_turns: int,
    written_message: dict[str, Any],
    visible_state: dict[str, Any],
    put_latency_ms: float,
    get_latency_ms: float,
) -> dict[str, Any]:
    visible_messages = visible_state.get("messages") or []
    context_text = visible_state.get("context") or ""
    return {
        "turn_index": turn_index,
        "total_turns": total_turns,
        "written_message_id": written_message.get("id"),
        "written_role": written_message.get("role"),
        "written_content_preview": (written_message.get("content") or "")[:160],
        "put_latency_ms": round(put_latency_ms, 3),
        "get_latency_ms": round(get_latency_ms, 3),
        "visible_message_count": len(visible_messages),
        "visible_message_ids": [message.get("id") for message in visible_messages],
        "context_present": bool(context_text),
        "context_length": len(context_text),
        "context_percentage_total_used": visible_state.get(
            "context_percentage_total_used"
        ),
        "context_percentage_until_summarization": visible_state.get(
            "context_percentage_until_summarization"
        ),
    }


def print_turn_progress(
    *,
    turn_index: int,
    total_turns: int,
    written_message: dict[str, Any],
    visible_state: dict[str, Any],
    put_latency_ms: float,
    get_latency_ms: float,
) -> None:
    visible_messages = visible_state.get("messages") or []
    context_text = visible_state.get("context") or ""
    last_visible_id = visible_messages[-1]["id"] if visible_messages else "-"
    print(
        f"[{turn_index:04d}/{total_turns:04d}] "
        f"role={written_message['role']:<9} "
        f"put_ms={put_latency_ms:>8.2f} "
        f"get_ms={get_latency_ms:>8.2f} "
        f"visible={len(visible_messages):>3} "
        f"context_len={len(context_text):>4} "
        f"last_visible={last_visible_id}"
    )


def write_snapshots(path: Path, snapshots: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for snapshot in snapshots:
            handle.write(json.dumps(snapshot) + "\n")


async def replay_dataset(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_dataset(args.dataset)
    session_id, namespace, user_id = resolve_session_metadata(
        payload,
        session_id_override=args.session_id,
        namespace_override=args.namespace,
        user_id_override=args.user_id,
    )

    all_messages = payload.get("messages") or []
    total_turns = len(all_messages)
    if args.stop_after is not None:
        total_turns = min(total_turns, args.stop_after)

    put_latencies_ms: list[float] = []
    get_latencies_ms: list[float] = []
    snapshots: list[dict[str, Any]] = []
    first_summary_turn: int | None = None
    final_state: dict[str, Any] | None = None

    start_run = perf_counter()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=60.0) as client:
        if args.reset_session:
            await delete_session(
                client,
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )

        for turn_index in range(1, total_turns + 1):
            message = all_messages[turn_index - 1]
            put_payload = build_put_payload(payload, turn_index)

            _put_response, put_latency_ms = await timed_put(
                client,
                session_id=session_id,
                payload=put_payload,
                namespace=namespace,
                user_id=user_id,
                context_window_max=args.context_window_max,
                model_name=args.model_name,
            )
            visible_state, get_latency_ms = await timed_get(
                client,
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
                context_window_max=args.context_window_max,
                model_name=args.model_name,
                recent_limit=args.recent_limit,
            )

            put_latencies_ms.append(put_latency_ms)
            get_latencies_ms.append(get_latency_ms)
            final_state = visible_state

            if first_summary_turn is None and (visible_state.get("context") or ""):
                first_summary_turn = turn_index

            snapshot = build_snapshot(
                turn_index=turn_index,
                total_turns=total_turns,
                written_message=message,
                visible_state=visible_state,
                put_latency_ms=put_latency_ms,
                get_latency_ms=get_latency_ms,
            )
            snapshots.append(snapshot)

            if not args.quiet:
                print_turn_progress(
                    turn_index=turn_index,
                    total_turns=total_turns,
                    written_message=message,
                    visible_state=visible_state,
                    put_latency_ms=put_latency_ms,
                    get_latency_ms=get_latency_ms,
                )

            if args.sleep_seconds > 0:
                await asyncio.sleep(args.sleep_seconds)

    total_runtime_ms = (perf_counter() - start_run) * 1000

    if args.snapshot_file is not None:
        write_snapshots(args.snapshot_file, snapshots)

    final_messages = (final_state or {}).get("messages") or []
    final_context = (final_state or {}).get("context") or ""

    return {
        "dataset": str(args.dataset),
        "session_id": session_id,
        "namespace": namespace,
        "user_id": user_id,
        "turns_replayed": total_turns,
        "summary_first_seen_turn": first_summary_turn,
        "final_visible_message_count": len(final_messages),
        "final_visible_message_ids": [message.get("id") for message in final_messages],
        "final_context_present": bool(final_context),
        "final_context_length": len(final_context),
        "recent_limit": args.recent_limit,
        "context_window_max": args.context_window_max,
        "model_name": args.model_name,
        "total_runtime_ms": total_runtime_ms,
        "put_latency": summarize_latencies(put_latencies_ms),
        "get_latency": summarize_latencies(get_latencies_ms),
        "snapshot_file": str(args.snapshot_file) if args.snapshot_file else None,
    }


def print_final_report(summary: dict[str, Any]) -> None:
    print()
    print("Replay Summary")
    print(f"dataset={summary['dataset']}")
    print(f"session_id={summary['session_id']}")
    print(f"turns_replayed={summary['turns_replayed']}")
    print(f"summary_first_seen_turn={summary['summary_first_seen_turn']}")
    print(f"final_visible_message_count={summary['final_visible_message_count']}")
    print(f"final_context_present={summary['final_context_present']}")
    print(f"final_context_length={summary['final_context_length']}")
    print(f"total_runtime_ms={summary['total_runtime_ms']:.2f}")

    put_latency = summary["put_latency"]
    print(
        "put_latency_ms="
        f"count={int(put_latency['count'])} "
        f"min={put_latency['min_ms']:.2f} "
        f"avg={put_latency['avg_ms']:.2f} "
        f"p50={put_latency['p50_ms']:.2f} "
        f"p95={put_latency['p95_ms']:.2f} "
        f"max={put_latency['max_ms']:.2f}"
    )

    get_latency = summary["get_latency"]
    print(
        "get_latency_ms="
        f"count={int(get_latency['count'])} "
        f"min={get_latency['min_ms']:.2f} "
        f"avg={get_latency['avg_ms']:.2f} "
        f"p50={get_latency['p50_ms']:.2f} "
        f"p95={get_latency['p95_ms']:.2f} "
        f"max={get_latency['max_ms']:.2f}"
    )

    if summary["snapshot_file"] is not None:
        print(f"snapshot_file={summary['snapshot_file']}")


async def main_async() -> None:
    args = parse_args()
    summary = await replay_dataset(args)
    print_final_report(summary)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
