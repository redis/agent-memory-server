"""Helpers for parsing JSON-shaped LLM responses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_JSON_START_RE = re.compile(r"[{\[]")


def parse_llm_json(content: str) -> Any:
    """Parse JSON from raw, fenced, or prose-wrapped LLM responses."""
    normalized = content.strip()
    decoder = json.JSONDecoder()

    try:
        return decoder.decode(normalized)
    except json.JSONDecodeError as error:
        original_error = error

    for candidate in _iter_json_candidates(normalized):
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return parsed
        except json.JSONDecodeError:
            continue

    raise original_error


def _iter_json_candidates(content: str) -> Iterator[str]:
    """Yield likely JSON payloads embedded within an LLM response."""
    seen: set[str] = set()

    for match in _CODE_FENCE_RE.finditer(content):
        candidate = match.group(1).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    # Fall back to scanning for embedded JSON objects/arrays inside prose.
    for match in _JSON_START_RE.finditer(content):
        candidate = content[match.start() :].lstrip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate
