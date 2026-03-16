import re
from typing import Any


def decode_tag_values(raw: Any) -> list[str]:
    """Decode TAG values from canonical or legacy serialized forms."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(value).strip() for value in raw if str(value).strip()]
    if isinstance(raw, str):
        parts = re.split(r"[|,]", raw)
        return [part.strip() for part in parts if part.strip()]
    return []


def encode_tag_values(values: list[str] | None) -> str:
    """Encode TAG values using AMS's canonical comma-separated format."""
    if not values:
        return ""
    return ",".join(value.strip() for value in values if value and value.strip())
