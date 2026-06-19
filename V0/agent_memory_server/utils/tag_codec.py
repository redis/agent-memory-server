import re
from typing import Any


def validate_no_commas_in_tags(
    values: list[str] | None, field_name: str
) -> list[str] | None:
    """Validate that no tag value contains a comma."""
    if not values:
        return values
    for idx, value in enumerate(values):
        if "," in value:
            raise ValueError(
                f"{field_name}[{idx}] contains a comma: {value!r}. "
                "Commas are not allowed because they are used as "
                "delimiters in storage."
            )
    return values


def sanitize_tag_values(values: list[str] | None) -> list[str] | None:
    """Replace commas with spaces, strip whitespace, and drop empty strings.

    Intended for LLM-generated content where rejecting would crash background
    tasks.  Returns ``None`` when *values* is ``None`` or when all values are
    empty after cleaning.
    """
    if values is None:
        return None
    cleaned = [
        re.sub(r"\s+", " ", str(v).replace(",", " ")).strip()
        for v in values
        if v is not None
    ]
    return [v for v in cleaned if v] or None


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
    """Encode TAG values using AMS's canonical comma-separated format.

    Raises ``ValueError`` if any value contains a comma (defense-in-depth;
    all callers should already have clean data by this point).
    """
    if not values:
        return ""
    for value in values:
        if value and "," in value:
            raise ValueError(
                f"Tag value contains a comma: {value!r}. "
                "Commas are not allowed because they are used as "
                "delimiters in storage. Use sanitize_tag_values() for "
                "LLM-generated content."
            )
    return ",".join(value.strip() for value in values if value and value.strip())
