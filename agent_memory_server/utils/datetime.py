from datetime import UTC, datetime


def parse_iso8601_datetime(value: str) -> datetime:
    """Parse an ISO 8601 datetime string and normalize naive values to UTC."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid ISO 8601 datetime format {value!r}: expected str")

    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)

    return parsed
