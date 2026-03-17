from datetime import datetime


def parse_iso8601_datetime(value: str) -> datetime:
    """Parse an ISO 8601 datetime string, including a trailing Z suffix."""
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value)
