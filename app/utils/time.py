from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def now_utc_iso() -> str:
    """Return current time as a UTC ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def normalize_to_utc_iso(ts: Optional[str]) -> str:
    """Normalize an optional timestamp string to UTC ISO 8601.

    - If ts is None/empty, returns current UTC time.
    - Accepts ISO strings with or without timezone.
    - Falls back to current UTC time if parsing fails.
    """
    if not ts:
        return now_utc_iso()

    try:
        from dateutil.parser import isoparse

        dt = isoparse(ts)
    except Exception:
        return now_utc_iso()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.isoformat()
