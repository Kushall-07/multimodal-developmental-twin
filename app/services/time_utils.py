"""
Canonical timestamp normalization for Twin event store.
All Twin reads that sort or compare timestamps should use parse_to_utc_aware.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_to_utc_aware(ts: Any) -> datetime:
    """
    Convert a timestamp (string/datetime/None) to timezone-aware UTC datetime.
    - naive datetime -> assume UTC
    - iso string without tz -> assume UTC
    - iso string with tz -> convert to UTC
    """
    if ts is None:
        return datetime.min.replace(tzinfo=timezone.utc)

    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    s = str(ts).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
