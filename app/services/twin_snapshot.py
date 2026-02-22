from __future__ import annotations

from typing import Any, Dict, List

from app.services.time_utils import parse_to_utc_aware


def latest_snapshot_from_events(child_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute latest state per modality from a list of twin events.

    Each event is expected to have keys: child_id, modality, payload, timestamp (ISO string).
    Returns a pure-JSON-friendly structure:
    {"child_id": ..., "snapshot": {modality: {"timestamp": ..., "payload": {...}}, ...}}
    Uses parse_to_utc_aware for all timestamp comparison (no naive/aware mix).
    """

    if not events:
        return {"child_id": child_id, "snapshot": {}}

    events_sorted = sorted(events, key=lambda e: parse_to_utc_aware(e.get("timestamp")), reverse=True)

    snapshot: Dict[str, Dict[str, Any]] = {}
    for e in events_sorted:
        m = e.get("modality")
        if not m:
            continue
        if m not in snapshot:
            snapshot[m] = {
                "timestamp": e.get("timestamp"),
                "payload": e.get("payload") or {},
            }

    return {"child_id": child_id, "snapshot": snapshot}
