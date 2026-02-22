from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.db.models import TwinState
from app.db.session import SessionLocal
from app.services.time_utils import parse_to_utc_aware
from app.utils.time import now_utc_iso, normalize_to_utc_iso


def save_twin_event(
    *,
    child_id: str,
    modality: str,
    payload: Dict[str, Any],
    timestamp: Optional[str] = None,
) -> str:
    """Persist a single modality event into the TwinState timeline.

    Canonical writer for all modalities (learning, growth, etc.). Stores
    server_timestamp (UTC) in snapshot for consistent timezone-aware timestamps.
    """
    # Normalize client timestamp (if any) and always record a server-side UTC timestamp
    client_ts = normalize_to_utc_iso(timestamp)
    server_ts = now_utc_iso()
    record_id = f"{child_id}:{server_ts}"

    db = SessionLocal()
    try:
        row = TwinState(
            id=record_id,
            child_id=child_id,
            snapshot={
                "modality": modality,
                "payload": payload,
                "server_timestamp": server_ts,
                "client_timestamp": client_ts,
            },
        )
        if modality == "growth":
            row.growth_waz = payload.get("waz")
            row.growth_haz = payload.get("haz")
            row.growth_whz = payload.get("whz")
            row.growth_overall_risk = payload.get("overall_risk")
            row.growth_confidence = payload.get("confidence")

        db.add(row)
        db.commit()
        return record_id
    finally:
        db.close()


def get_events(child_id: str, limit: int = 200) -> list[dict]:
    """Return raw twin events for a child (oldest -> newest)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(TwinState)
            .filter(TwinState.child_id == child_id)
            .order_by(TwinState.created_at.desc())
            .limit(limit)
            .all()
        )
        events: list[dict] = []
        for r in reversed(rows):
            snap = r.snapshot or {}
            events.append(
                {
                    "child_id": r.child_id,
                    "modality": snap.get("modality"),
                    "payload": snap.get("payload"),
                    "timestamp": snap.get("server_timestamp")
                    or (r.created_at.isoformat() if r.created_at else None),
                }
            )
        return events
    finally:
        db.close()


def get_latest_by_modality(child_id: str) -> Dict[str, Any]:
    """Return latest event per modality. Sorts using normalized UTC timestamps."""
    events: List[Dict[str, Any]] = get_events(child_id)
    events_sorted = sorted(events, key=lambda e: parse_to_utc_aware(e.get("timestamp")), reverse=True)

    latest: Dict[str, Dict[str, Any]] = {}
    for e in events_sorted:
        m = e.get("modality")
        if not m:
            continue
        if m not in latest:
            latest[m] = {
                "timestamp": e.get("timestamp"),
                "payload": e.get("payload"),
            }

    return {"child_id": child_id, "snapshot": latest}


def get_latest_events_snapshot(child_id: str) -> Dict[str, Any]:
    """Return latest event per modality as { growth: {payload, timestamp}, learning: {...}, emotion: {...} }.
    Uses parse_to_utc_aware for all timestamp comparison to avoid naive/aware mix."""
    events = get_events(child_id)

    latest: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        m = ev.get("modality")
        if not m:
            continue
        ts = ev.get("timestamp")
        if m not in latest:
            latest[m] = ev
        else:
            if parse_to_utc_aware(ts) > parse_to_utc_aware(latest[m].get("timestamp")):
                latest[m] = ev

    snapshot: Dict[str, Dict[str, Any]] = {}
    for mod, ev in latest.items():
        snapshot[mod] = {
            "timestamp": ev.get("timestamp"),
            "payload": ev.get("payload", {}) or {},
        }
    return snapshot


def get_all_events_grouped() -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns: { child_id: [ {"child_id", "modality", "payload", "timestamp"}, ... ] }
    All events from TwinState, grouped by child, ordered by time.
    """
    db = SessionLocal()
    try:
        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        rows = db.query(TwinState).order_by(TwinState.created_at.asc()).all()
        for r in rows:
            snap = r.snapshot or {}
            ts = snap.get("server_timestamp") or (
                r.created_at.isoformat() if r.created_at else None
            )
            out[r.child_id].append({
                "child_id": r.child_id,
                "modality": snap.get("modality"),
                "payload": snap.get("payload") or {},
                "timestamp": ts,
            })
        return dict(out)
    finally:
        db.close()
