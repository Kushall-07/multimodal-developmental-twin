from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel

from app.db.session import SessionLocal
from app.db.models import TwinState


class TwinUpdateResponse(BaseModel):
    id: str
    child_id: str
    snapshot: Dict[str, Any]
    growth_waz: str | None
    growth_haz: str | None
    growth_whz: str | None
    growth_overall_risk: str | None
    growth_confidence: str | None


def persist_twin_update(*, child_id: str, modality: str, payload: Dict[str, Any]) -> str:
    """Persist a twin update event and return its record ID."""
    record_id = f"{child_id}:{datetime.utcnow().isoformat()}"

    db = SessionLocal()
    try:
        row = TwinState(
            id=record_id,
            child_id=child_id,
            snapshot={"modality": modality, "payload": payload},
        )

        if modality == "growth":
            row.growth_waz = payload.get("waz")
            row.growth_haz = payload.get("haz")
            row.growth_whz = payload.get("whz")
            row.growth_overall_risk = payload.get("overall_risk")
            row.growth_confidence = payload.get("confidence")

        db.add(row)
        db.commit()
        db.refresh(row)
        return TwinUpdateResponse.from_orm(row).dict()
    finally:
        db.close()


def list_events(child_id: str, limit: int = 500) -> list[dict]:
    """Return twin events for a child as a plain list of dicts.

    This is the single source of truth for the /twin/events API and
    is intentionally JSON-only (no ORM instances, no datetime objects).
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(TwinState)
            .filter(TwinState.child_id == child_id)
            .order_by(TwinState.created_at.asc())
            .limit(limit)
            .all()
        )
        events: list[dict] = []
        for r in rows:
            snap = r.snapshot or {}
            events.append(
                {
                    "child_id": r.child_id,
                    "modality": snap.get("modality"),
                    "payload": snap.get("payload") or {},
                    "timestamp": snap.get("server_timestamp")
                    or (r.created_at.isoformat() if r.created_at else None),
                }
            )
        return events
    finally:
        db.close()


def list_child_ids(limit: int = 5000) -> list[str]:
    """Return distinct child_ids present in the TwinState table.

    Used by policy and population-level views.
    """
    db = SessionLocal()
    try:
        rows = db.query(TwinState.child_id).distinct().limit(limit).all()
        return [r[0] for r in rows]
    finally:
        db.close()
