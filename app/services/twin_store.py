from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from app.db.session import SessionLocal
from app.db.models import TwinState


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
        return record_id
    finally:
        db.close()
