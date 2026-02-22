from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.services import twin_store
from app.services.twin_snapshot import latest_snapshot_from_events


router = APIRouter(prefix="/policy", tags=["policy"])


@router.get("/top")
def top(limit: int = 10) -> Dict[str, Any]:
    """Return top-N children ranked by latest fusion global risk.

    This uses the same twin event storage and snapshot builder as the rest
    of the system, making policy view consistent with individual views.
    """
    try:
        ids: List[str] = twin_store.list_child_ids(limit=5000)
        rows: List[Dict[str, Any]] = []
        for cid in ids:
            events = twin_store.list_events(cid, limit=500)
            snap = latest_snapshot_from_events(cid, events).get("snapshot", {}) or {}
            fusion = snap.get("fusion") or {}
            payload = fusion.get("payload") or {}
            r = float(payload.get("global_development_risk", 0.0))
            rows.append(
                {
                    "child_id": cid,
                    "global_risk": r,
                    "global_risk_pct": r * 100.0,
                    "dominant_modality": payload.get("dominant_modality", "unknown"),
                }
            )

        rows.sort(key=lambda x: x["global_risk"], reverse=True)
        top_rows = rows[:limit]
        return {"rows": top_rows, "count": len(rows)}
    except Exception as ex:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Policy view failed: {ex}")
