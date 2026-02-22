from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.twin import TwinEventsResponse, TwinLatestSnapshotResponse
from app.services import twin_store
from app.services.twin_snapshot import latest_snapshot_from_events


router = APIRouter(prefix="/twin/events", tags=["twin-events"])


@router.get("/{child_id}", response_model=TwinEventsResponse)
def get_events(child_id: str) -> TwinEventsResponse:
    """Return full event list for a child in a stable JSON shape."""
    try:
        events = twin_store.list_events(child_id)
        return TwinEventsResponse(value=events, Count=len(events))
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load events: {e}")


@router.get("/latest/{child_id}", response_model=TwinLatestSnapshotResponse)
def get_latest_snapshot(child_id: str) -> TwinLatestSnapshotResponse:
    """Return latest snapshot per modality for a child."""
    try:
        events = twin_store.list_events(child_id)
        snap = latest_snapshot_from_events(child_id, events)
        return TwinLatestSnapshotResponse(**snap)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build latest snapshot: {e}")
