from __future__ import annotations

from fastapi import APIRouter

from app.services.twin_service import get_events, get_latest_by_modality


router = APIRouter(prefix="/twin", tags=["twin"])


@router.get("/events/{child_id}")
def twin_events(child_id: str):
    """Return full event timeline for a child (oldest -> newest)."""
    return get_events(child_id)


@router.get("/events/latest/{child_id}")
def twin_events_latest(child_id: str):
    """Return latest event payload per modality for a child."""
    return get_latest_by_modality(child_id)
