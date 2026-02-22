from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.twin_service import get_latest_events_snapshot, save_twin_event
from app.ml.interventions.engine import compute_severity, priority_from_severity, build_recommendations
from app.ml.interventions.schemas import RecRequest, RecommendationsPayload


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationsPayload)
def recommend(req: RecRequest):
    try:
        snap = get_latest_events_snapshot(req.child_id)
        g = (snap.get("growth", {}).get("payload") or {})
        l = (snap.get("learning", {}).get("payload") or {})
        e = (snap.get("emotion", {}).get("payload") or {})
        f = (snap.get("fusion", {}).get("payload") or {})

        signals = {
            "growth_risk": float(g.get("overall_risk", 0.0)),
            "learning_risk": float(l.get("learning_risk", 0.0)),
            "emotion_risk": float(e.get("distress_risk", 0.0)),
            "fusion_risk": float(f.get("global_development_risk", 0.0)),
            "dominant_modality": f.get("dominant_modality", "unknown"),
        }

        sev = compute_severity(signals)
        priority = priority_from_severity(sev)
        recs = build_recommendations(signals, max_items=req.max_items)

        payload = {
            "priority_level": priority,
            "severity_score": sev,
            "dominant_modality": signals["dominant_modality"],
            "recommendations": recs,
            "model_version": "interventions-v1",
        }

        save_twin_event(child_id=req.child_id, modality="recommendations", payload=payload, timestamp=None)

        return RecommendationsPayload(**payload)

    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {ex}")

