from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.twin_service import get_latest_events_snapshot, save_twin_event
from app.ml.fusion.infer import predict_from_snapshot


router = APIRouter(prefix="/fusion", tags=["fusion"])


class FusionScoreRequest(BaseModel):
    child_id: str = Field(..., min_length=1)


@router.post("/score")
def score_fusion(req: FusionScoreRequest):
    try:
        snap = get_latest_events_snapshot(req.child_id)
        out = predict_from_snapshot(snap)

        payload = {
            "global_development_risk": out["global_development_risk"],
            "global_development_risk_pct": round(out["global_development_risk"] * 100.0, 2),
            "dominant_modality": out["dominant_modality"],
            "contributions": out["contributions"],
            "confidence": out["confidence"],
            "model_version": out["model_version"],
            "explainability_method": out.get("explainability_method", "unknown"),
        }
        if "features" in out:
            payload["features"] = out["features"]

        save_twin_event(child_id=req.child_id, modality="fusion", payload=payload, timestamp=None)
        return {"child_id": req.child_id, **payload}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion scoring failed: {e}")
