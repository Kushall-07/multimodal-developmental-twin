from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.learning.infer import load_bundle_once, predict_learning_risk
from app.services.twin_service import save_twin_event


router = APIRouter(prefix="/learning", tags=["learning"])


class LearningScoreRequest(BaseModel):
    child_id: str = Field(..., min_length=1)
    features: Dict[str, Any] = Field(..., description="Student features excluding G1/G2/G3")
    timestamp: Optional[str] = None


class LearningScoreResponse(BaseModel):
    child_id: str
    learning_risk: float
    learning_risk_pct: float
    risk_level: str
    top_factors: list
    confidence: float
    model_version: str


@router.on_event("startup")
def _warmup() -> None:
    # Ensure model is loaded when the router starts
    load_bundle_once()


@router.post("/score", response_model=LearningScoreResponse)
def score_learning(req: LearningScoreRequest) -> LearningScoreResponse:
    try:
        out = predict_learning_risk(req.features)
    except Exception as e:  # pragma: no cover - surfaced as HTTP error
        raise HTTPException(status_code=400, detail=f"Learning inference failed: {e}")

    payload = {
        "at_risk_pred": out["at_risk_pred"],
        "learning_risk": out["learning_risk"],
        "learning_risk_pct": out["learning_risk_pct"],
        "risk_level": out["risk_level"],
        "top_factors": out["top_factors"],
        "confidence": out["confidence"],
        "imputed_fields": out.get("imputed_fields", []),
        "model_version": out["model_version"],
        "input_summary": _input_summary(req.features),
    }

    save_twin_event(
        child_id=req.child_id,
        modality="learning",
        payload=payload,
        timestamp=req.timestamp,
    )

    return LearningScoreResponse(
        child_id=req.child_id,
        learning_risk=out["learning_risk"],
        learning_risk_pct=out["learning_risk_pct"],
        risk_level=out["risk_level"],
        top_factors=out["top_factors"],
        confidence=out["confidence"],
        model_version=out["model_version"],
    )


def _input_summary(features: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "age",
        "sex",
        "studytime",
        "failures",
        "absences",
        "schoolsup",
        "famsup",
        "internet",
    ]
    return {k: features.get(k) for k in keys if k in features}
