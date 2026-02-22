from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
import io

from app.ml.emotion.infer import load_emotion_model_once, predict_emotion
from app.services.twin_service import save_twin_event


router = APIRouter(prefix="/emotion", tags=["emotion"])


class EmotionScoreResponse(BaseModel):
    child_id: str
    pred_emotion: str
    emotion_probs: dict
    distress_risk: float
    distress_risk_pct: float
    confidence: float
    uncertainty: float
    risk_level: str
    model_version: str


@router.on_event("startup")
def _warmup() -> None:
    try:
        load_emotion_model_once()
    except FileNotFoundError:
        pass  # model not trained yet; will raise on first /score


@router.post("/score", response_model=EmotionScoreResponse)
async def score_emotion(
    child_id: str = Form(..., min_length=1),
    image: UploadFile = File(...),
) -> EmotionScoreResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        out = predict_emotion(img)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion inference failed: {e}")

    # Safeguard: downgrade risk interpretation when model is uncertain
    confidence = float(out.get("confidence", 0.0))
    uncertainty = float(out.get("uncertainty", max(0.0, 1.0 - confidence)))
    risk_level = out.get("risk_level", "UNKNOWN")
    if confidence < 0.35:
        risk_level = "LOW_CONFIDENCE"

    payload = {
        "pred_emotion": out["pred_emotion"],
        "emotion_probs": out["emotion_probs"],
        "distress_risk": out["distress_risk"],
        "distress_risk_pct": out["distress_risk_pct"],
        "confidence": confidence,
        "uncertainty": uncertainty,
        "risk_level": risk_level,
        "model_version": out["model_version"],
    }
    save_twin_event(child_id=child_id, modality="emotion", payload=payload, timestamp=None)

    return EmotionScoreResponse(
        child_id=child_id,
        pred_emotion=out["pred_emotion"],
        emotion_probs=out["emotion_probs"],
        distress_risk=out["distress_risk"],
        distress_risk_pct=out["distress_risk_pct"],
        confidence=confidence,
        uncertainty=uncertainty,
        risk_level=risk_level,
        model_version=out["model_version"],
    )
