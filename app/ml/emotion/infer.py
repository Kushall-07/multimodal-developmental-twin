"""Emotion inference: load trained model, predict from image, map to distress_risk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .config import MODEL_PATH, META_PATH, IMG_SIZE, CLASSES

# 4-class distress: high for sad/angry, moderate for bored, low for happy
def _distress_from_4class(probs: dict[str, float]) -> float:
    distress = 0.0
    distress += probs.get("angry", 0.0) * 1.0
    distress += probs.get("sad", 0.0) * 1.0
    distress += probs.get("bored", 0.0) * 0.6
    distress += probs.get("happy", 0.0) * 0.05
    return min(1.0, distress)

_model_and_meta: tuple[Any, dict] | None = None


def _get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def load_emotion_model_once():
    """Load trained emotion model and metadata (cached). Uses metadata.json for 4-class list when present."""
    global _model_and_meta
    if _model_and_meta is not None:
        return _model_and_meta
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}. Run training first.")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    idx_to_class = ckpt["idx_to_class"]
    num_classes = len(idx_to_class)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    meta = {
        "idx_to_class": idx_to_class,
        "model_version": ckpt.get("model_version", "emotion-4class-v1"),
    }
    if META_PATH.exists():
        file_meta = json.loads(META_PATH.read_text())
        meta["classes"] = file_meta.get("classes", list(idx_to_class.values()))
    else:
        meta["classes"] = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]
    _model_and_meta = (model, meta)
    return _model_and_meta


def image_to_tensor(image: Image.Image):
    """Convert PIL image to batch tensor (1, 3, H, W)."""
    transform = _get_transform()
    return transform(image).unsqueeze(0)


def probs_to_distress_risk(probs: dict[str, float]) -> float:
    """Map 4-class emotion probabilities to distress_risk in [0, 1]. High: angry/sad; moderate: bored; low: happy."""
    return _distress_from_4class(probs)


def predict_emotion(image: Image.Image) -> dict[str, Any]:
    """
    Run emotion model on image. Returns emotion_probs, pred_emotion, distress_risk, etc.
    """
    model, meta = load_emotion_model_once()
    idx_to_class = meta["idx_to_class"]
    model_version = meta["model_version"]

    x = image_to_tensor(image)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    pred_idx = int(logits.argmax(dim=1).item())
    pred_emotion = idx_to_class.get(pred_idx, CLASSES[0])

    emotion_probs = {idx_to_class.get(i, f"class_{i}"): round(p, 4) for i, p in enumerate(probs)}
    distress_risk = probs_to_distress_risk(emotion_probs)
    distress_risk_pct = round(distress_risk * 100.0, 2)

    # Confidence / uncertainty from max class probability
    confidence = max(emotion_probs.values()) if emotion_probs else 0.0
    uncertainty = round(1.0 - confidence, 4)

    # Qualitative risk level based on distress_risk
    if distress_risk < 0.3:
        risk_level = "LOW"
    elif distress_risk < 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "pred_emotion": pred_emotion,
        "emotion_probs": emotion_probs,
        "distress_risk": round(distress_risk, 4),
        "distress_risk_pct": distress_risk_pct,
        "confidence": round(confidence, 4),
        "uncertainty": uncertainty,
        "risk_level": risk_level,
        "model_version": model_version,
    }
