from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_pipeline, load_shap_background
from .config import (
    FEATURE_DEFAULTS_PATH,
    METADATA_PATH,
    PIPELINE_PATH,
    RISK_LEVELS,
    SHAP_BG_PATH,
    TOP_FACTORS_N,
)
from .explain import explain_fallback_importance, explain_instance_shap


@dataclass
class LearningModelBundle:
    pipeline: Any
    metadata: dict
    shap_bg: np.ndarray | None
    defaults: dict


_BUNDLE: LearningModelBundle | None = None


def load_bundle_once() -> LearningModelBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    pipe = load_pipeline(PIPELINE_PATH)
    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    shap_bg = None
    if SHAP_BG_PATH.exists():
        try:
            shap_bg = load_shap_background(SHAP_BG_PATH)
        except Exception:
            shap_bg = None

    defaults: dict = {}
    if FEATURE_DEFAULTS_PATH.exists():
        try:
            defaults = json.loads(FEATURE_DEFAULTS_PATH.read_text())
        except Exception:
            defaults = {}

    _BUNDLE = LearningModelBundle(pipeline=pipe, metadata=meta, shap_bg=shap_bg, defaults=defaults)
    return _BUNDLE


def risk_level_from_p(p: float) -> str:
    for lo, hi, name in RISK_LEVELS:
        if lo <= p < hi:
            return name
    return "HIGH"


def confidence_from_p(p: float) -> float:
    """Confidence from probability: 0 near 0.5, 1 near 0 or 1."""
    return float(abs(p - 0.5) * 2.0)


def _fill_missing_features(features: dict, required_cols: list[str], defaults: dict) -> dict:
    """Fill missing required features using dataset-based defaults.

    - If a required feature is missing and has a default, use that.
    - If no default is available, raise to surface schema mismatch early.
    - Drop unexpected keys so the model sees exactly the schema it was trained on.
    """
    out = dict(features)
    missing = [c for c in required_cols if c not in out]
    for c in missing:
        if c in defaults:
            out[c] = defaults[c]
        else:
            raise ValueError(f"Missing required feature '{c}' and no default available")

    # keep only required columns, in consistent order
    return {k: out[k] for k in required_cols}


def predict_learning_risk(input_dict: dict) -> dict:
    bundle = load_bundle_once()
    pipe = bundle.pipeline
    required_cols = bundle.metadata.get("feature_cols", [])
    if not required_cols:
        raise ValueError("Model metadata missing 'feature_cols'")

    # Track which fields were imputed for transparency
    original_features = dict(input_dict)
    missing = [c for c in required_cols if c not in original_features]

    features_filled = _fill_missing_features(original_features, required_cols, bundle.defaults)
    X = pd.DataFrame([features_filled])
    p = float(pipe.predict_proba(X)[:, 1][0])

    level = risk_level_from_p(p)
    conf = confidence_from_p(p)

    top_factors = explain_instance_shap(
        pipe, X, shap_bg=bundle.shap_bg, top_n=TOP_FACTORS_N
    )
    if top_factors is None:
        top_factors = explain_fallback_importance(pipe, top_n=TOP_FACTORS_N)

    return {
        "learning_risk": p,
        "learning_risk_pct": p * 100.0,
        "risk_level": level,
        "at_risk_pred": bool(p >= 0.5),
        "top_factors": top_factors,
        "confidence": conf,
        "imputed_fields": missing,
        "model_version": bundle.metadata.get("model_version", "unknown"),
    }
