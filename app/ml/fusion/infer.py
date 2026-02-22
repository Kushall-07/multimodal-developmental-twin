"""Fusion inference: XGBoost or fallback weighted-mean from latest snapshot."""
from __future__ import annotations

import json
import os
from joblib import load

from .config import META_PATH, MODEL_PATH, MODEL_VERSION
from .features import build_fusion_features

_bundle = None


def _safe_norm_contrib(d: dict) -> dict:
    s = sum(max(float(v), 0.0) for v in d.values())
    if s <= 1e-12:
        n = len(d)
        return {k: 1.0 / n for k in d} if n else d
    return {k: float(max(v, 0.0)) / s for k, v in d.items()}


def _dominant_from_contrib(contrib: dict) -> str:
    if not contrib:
        return "unknown"
    return max(contrib, key=lambda k: contrib[k])


def dominant_by_impact(feats: dict) -> str:
    impact = {
        "growth": feats["growth_risk"] * (feats["growth_conf"] * feats["growth_present"]),
        "learning": feats["learning_risk"] * (feats["learning_conf"] * feats["learning_present"]),
        "emotion": feats["emotion_risk"] * (feats["emotion_conf"] * feats["emotion_present"]),
    }
    return max(impact, key=impact.get) if impact else "unknown"


def _load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return load(MODEL_PATH)


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = _load_bundle()
    return _bundle


def fusion_model_is_reliable() -> bool:
    """Gate: use model only if training metrics meet minimum quality (production gating)."""
    try:
        meta = json.loads(META_PATH.read_text())
        m = meta.get("metrics", {}) or {}
        roc = m.get("roc_auc")
        f1 = m.get("f1")
        rows = m.get("rows", 0)
        if rows < 200:
            return False
        if roc is None or roc < 0.58:
            return False
        if f1 is None or f1 < 0.25:
            return False
        return True
    except Exception:
        return False


def fallback_fusion(feats: dict) -> tuple[float, str, float]:
    """Reliability-weighted mean (fallback when model not trained). Returns (fused_risk, dominant_modality, confidence)."""
    w_g = feats["growth_conf"] * feats["growth_present"]
    w_l = feats["learning_conf"] * feats["learning_present"]
    w_e = feats["emotion_conf"] * feats["emotion_present"]

    risks = [
        (feats["growth_risk"], w_g, "growth"),
        (feats["learning_risk"], w_l, "learning"),
        (feats["emotion_risk"], w_e, "emotion"),
    ]
    denom = sum(w for _, w, _ in risks) + 1e-6
    fused = sum(r * w for r, w, _ in risks) / denom

    dom = max(risks, key=lambda x: x[0])[2]
    conf = max(w_g, w_l, w_e) if denom > 0 else 0.0

    return fused, dom, conf


def predict_from_snapshot(snapshot: dict) -> dict:
    feats = build_fusion_features(snapshot)
    b = get_bundle()

    # Quality gate: no model or model not reliable -> fallback (champion-challenger)
    if b is None or not fusion_model_is_reliable():
        fused, _, conf = fallback_fusion(feats)
        w_g = feats["growth_conf"] * feats["growth_present"]
        w_l = feats["learning_conf"] * feats["learning_present"]
        w_e = feats["emotion_conf"] * feats["emotion_present"]
        contrib = _safe_norm_contrib({"growth": w_g, "learning": w_l, "emotion": w_e})
        return {
            "global_development_risk": float(fused),
            "dominant_modality": dominant_by_impact(feats),
            "confidence": float(conf),
            "model_version": "fusion-fallback-v0",
            "features": feats,
            "contributions": contrib,
            "explainability_method": "fallback-weighted",
        }

    model = b["model"]
    cols = b["features"]

    x = [feats[c] for c in cols]
    prob = float(model.predict_proba([x])[0][1])

    # ---------- SHAP-based per-request contributions ----------
    modality_contrib = None
    shap_available = False
    try:
        import shap  # optional
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values([x])

        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if hasattr(sv, "shape") and len(getattr(sv, "shape", [])) == 2:
            sv = sv[0]

        abs_sv = [abs(float(v)) for v in sv]
        per_feat = dict(zip(cols, abs_sv))

        modality_contrib = {
            "growth": sum(v for k, v in per_feat.items() if k.startswith("growth")),
            "learning": sum(v for k, v in per_feat.items() if k.startswith("learning")),
            "emotion": sum(v for k, v in per_feat.items() if k.startswith("emotion")),
        }
        modality_contrib = _safe_norm_contrib(modality_contrib)
        shap_available = True

    except Exception:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            modality_contrib = _safe_norm_contrib({"growth": 1.0, "learning": 1.0, "emotion": 1.0})
        else:
            per_feat = dict(zip(cols, [float(v) for v in importances]))
            modality_contrib = {
                "growth": sum(v for k, v in per_feat.items() if k.startswith("growth")),
                "learning": sum(v for k, v in per_feat.items() if k.startswith("learning")),
                "emotion": sum(v for k, v in per_feat.items() if k.startswith("emotion")),
            }
            modality_contrib = _safe_norm_contrib(modality_contrib)

    dominant = _dominant_from_contrib(modality_contrib)

    confidence = float(min(1.0, max(0.0, abs(prob - 0.5) * 2.0)))

    return {
        "global_development_risk": prob,
        "dominant_modality": dominant,
        "confidence": confidence,
        "model_version": b.get("model_version", MODEL_VERSION),
        "explainability_method": "shap" if shap_available else "feature_importance",
        "features": feats,
        "contributions": modality_contrib,
    }
