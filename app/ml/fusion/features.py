from __future__ import annotations

from typing import Any, Dict


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def build_fusion_features(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    snapshot format (from /twin/events/latest/{child_id}):
      { "growth": {"payload": {...}, "timestamp": ...}, "learning": {...}, "emotion": {...} }
    """
    feats: Dict[str, float] = {}

    # Growth
    g = snapshot.get("growth")
    if g and isinstance(g, dict):
        gp = g.get("payload", {}) or {}
        feats["growth_risk"] = _safe_float(gp.get("overall_risk"), 0.0)
        feats["growth_conf"] = _safe_float(gp.get("confidence"), 0.0)
        feats["growth_present"] = 1.0
    else:
        feats["growth_risk"] = 0.0
        feats["growth_conf"] = 0.0
        feats["growth_present"] = 0.0

    # Learning
    l = snapshot.get("learning")
    if l and isinstance(l, dict):
        lp = l.get("payload", {}) or {}
        feats["learning_risk"] = _safe_float(lp.get("learning_risk"), 0.0)
        feats["learning_conf"] = _safe_float(lp.get("confidence"), 0.0)
        feats["learning_present"] = 1.0
    else:
        feats["learning_risk"] = 0.0
        feats["learning_conf"] = 0.0
        feats["learning_present"] = 0.0

    # Emotion
    e = snapshot.get("emotion")
    if e and isinstance(e, dict):
        ep = e.get("payload", {}) or {}
        feats["emotion_risk"] = _safe_float(ep.get("distress_risk"), 0.0)
        conf = ep.get("confidence")
        if conf is None:
            probs = ep.get("emotion_probs", {}) or {}
            conf = max([_safe_float(v, 0.0) for v in probs.values()], default=0.0)
        feats["emotion_conf"] = _safe_float(conf, 0.0)
        feats["emotion_present"] = 1.0
    else:
        feats["emotion_risk"] = 0.0
        feats["emotion_conf"] = 0.0
        feats["emotion_present"] = 0.0

    return feats


def max_current_risk(feats: Dict[str, float]) -> float:
    return max(feats["growth_risk"], feats["learning_risk"], feats["emotion_risk"])
