from __future__ import annotations

from typing import Any, Dict, List

from app.ml.interventions.catalog import CATALOG


def compute_severity(s: Dict[str, float]) -> float:
    """Continuous severity score in [0,1] from fused + modality risks."""
    sev = (
        0.55 * s.get("fusion_risk", 0.0)
        + 0.20 * s.get("emotion_risk", 0.0)
        + 0.15 * s.get("learning_risk", 0.0)
        + 0.10 * s.get("growth_risk", 0.0)
    )
    return max(0.0, min(1.0, float(sev)))


def priority_from_severity(sev: float) -> str:
    if sev >= 0.70:
        return "HIGH"
    if sev >= 0.40:
        return "MEDIUM"
    return "LOW"


def build_recommendations(signal_summary: Dict[str, Any], max_items: int = 5) -> List[Dict[str, Any]]:
    """Map signal summary to a ranked list of intervention recommendations."""
    applicable: List[Dict[str, Any]] = []
    for item in CATALOG:
        try:
            if item["when"](signal_summary):
                applicable.append(item)
        except Exception:
            continue

    def score_item(it: Dict[str, Any]) -> int:
        p = it.get("priority")
        base = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(p, 1)
        return base

    applicable.sort(key=score_item, reverse=True)
    out: List[Dict[str, Any]] = []
    for it in applicable[: max_items]:
        out.append(
            {
                "title": it["title"],
                "stakeholder": it["stakeholder"],
                "priority": it["priority"],
                "expected_timeframe_days": it["timeframe_days"],
                "rationale": it["reason"](signal_summary),
                "signals_used": {
                    "fusion_risk": signal_summary.get("fusion_risk"),
                    "growth_risk": signal_summary.get("growth_risk"),
                    "learning_risk": signal_summary.get("learning_risk"),
                    "emotion_risk": signal_summary.get("emotion_risk"),
                },
            }
        )
    return out

