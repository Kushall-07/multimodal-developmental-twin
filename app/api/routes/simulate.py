from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.twin_service import get_latest_events_snapshot, save_twin_event
from app.ml.fusion.infer import fallback_fusion
from app.ml.interventions.engine import compute_severity, priority_from_severity, build_recommendations


router = APIRouter(prefix="/simulate", tags=["simulate"])


class SimRequest(BaseModel):
    child_id: str
    overrides: Dict[str, float] = Field(default_factory=dict)
    max_items: int = 5
    save_to_twin: bool = True


def _extract_signals(snapshot: dict) -> dict:
    g = (snapshot.get("growth", {}).get("payload") or {})
    l = (snapshot.get("learning", {}).get("payload") or {})
    e = (snapshot.get("emotion", {}).get("payload") or {})
    f = (snapshot.get("fusion", {}).get("payload") or {})

    return {
        "growth_risk": float(g.get("overall_risk", 0.0)),
        "growth_conf": float(g.get("confidence", 0.0)),
        "growth_present": 1.0 if "growth" in snapshot else 0.0,
        "learning_risk": float(l.get("learning_risk", 0.0)),
        "learning_conf": float(l.get("confidence", 0.0)),
        "learning_present": 1.0 if "learning" in snapshot else 0.0,
        "emotion_risk": float(e.get("distress_risk", 0.0)),
        "emotion_conf": float(max((e.get("emotion_probs") or {}).values(), default=0.0)),
        "emotion_present": 1.0 if "emotion" in snapshot else 0.0,
        "fusion_risk": float(f.get("global_development_risk", 0.0)),
        "dominant_modality": f.get("dominant_modality", "unknown"),
    }


@router.post("")
def simulate(req: SimRequest) -> Dict[str, Any]:
    try:
        snap = get_latest_events_snapshot(req.child_id)
        s = _extract_signals(snap)

        # baseline fusion via fallback (risk, dominant, confidence)
        base_risk, base_dom, base_conf = fallback_fusion(s)

        # apply overrides (clamped to [0,1])
        sim = dict(s)
        for k, v in (req.overrides or {}).items():
            if k in {"growth_risk", "learning_risk", "emotion_risk"}:
                sim[k] = float(max(0.0, min(1.0, v)))

        sim_risk, sim_dom, sim_conf = fallback_fusion(sim)

        # severity & interventions before / after
        base_sev = compute_severity(
            {
                "fusion_risk": base_risk,
                "growth_risk": s["growth_risk"],
                "learning_risk": s["learning_risk"],
                "emotion_risk": s["emotion_risk"],
                "dominant_modality": base_dom,
            }
        )
        sim_sev = compute_severity(
            {
                "fusion_risk": sim_risk,
                "growth_risk": sim["growth_risk"],
                "learning_risk": sim["learning_risk"],
                "emotion_risk": sim["emotion_risk"],
                "dominant_modality": sim_dom,
            }
        )

        base_recs = build_recommendations(
            {
                "fusion_risk": base_risk,
                "growth_risk": s["growth_risk"],
                "learning_risk": s["learning_risk"],
                "emotion_risk": s["emotion_risk"],
            },
            max_items=req.max_items,
        )
        sim_recs = build_recommendations(
            {
                "fusion_risk": sim_risk,
                "growth_risk": sim["growth_risk"],
                "learning_risk": sim["learning_risk"],
                "emotion_risk": sim["emotion_risk"],
            },
            max_items=req.max_items,
        )

        # impact analysis for overridden risks
        impact: Dict[str, float] = {}
        for k in ["growth_risk", "learning_risk", "emotion_risk"]:
            if k in req.overrides:
                impact[k] = float(sim[k] - s[k])

        payload = {
            "baseline": {
                "global_risk": float(base_risk),
                "global_risk_pct": float(base_risk) * 100.0,
                "dominant_modality": base_dom,
                "confidence": float(base_conf),
                "severity_score": float(base_sev),
                "priority_level": priority_from_severity(base_sev),
                "recommendations": base_recs,
            },
            "simulated": {
                "global_risk": float(sim_risk),
                "global_risk_pct": float(sim_risk) * 100.0,
                "dominant_modality": sim_dom,
                "confidence": float(sim_conf),
                "severity_score": float(sim_sev),
                "priority_level": priority_from_severity(sim_sev),
                "recommendations": sim_recs,
            },
            "delta": {
                "risk_change": float(sim_risk - base_risk),
                "risk_change_pct_points": float((sim_risk - base_risk) * 100.0),
                "overrides_applied": req.overrides,
                "impact": impact,
            },
            "model_version": "simulate-fallback-v1",
        }

        if req.save_to_twin:
            save_twin_event(child_id=req.child_id, modality="simulation", payload=payload, timestamp=None)

        return {"child_id": req.child_id, **payload}

    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {ex}")

