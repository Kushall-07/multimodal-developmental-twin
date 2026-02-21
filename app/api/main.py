from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List

import yaml
from fastapi import FastAPI

from app.schemas.growth import GrowthInput, GrowthOutput
from app.schemas.twin import TwinUpdate, TwinUpdateResponse
from app.db.session import init_db, SessionLocal
from app.db.models import TwinState
from app.services.twin_store import persist_twin_update
from sqlalchemy import desc
from src.models.growth.who_lms import (
    compute_haz,
    compute_waz,
    compute_whz,
    load_who_reference,
)
from src.models.growth.growth_risk import GrowthScores, score_growth_risk


def load_config() -> dict:
    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("configs/config.yaml not found")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
who_ref = load_who_reference(cfg["paths"]["who_lms_dir"])

app = FastAPI(title="Child Development Digital Twin API", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    """Initialize database schema on startup."""
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/growth/score", response_model=GrowthOutput)
def growth_score(inp: GrowthInput) -> GrowthOutput:
    waz = compute_waz(who_ref, inp.sex, inp.age_months, inp.weight_kg)
    haz = compute_haz(who_ref, inp.sex, inp.age_months, inp.height_cm)
    whz = compute_whz(who_ref, inp.sex, inp.height_cm, inp.weight_kg)

    scores = GrowthScores(waz=waz, haz=haz, whz=whz)
    risk = score_growth_risk(
        scores,
        stunting_th=cfg["growth"]["stunting_haz_threshold"],
        wasting_th=cfg["growth"]["wasting_whz_threshold"],
        underweight_th=cfg["growth"]["underweight_waz_threshold"],
    )

    payload = {
        "waz": waz,
        "haz": haz,
        "whz": whz,
        "stunting_risk": risk.stunting_risk,
        "wasting_risk": risk.wasting_risk,
        "underweight_risk": risk.underweight_risk,
        "overall_risk": risk.overall_risk,
        "confidence": risk.confidence,
    }

    _ = persist_twin_update(child_id=inp.child_id, modality="growth", payload=payload)

    return GrowthOutput(
        child_id=inp.child_id,
        waz=waz,
        haz=haz,
        whz=whz,
        stunting_risk=risk.stunting_risk,
        wasting_risk=risk.wasting_risk,
        underweight_risk=risk.underweight_risk,
        overall_risk=risk.overall_risk,
        confidence=risk.confidence,
    )


@app.post("/twin/update", response_model=TwinUpdateResponse)
def twin_update(inp: TwinUpdate) -> TwinUpdateResponse:
    """Persist a digital twin update event.

    For now, we special-case the "growth" modality and store its scalar outputs
    into dedicated columns while keeping the full payload snapshot as JSON.
    """
    record_id = f"{inp.child_id}:{datetime.utcnow().isoformat()}"

    db = SessionLocal()
    try:
        row = TwinState(
            id=record_id,
            child_id=inp.child_id,
            snapshot={"modality": inp.modality, "payload": inp.payload},
        )

        if inp.modality == "growth":
            row.growth_waz = inp.payload.get("waz")
            row.growth_haz = inp.payload.get("haz")
            row.growth_whz = inp.payload.get("whz")
            row.growth_overall_risk = inp.payload.get("overall_risk")
            row.growth_confidence = inp.payload.get("confidence")

        db.add(row)
        db.commit()
        return TwinUpdateResponse(status="ok", record_id=record_id)
    finally:
        db.close()


@app.get("/twin/history/{child_id}")
def twin_history(child_id: str, limit: int = 20):
    """Return recent twin_state records for a child (oldest -> newest)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(TwinState)
            .filter(TwinState.child_id == child_id)
            .order_by(desc(TwinState.created_at))
            .limit(limit)
            .all()
        )
        history = [
            {
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "growth_overall_risk": r.growth_overall_risk,
                "growth_confidence": r.growth_confidence,
            }
            for r in rows
        ][::-1]
        return history
    finally:
        db.close()


@app.get("/twin/latest/{child_id}")
def twin_latest(child_id: str):
    db = SessionLocal()
    try:
        row = (
            db.query(TwinState)
            .filter(TwinState.child_id == child_id)
            .order_by(desc(TwinState.created_at))
            .first()
        )
        if row is None:
            return {"child_id": child_id, "status": "not_found"}
        return {
            "child_id": child_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "growth": {
                "waz": row.growth_waz,
                "haz": row.growth_haz,
                "whz": row.growth_whz,
                "overall_risk": row.growth_overall_risk,
                "confidence": row.growth_confidence,
            },
            "snapshot": row.snapshot,
        }
    finally:
        db.close()
