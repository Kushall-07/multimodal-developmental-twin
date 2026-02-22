from __future__ import annotations

from pathlib import Path
from datetime import datetime

import yaml
from fastapi import FastAPI, HTTPException

from app.schemas.growth import GrowthInput, GrowthOutput
from app.schemas.twin import TwinUpdate, TwinUpdateResponse
from app.db.session import init_db, SessionLocal
from app.db.models import TwinState
from app.services.twin_service import save_twin_event
from app.api.routes.learning import router as learning_router
from app.api.routes.twin_events import router as twin_events_router
from app.api.routes.emotion import router as emotion_router
from app.api.routes.fusion import router as fusion_router
from app.api.routes.recommendations import router as rec_router
from app.api.routes.simulate import router as simulate_router
from app.api.routes.policy import router as policy_router
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
who_ref = None  # loaded lazily at startup

app = FastAPI(title="Child Development Digital Twin API", version="0.1.0")

app.include_router(learning_router)
app.include_router(twin_events_router)
app.include_router(emotion_router)
app.include_router(fusion_router)
app.include_router(rec_router)
app.include_router(simulate_router)
app.include_router(policy_router)


@app.on_event("startup")
def _startup() -> None:
    """Initialize database schema and load WHO reference on startup."""
    global who_ref
    init_db()

    try:
        who_ref = load_who_reference(cfg["paths"]["who_lms_dir"])
    except Exception as e:  # pragma: no cover - defensive for hackathon stability
        who_ref = None
        print(f"[WARN] WHO reference not loaded: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/growth/score", response_model=GrowthOutput)
def growth_score(inp: GrowthInput) -> GrowthOutput:
    if who_ref is None:
        # WHO reference not available; fail this endpoint gracefully without killing the API
        raise HTTPException(
            status_code=503,
            detail="WHO growth reference data not loaded. Check server configuration (who_lms_dir)",
        )

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

    # Log growth outcome into the twin timeline via the canonical event writer
    # timestamp=None => twin_service will normalize and stamp server/client times
    save_twin_event(child_id=inp.child_id, modality="growth", payload=payload, timestamp=None)

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
def twin_history(child_id: str, limit: int = 50):
    """Return recent twin events for a child (oldest -> newest).

    Each event is modality-agnostic and exposes the stored payload and timestamp
    from the TwinState.snapshot JSON.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(TwinState)
            .filter(TwinState.child_id == child_id)
            .order_by(desc(TwinState.created_at))
            .limit(limit)
            .all()
        )
        events = []
        for r in reversed(rows):  # oldest -> newest
            snap = r.snapshot or {}
            events.append(
                {
                    "child_id": r.child_id,
                    "modality": snap.get("modality"),
                    "payload": snap.get("payload"),
                    "timestamp": (snap.get("server_timestamp") or (
                        r.created_at.isoformat() if r.created_at else None
                    )),
                }
            )
        return events
    finally:
        db.close()


@app.get("/twin/latest/{child_id}")
def twin_latest(child_id: str):
    """Return a merged snapshot built from the latest event per modality."""
    db = SessionLocal()
    try:
        rows = (
            db.query(TwinState)
            .filter(TwinState.child_id == child_id)
            .order_by(TwinState.created_at.asc())
            .all()
        )
        if not rows:
            return {"child_id": child_id, "status": "not_found"}

        latest_by_modality = {}
        latest_ts = None
        for r in rows:
            snap = r.snapshot or {}
            mod = snap.get("modality") or "unknown"
            payload = snap.get("payload")
            ts = snap.get("server_timestamp") or (
                r.created_at.isoformat() if r.created_at else None
            )
            if payload is not None:
                latest_by_modality[mod] = payload
            if ts is not None:
                latest_ts = ts

        return {
            "child_id": child_id,
            "updated_at": latest_ts,
            "snapshot": latest_by_modality,
        }
    finally:
        db.close()
