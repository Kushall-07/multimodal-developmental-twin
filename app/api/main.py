from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI

from app.schemas.growth import GrowthInput, GrowthOutput
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
