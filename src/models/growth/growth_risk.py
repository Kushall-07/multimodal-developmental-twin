from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GrowthScores:
    waz: float
    haz: float
    whz: float


@dataclass(frozen=True)
class GrowthRisk:
    stunting_risk: float
    wasting_risk: float
    underweight_risk: float
    overall_risk: float
    confidence: float


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def score_growth_risk(
    scores: GrowthScores,
    stunting_th: float = -2.0,
    wasting_th: float = -2.0,
    underweight_th: float = -2.0,
) -> GrowthRisk:
    """
    Converts WHO z-scores into calibrated-ish risk proxies.
    This is a baseline scoring layer; the *real* model will be learned using NFHS/DHS features.
    """
    # If z-score is NaN => we reduce confidence, not invent values.
    z_list = [scores.haz, scores.whz, scores.waz]
    valid = [z for z in z_list if np.isfinite(z)]
    confidence = 0.35 + 0.65 * (len(valid) / 3.0)

    # Map "distance below threshold" into a smooth risk.
    stunting = _sigmoid(-(scores.haz - stunting_th) * 2.0) if np.isfinite(scores.haz) else 0.5
    wasting = _sigmoid(-(scores.whz - wasting_th) * 2.0) if np.isfinite(scores.whz) else 0.5
    underwt = _sigmoid(-(scores.waz - underweight_th) * 2.0) if np.isfinite(scores.waz) else 0.5

    overall = float(np.mean([stunting, wasting, underwt]))
    return GrowthRisk(
        stunting_risk=stunting,
        wasting_risk=wasting,
        underweight_risk=underwt,
        overall_risk=overall,
        confidence=float(confidence),
    )
