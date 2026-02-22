from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    title: str
    stakeholder: str
    priority: str
    expected_timeframe_days: int
    rationale: List[str]
    signals_used: Dict[str, Any]


class RecommendationsPayload(BaseModel):
    priority_level: str
    severity_score: float
    dominant_modality: str
    recommendations: List[RecommendationItem]
    model_version: str


class RecRequest(BaseModel):
    child_id: str
    max_items: int = 5

