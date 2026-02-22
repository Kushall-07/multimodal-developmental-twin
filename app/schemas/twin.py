from typing import Any, Dict, List

from pydantic import BaseModel


class TwinUpdate(BaseModel):
    child_id: str
    growth_overall_risk: float
    snapshot: dict


class TwinUpdateResponse(BaseModel):
    id: int
    child_id: str
    growth_overall_risk: float
    snapshot: dict


class TwinEventOut(BaseModel):
    child_id: str
    modality: str
    payload: Dict[str, Any]
    timestamp: str


class TwinEventsResponse(BaseModel):
    value: List[TwinEventOut]
    Count: int


class LatestModalityState(BaseModel):
    timestamp: str
    payload: Dict[str, Any]


class TwinLatestSnapshotResponse(BaseModel):
    child_id: str
    snapshot: Dict[str, LatestModalityState]
