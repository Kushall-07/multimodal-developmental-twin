from typing import Any, Dict

from pydantic import BaseModel


class TwinUpdate(BaseModel):
    child_id: str
    modality: str  # "growth", later "learning", "emotion", "engagement"
    payload: Dict[str, Any]


class TwinUpdateResponse(BaseModel):
    status: str
    record_id: str
