from typing import Literal

from pydantic import BaseModel, Field


class GrowthInput(BaseModel):
    child_id: str = Field(..., description="Unique child identifier")
    sex: Literal["M", "F"]
    age_months: float = Field(..., ge=0)
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)


class GrowthOutput(BaseModel):
    child_id: str
    waz: float
    haz: float
    whz: float
    stunting_risk: float
    wasting_risk: float
    underweight_risk: float
    overall_risk: float
    confidence: float
