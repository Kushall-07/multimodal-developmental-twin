from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, String
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class TwinState(Base):
    __tablename__ = "twin_state"

    id = Column(String, primary_key=True)  # e.g., f"{child_id}:{timestamp}"
    child_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # store domain outputs
    growth_waz = Column(Float, nullable=True)
    growth_haz = Column(Float, nullable=True)
    growth_whz = Column(Float, nullable=True)
    growth_overall_risk = Column(Float, nullable=True)
    growth_confidence = Column(Float, nullable=True)

    # optional: store full JSON snapshot for extensibility
    snapshot = Column(JSON, nullable=True)
