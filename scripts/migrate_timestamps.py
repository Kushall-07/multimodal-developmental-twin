from __future__ import annotations

"""One-off migration script to normalize historical twin event timestamps.

Run manually:

    python -m scripts.migrate_timestamps

This will normalize any existing timestamps in the TwinState table to
UTC-aware ISO 8601 strings in the snapshot.server_timestamp field.
"""

from typing import Any

from app.db.session import SessionLocal
from app.db.models import TwinState
from app.utils.time import normalize_to_utc_iso


def migrate() -> None:
    db = SessionLocal()
    try:
        rows: list[Any] = db.query(TwinState).all()
        for r in rows:
            snap = r.snapshot or {}
            ts = snap.get("server_timestamp")
            # Fallback to created_at if no server_timestamp snapshot yet
            if not ts and getattr(r, "created_at", None) is not None:
                ts = r.created_at.isoformat()

            norm = normalize_to_utc_iso(ts)
            snap["server_timestamp"] = norm
            r.snapshot = snap

        db.commit()
        print("✅ Timestamp migration complete")
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
