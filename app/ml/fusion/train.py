"""
Train fusion XGBoost on real Twin timeline.
Label = escalation within next HORIZON_EVENTS (risk increase or crossing HIGH_RISK).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

from app.services.twin_service import get_all_events_grouped
from .config import (
    MODEL_PATH,
    META_PATH,
    MODEL_VERSION,
    HORIZON_EVENTS,
    ESCALATION_DELTA,
    HIGH_RISK,
    MIN_ROWS_TO_TRAIN,
)
from .features import build_fusion_features, max_current_risk


def _parse_ts(ts):
    """
    Returns timezone-aware datetime in UTC for any input:
    - '2026-02-21T18:06:49.644214' -> assume UTC
    - '2026-02-21T18:36:53.876194+00:00' -> keep
    """
    if ts is None:
        return datetime.min.replace(tzinfo=timezone.utc)

    if isinstance(ts, datetime):
        # make UTC-aware
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    s = str(ts).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)

    # If naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt


def build_training_table() -> pd.DataFrame:
    """
    Returns dataframe with columns:
      growth_risk, growth_conf, growth_present, ...
      y (escalation label)
    """
    rows = []
    by_child = get_all_events_grouped()

    for child_id, events in by_child.items():
        events = sorted(events, key=lambda e: _parse_ts(e["timestamp"]))
        latest = {}
        timeline = []

        for ev in events:
            latest[ev["modality"]] = {"payload": ev["payload"], "timestamp": ev["timestamp"]}
            snap = dict(latest)
            feats = build_fusion_features(snap)
            timeline.append({
                "child_id": child_id,
                "timestamp": ev["timestamp"],
                **feats,
                "current_max_risk": max_current_risk(feats),
            })

        if len(timeline) < (HORIZON_EVENTS + 1):
            continue

        for i in range(len(timeline) - HORIZON_EVENTS):
            cur = timeline[i]
            future_window = timeline[i + 1 : i + 1 + HORIZON_EVENTS]
            future_max = max(fw["current_max_risk"] for fw in future_window)

            y = 1 if (future_max >= HIGH_RISK or (future_max - cur["current_max_risk"]) >= ESCALATION_DELTA) else 0
            row = {k: cur[k] for k in cur.keys() if k not in ("child_id", "timestamp", "current_max_risk")}
            row["y"] = y
            rows.append(row)

    return pd.DataFrame(rows)


def train() -> None:
    df = build_training_table()
    if df.empty or len(df) < MIN_ROWS_TO_TRAIN:
        print(f"[WARN] Not enough fusion rows to train (have {len(df)}). Need {MIN_ROWS_TO_TRAIN}.")
        print("Run more growth/learning/emotion scores for multiple child_ids to build timeline, then retrain.")
        return

    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, probs)) if len(set(y_test)) > 1 else None,
        "f1": float(f1_score(y_test, pred)),
        "pos_rate": float(y.mean()),
        "rows": int(len(df)),
        "features": list(X.columns),
    }

    dump({"model": model, "features": list(X.columns), "model_version": MODEL_VERSION}, MODEL_PATH)
    META_PATH.write_text(json.dumps({"model_version": MODEL_VERSION, "metrics": metrics}, indent=2))

    print("✅ Fusion model training done")
    print("Saved pipeline to:", MODEL_PATH)
    print("Metrics:", metrics)


if __name__ == "__main__":
    train()
