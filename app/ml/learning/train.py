from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import (
    ARTIFACT_DIR,
    DATA_PATH,
    FEATURE_DEFAULTS_PATH,
    METADATA_PATH,
    MODEL_VERSION,
    PIPELINE_PATH,
    RANDOM_STATE,
    SHAP_BG_PATH,
)
from .data import build_label, get_feature_columns, load_uci_student_csv, split_data
from .pipeline import build_pipeline
from .artifacts import ensure_dir, save_json, save_pipeline, save_shap_background


def _infer_column_types(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols]
    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return cat_cols, num_cols


def run_train(calibrate: bool = True) -> None:
    t0 = time.time()
    ensure_dir(ARTIFACT_DIR)

    df = load_uci_student_csv(str(DATA_PATH))
    df = build_label(df)

    feature_cols = get_feature_columns(df)
    cat_cols, num_cols = _infer_column_types(df, feature_cols)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df[feature_cols + ["at_risk"]], random_state=RANDOM_STATE
    )

    pipe = build_pipeline(cat_cols, num_cols, calibrate=calibrate, random_state=RANDOM_STATE)
    pipe.fit(X_train[feature_cols], y_train)

    def eval_split(X, y, name: str) -> dict:
        p = pipe.predict_proba(X[feature_cols])[:, 1]
        pred = (p >= 0.5).astype(int)
        return {
            f"{name}_roc_auc": float(roc_auc_score(y, p)),
            f"{name}_pr_auc": float(average_precision_score(y, p)),
            f"{name}_f1": float(f1_score(y, pred)),
            f"{name}_pos_rate": float(np.mean(y)),
        }

    metrics: dict = {}
    metrics.update(eval_split(X_val, y_val, "val"))
    metrics.update(eval_split(X_test, y_test, "test"))

    # Compute dataset-based feature defaults from the train split
    defaults: dict = {}
    train_df = X_train[feature_cols].copy()
    for c in feature_cols:
        if train_df[c].dtype == "object":
            mode = train_df[c].mode(dropna=True)
            if not mode.empty:
                defaults[c] = mode.iloc[0]
        else:
            defaults[c] = float(train_df[c].median())
    FEATURE_DEFAULTS_PATH.write_text(json.dumps(defaults, indent=2))

    save_pipeline(pipe, PIPELINE_PATH)

    # Optional SHAP background in transformed space
    shap_bg_saved = False
    try:
        pre = pipe.named_steps["preprocess"]
        Xbg = X_train[feature_cols].sample(n=min(120, len(X_train)), random_state=RANDOM_STATE)
        Xbg_t = pre.transform(Xbg)
        save_shap_background(Xbg_t.astype(float), SHAP_BG_PATH)
        shap_bg_saved = True
    except Exception:
        shap_bg_saved = False

    meta = {
        "model_version": MODEL_VERSION,
        "dataset": "UCI Student Performance (student-mat.csv)",
        "label": "at_risk = (G3 < 10)",
        "excluded_leakage_cols": ["G1", "G2", "G3"],
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "calibrated": bool(calibrate),
        "metrics": metrics,
        "shap_background_saved": shap_bg_saved,
        "train_seconds": round(time.time() - t0, 2),
    }
    save_json(meta, METADATA_PATH)

    print("✅ Learning model training done")
    print("Saved pipeline to:", PIPELINE_PATH)
    print("Metrics:", metrics)


if __name__ == "__main__":
    run_train(calibrate=True)
