import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ----------------------------
# Config
# ----------------------------
RAW_PATH = Path("data/raw/uci_student/student-mat.csv")
MODEL_DIR = Path("models/learning")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "uci_xgb_pipeline.joblib"
META_PATH = MODEL_DIR / "uci_meta.json"

# Early-warning feature set (no G1/G2/G3)
# Keep it reasonably sized so UI can ask for these fields.
FEATURE_COLS = [
    "sex",
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "famsup",
    "paid",
    "higher",
    "internet",
    "goout",
    "Dalc",
    "Walc",
    "health",
]

TARGET_COL = "G3"
RISK_THRESHOLD_G3 = 10  # at_risk = 1 if G3 < 10


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, sep=";")

    # Validate columns exist
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Label: at-risk for low final grade
    y = (df[TARGET_COL] < RISK_THRESHOLD_G3).astype(int)
    X = df[FEATURE_COLS].copy()

    # Identify categorical vs numeric
    categorical = [c for c in FEATURE_COLS if X[c].dtype == "object"]
    numeric = [c for c in FEATURE_COLS if c not in categorical]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    # Evaluate
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, pred, digits=4)

    print("\n=== UCI Learning Model Results ===")
    print(f"AUC: {auc:.4f}")
    print(report)

    # Save pipeline
    joblib.dump(pipe, MODEL_PATH)

    meta = {
        "dataset": str(RAW_PATH),
        "features": FEATURE_COLS,
        "target": "at_risk (G3 < 10)",
        "auc": float(auc),
        "risk_threshold_g3": RISK_THRESHOLD_G3,
        "notes": "Early-warning: excludes G1/G2 to avoid leakage.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model pipeline to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    main()
