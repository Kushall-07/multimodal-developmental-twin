from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uci_student" / "student-mat.csv"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "learning"
PIPELINE_PATH = ARTIFACT_DIR / "learning_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
SHAP_BG_PATH = ARTIFACT_DIR / "shap_background.npy"
FEATURE_DEFAULTS_PATH = ARTIFACT_DIR / "feature_defaults.json"

MODEL_VERSION = "learning-xgb-v1"

LABEL_COL = "G3"
LEAKAGE_COLS = ["G1", "G2", "G3"]  # exclude from features
SEP = ";"

# Risk thresholds (0..1 probability)
RISK_LEVELS = [
    (0.00, 0.33, "LOW"),
    (0.33, 0.66, "MEDIUM"),
    (0.66, 1.01, "HIGH"),
]

TOP_FACTORS_N = 8
RANDOM_STATE = 42
