from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "models" / "fusion"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "fusion_pipeline.joblib"
META_PATH = ARTIFACT_DIR / "metadata.json"

MODEL_VERSION = "fusion-xgb-v1"

# label horizon
HORIZON_EVENTS = 2  # look ahead next 2 events (hackathon-friendly)
ESCALATION_DELTA = 0.15  # risk increase threshold
HIGH_RISK = 0.70  # crossing threshold counts as escalation

MIN_ROWS_TO_TRAIN = 30  # if fewer, we skip training and fallback in inference
