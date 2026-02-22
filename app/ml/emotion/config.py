from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "emotion" / "fer2013"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

ARTIFACT_DIR = PROJECT_ROOT / "models" / "emotion"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "emotion_model.pt"
META_PATH = ARTIFACT_DIR / "metadata.json"

IMG_SIZE = 160  # best CPU compromise for FER faces
BATCH_SIZE = 32
NUM_WORKERS = 0  # windows safe
MODEL_VERSION = "emotion-4class-v1"

# 4-class target (angry, happy, sad, bored); bored = proxy for FER2013 neutral
TARGET = ["angry", "happy", "sad", "bored"]
MAP_TO_TARGET = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "bored",
    # disgust, fear, surprise ignored
}
CLASSES = TARGET  # infer uses this or metadata
