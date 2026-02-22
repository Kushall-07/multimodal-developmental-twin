from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pipeline(pipeline: Any, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Any:
    return joblib.load(path)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def save_shap_background(X_bg, path: Path) -> None:
    ensure_dir(path.parent)
    np.save(path, X_bg)


def load_shap_background(path: Path):
    return np.load(path, allow_pickle=False)
