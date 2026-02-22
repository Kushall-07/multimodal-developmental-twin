from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEP, LEAKAGE_COLS


def load_uci_student_csv(path: str | None) -> pd.DataFrame:
    if path is None:
        raise ValueError("CSV path must be provided")
    df = pd.read_csv(path, sep=SEP)
    return df


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary at_risk label based on final grade G3 < 10."""
    df = df.copy()
    df["at_risk"] = (df["G3"] < 10).astype(int)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Stratified split: train/val/test ≈ 70/15/15."""
    y = df["at_risk"]
    X = df.drop(columns=["at_risk"])

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, stratify=y_tv, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns excluding leakage and label."""
    cols = [c for c in df.columns if c not in LEAKAGE_COLS + ["at_risk"]]
    return cols
