from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Sex = Literal["M", "F"]


@dataclass(frozen=True)
class WHOReference:
    """
    Holds WHO LMS reference tables loaded from CSV files in a directory.

    Expectation:
      - Each CSV has columns: sex, x, L, M, S
      - 'x' is the index variable:
          * age_months for WFA/LFA/HFA
          * length_cm or height_cm for WFL/WFH
      - sex values: 'M' or 'F'
    """

    wfa: pd.DataFrame | None = None
    hfa: pd.DataFrame | None = None
    wfh: pd.DataFrame | None = None


def _load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sex", "x", "L", "M", "S"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}. Required={sorted(required)}"
        )
    df = df.copy()
    df["sex"] = df["sex"].astype(str).str.upper()
    return df.sort_values(["sex", "x"]).reset_index(drop=True)


def load_who_reference(who_lms_dir: str) -> WHOReference:
    """
    Loads WHO LMS reference tables from a directory.
    Place your real WHO LMS CSVs here:

      data/raw/who/wfa_lms.csv  (x=age_months)
      data/raw/who/hfa_lms.csv  (x=age_months)
      data/raw/who/wfh_lms.csv  (x=height_or_length_cm)

    NOTE: We are not generating any synthetic values. These must be real published tables.
    """
    d = Path(who_lms_dir)
    if not d.exists():
        raise FileNotFoundError(f"WHO LMS directory not found: {d}")

    wfa_path = d / "wfa_lms.csv"
    hfa_path = d / "hfa_lms.csv"
    wfh_path = d / "wfh_lms.csv"

    ref = WHOReference(
        wfa=_load_table(wfa_path) if wfa_path.exists() else None,
        hfa=_load_table(hfa_path) if hfa_path.exists() else None,
        wfh=_load_table(wfh_path) if wfh_path.exists() else None,
    )
    if ref.wfa is None and ref.hfa is None and ref.wfh is None:
        raise FileNotFoundError(
            f"No WHO LMS CSVs found in {d}. Expected at least one of: wfa_lms.csv, hfa_lms.csv, wfh_lms.csv"
        )
    return ref


def lms_zscore(value: float, L: float, M: float, S: float) -> float:
    """
    WHO LMS z-score formula:
      If L != 0: Z = ((value/M)^L - 1) / (L*S)
      If L == 0: Z = ln(value/M) / S
    """
    if value <= 0 or M <= 0 or S <= 0:
        return float("nan")
    if np.isclose(L, 0.0):
        return float(np.log(value / M) / S)
    return float(((value / M) ** L - 1.0) / (L * S))


def _interp_lms(df: pd.DataFrame, sex: Sex, x: float) -> tuple[float, float, float]:
    """
    Linear interpolation over x within a sex-specific LMS table.
    """
    sdf = df[df["sex"] == sex]
    if sdf.empty:
        raise ValueError(f"No rows for sex={sex}")

    xs = sdf["x"].to_numpy(dtype=float)
    if x < xs.min() or x > xs.max():
        # Out-of-range: return NaN instead of inventing values
        return float("nan"), float("nan"), float("nan")

    Ls = sdf["L"].to_numpy(dtype=float)
    Ms = sdf["M"].to_numpy(dtype=float)
    Ss = sdf["S"].to_numpy(dtype=float)

    L = float(np.interp(x, xs, Ls))
    M = float(np.interp(x, xs, Ms))
    S = float(np.interp(x, xs, Ss))
    return L, M, S


def compute_waz(ref: WHOReference, sex: Sex, age_months: float, weight_kg: float) -> float:
    if ref.wfa is None:
        raise ValueError("WFA LMS table not loaded (wfa_lms.csv missing).")
    L, M, S = _interp_lms(ref.wfa, sex, age_months)
    return lms_zscore(weight_kg, L, M, S)


def compute_haz(ref: WHOReference, sex: Sex, age_months: float, height_cm: float) -> float:
    if ref.hfa is None:
        raise ValueError("HFA LMS table not loaded (hfa_lms.csv missing).")
    L, M, S = _interp_lms(ref.hfa, sex, age_months)
    return lms_zscore(height_cm, L, M, S)


def compute_whz(ref: WHOReference, sex: Sex, height_cm: float, weight_kg: float) -> float:
    if ref.wfh is None:
        raise ValueError("WFH LMS table not loaded (wfh_lms.csv missing).")
    L, M, S = _interp_lms(ref.wfh, sex, height_cm)
    return lms_zscore(weight_kg, L, M, S)
