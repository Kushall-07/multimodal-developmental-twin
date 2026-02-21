from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


RAW_DIR = Path("data/raw/who")
OUT_DIR = Path("data/processed/who")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectedSheet:
    file: Path
    sheet: str
    kind: str  # "age" or "height"
    x_col: str
    l_col: str
    m_col: str
    s_col: str
    sex: str  # "M" or "F"


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _guess_sex_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    if "boy" in n or "boys" in n or "male" in n or "_m" in n:
        return "M"
    if "girl" in n or "girls" in n or "female" in n or "_f" in n:
        return "F"
    return None


def _find_lms_columns(cols) -> Optional[Tuple[str, str, str]]:
    # Try exact-ish matches first, then fallback to contains
    norm = {_norm(c): c for c in cols}
    for key in norm.keys():
        if key in ("l", "m", "s"):
            # just populates norm
            pass
    if all(k in norm for k in ("l", "m", "s")):
        return norm["l"], norm["m"], norm["s"]

    # contains-based
    def find_contains(token: str) -> Optional[str]:
        for k, orig in norm.items():
            if k == token or k.endswith(f"_{token}") or k.startswith(f"{token}_") or f"_{token}_" in k:
                return orig
        for k, orig in norm.items():
            if token in k:
                return orig
        return None

    l = find_contains("l")
    m = find_contains("m")
    s = find_contains("s")
    if l and m and s:
        return l, m, s
    return None


def _find_x_column(cols) -> Optional[Tuple[str, str]]:
    """Return (kind, x_col). kind="age" or "height"."""
    norm = {_norm(c): c for c in cols}
    # Common WHO naming patterns
    for key in (
        "age",
        "age_month",
        "age_months",
        "month",
        "months",
        "age_in_months",
        "day",
        "days",
        "age_in_days",
    ):
        if key in norm:
            return "age", norm[key]
    for key in ("height", "height_cm", "length", "length_cm", "len", "ht", "recumbent_length"):
        if key in norm:
            return "height", norm[key]

    # Fallback: partial matches
    for k, orig in norm.items():
        if "month" in k or "day" in k or k in ("age",):
            return "age", orig
    for k, orig in norm.items():
        if "height" in k or "length" in k or k in ("ht", "len"):
            return "height", orig

    return None


def detect_lms_sheets(xlsx_path: Path) -> list[DetectedSheet]:
    sex = _guess_sex_from_filename(xlsx_path.name)
    if sex is None:
        raise ValueError(
            f"Cannot infer sex from filename: {xlsx_path.name}. "
            f"Rename to include 'boys'/'girls' (or 'male'/'female')."
        )

    xl = pd.ExcelFile(xlsx_path)
    detected: list[DetectedSheet] = []

    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet_name=sheet, nrows=5)
        except Exception:
            continue

        if df is None or df.empty:
            continue

        cols = list(df.columns)
        xinfo = _find_x_column(cols)
        lms = _find_lms_columns(cols)

        if xinfo and lms:
            kind, x_col = xinfo
            l_col, m_col, s_col = lms
            detected.append(
                DetectedSheet(
                    file=xlsx_path,
                    sheet=sheet,
                    kind=kind,
                    x_col=x_col,
                    l_col=l_col,
                    m_col=m_col,
                    s_col=s_col,
                    sex=sex,
                )
            )

    return detected


def extract_lms(sheet: DetectedSheet) -> pd.DataFrame:
    df = pd.read_excel(sheet.file, sheet_name=sheet.sheet)

    # Keep only the needed cols + drop empty rows
    out = df[[sheet.x_col, sheet.l_col, sheet.m_col, sheet.s_col]].copy()
    out.columns = ["x", "L", "M", "S"]
    out["sex"] = sheet.sex

    # Clean numeric
    for c in ["x", "L", "M", "S"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["x", "L", "M", "S"]).sort_values("x").reset_index(drop=True)
    return out[["sex", "x", "L", "M", "S"]]


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_DIR.resolve()}")

    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {RAW_DIR.resolve()}")

    print("Found files:")
    for f in xlsx_files:
        print(" -", f.name)

    all_detected: list[DetectedSheet] = []
    for f in xlsx_files:
        det = detect_lms_sheets(f)
        if not det:
            print(f"\n[WARN] No LMS-like sheets detected in: {f.name}")
        else:
            all_detected.extend(det)

    if not all_detected:
        raise RuntimeError("No LMS sheets detected in any file. We need to inspect your Excel structure.")

    print("\nDetected LMS candidates:")
    for d in all_detected:
        print(
            f"- {d.file.name} | sheet='{d.sheet}' | kind={d.kind} | x={d.x_col} | "
            f"L={d.l_col} M={d.m_col} S={d.s_col} | sex={d.sex}"
        )

    def fname_lower(p: Path) -> str:
        return p.name.lower()

    hfa_rows = []
    wfa_rows = []
    wfh_rows = []

    for d in all_detected:
        n = fname_lower(d.file)

        # Height-for-age / Length-for-age
        if d.kind == "age" and (
            "height" in n or "length" in n or "lhfa" in n or "hfa" in n or "lfa" in n
        ):
            hfa_rows.append(extract_lms(d))
        # Weight-for-age
        elif d.kind == "age" and (
            "wfa" in n or "weight-for-age" in n or "wtfa" in n or "wgtfa" in n
        ):
            wfa_rows.append(extract_lms(d))
        # Weight-for-height/length
        elif d.kind == "height" and (
            "wfh" in n or "weight-for-length" in n or "weight-for-height" in n or "wfl" in n
        ):
            wfh_rows.append(extract_lms(d))

    # Fallback by kind only
    if not hfa_rows:
        hfa_rows = [extract_lms(d) for d in all_detected if d.kind == "age"]
    if not wfh_rows:
        wfh_rows = [extract_lms(d) for d in all_detected if d.kind == "height"]

    if hfa_rows:
        hfa = pd.concat(hfa_rows, ignore_index=True).drop_duplicates()
        hfa.to_csv(OUT_DIR / "hfa_lms.csv", index=False)
        print(f"\nSaved: {OUT_DIR/'hfa_lms.csv'}  rows={len(hfa)}")

    if wfa_rows:
        wfa = pd.concat(wfa_rows, ignore_index=True).drop_duplicates()
        wfa.to_csv(OUT_DIR / "wfa_lms.csv", index=False)
        print(f"Saved: {OUT_DIR/'wfa_lms.csv'}  rows={len(wfa)}")
    else:
        print("\n[NOTE] wfa_lms.csv not produced yet (weight-for-age files not detected).")

    if wfh_rows:
        wfh = pd.concat(wfh_rows, ignore_index=True).drop_duplicates()
        wfh.to_csv(OUT_DIR / "wfh_lms.csv", index=False)
        print(f"Saved: {OUT_DIR/'wfh_lms.csv'}  rows={len(wfh)}")

    print("\nDONE. Next: point the API loader to data/processed/who/*.csv")


if __name__ == "__main__":
    main()
