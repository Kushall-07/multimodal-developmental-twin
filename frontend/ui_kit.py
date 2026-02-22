import streamlit as st
import pandas as pd
from datetime import datetime


def set_page():
    st.set_page_config(
        page_title="Child Development Digital Twin",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean modern spacing + subtle background
    st.markdown(
        """
        <style>
        .main { padding-top: 0.5rem; }
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); padding: 14px 14px 10px 14px; border-radius: 14px; }
        div[data-testid="stMetricValue"] { font-size: 28px; }
        .card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 16px; }
        .muted { opacity: 0.75; }
        .small { font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    level = (level or "UNKNOWN").upper()
    if level == "HIGH":
        return "🔴 HIGH"
    if level == "MEDIUM":
        return "🟠 MEDIUM"
    if level == "LOW":
        return "🟢 LOW"
    return "⚪ UNKNOWN"


def pct(x: float) -> float:
    try:
        return float(x) * 100.0
    except Exception:
        return 0.0


def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def gauge(label: str, value01: float, hint: str = ""):
    v = clamp01(value01)
    st.markdown(
        f"**{label}**  <span class='muted small'>{hint}</span>",
        unsafe_allow_html=True,
    )
    st.progress(v)
    st.caption(f"{v*100:.2f}%")


def card(title: str, body_md: str):
    st.markdown(
        f"<div class='card'><h4 style='margin:0 0 8px 0'>{title}</h4>{body_md}</div>",
        unsafe_allow_html=True,
    )


def to_df_events(events):
    """Normalize twin events into a DataFrame with parsed timestamps."""
    # Supports either list or {"value": [...]} formats
    if isinstance(events, dict) and "value" in events:
        events = events["value"]
    if not isinstance(events, list):
        return pd.DataFrame()

    rows = []
    for e in events:
        if not isinstance(e, dict):
            continue
        rows.append(
            {
                "timestamp": e.get("timestamp"),
                "modality": e.get("modality"),
                "payload": e.get("payload"),
            }
        )

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns and len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df
