import json
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from ui_kit import set_page, gauge, risk_badge, card, to_df_events, clamp01


set_page()

st.markdown(
    """
    <style>
    /* Reduce whitespace and make layout tighter */
    .block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
    section[data-testid="stSidebar"] { width: 320px !important; }

    /* Make buttons same height and align nicely */
    div.stButton > button {
      height: 44px;
      border-radius: 12px;
      font-weight: 600;
    }

    /* Compact subheaders */
    h2, h3 { margin-top: 0.4rem; }

    /* Card look */
    .card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8001")
# Global child ID for tabs that don't have their own form (Learning, Emotion, Interventions, Twin Timeline)
if "child_id" not in st.session_state:
    st.session_state["child_id"] = "demo_001"
_active = st.sidebar.text_input("Active Child ID", value=st.session_state.get("child_id", "demo_001"), key="active_child_id_global")
st.session_state["child_id"] = _active.strip() or "demo_001"

st.title("AI Multimodal Child Development Digital Twin — Prototype")
st.caption("Growth (WHO z-scores) + Learning (UCI Student) + Digital Twin persistence")

# High-level navigation (tabs used for higher-level views)
tabs = st.tabs([
    "Growth",
    "Learning",
    "Emotion",
    "Fusion",
    "Interventions",
    "Twin Timeline",
])

# -----------------------
# Helpers
# -----------------------
def post_growth_score():
    payload = {
        "child_id": child_id,
        "sex": sex,
        "age_months": float(age_months),
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
    }
    r = requests.post(f"{API_BASE}/growth/score", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_latest_twin():
    r = requests.get(f"{API_BASE}/twin/latest/{child_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_latest_twin_snapshot(child_id_value: str) -> dict:
    """Fetch latest multimodal twin snapshot using the events/latest endpoint."""
    r = requests.get(f"{API_BASE}/twin/events/latest/{child_id_value}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_get(path: str):
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET failed: {url}\n{e}")
        return None


def api_post(path: str, payload: dict, files=None):
    url = f"{API_BASE}{path}"
    try:
        if files:
            r = requests.post(url, data=payload, files=files, timeout=60)
        else:
            r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"POST failed: {url}\n{e}")
        return None


def normalize_events(resp):
    # API returns either list[events] OR {"value": list, "Count": n}
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict) and isinstance(resp.get("value"), list):
        return resp["value"]
    return []


def ensure_child_id():
    child_id = (st.session_state.get("child_id") or "").strip()
    if not child_id:
        st.warning("Set Active Child ID in the sidebar to continue.")
        st.stop()
    return child_id


def fetch_latest_snapshot():
    """Fetch latest twin snapshot for active child and store in session."""
    cid = (st.session_state.get("child_id") or "").strip()
    if not cid:
        return
    raw = api_get(f"/twin/events/latest/{cid}")
    if raw is not None:
        st.session_state["twin_latest"] = raw
        st.session_state["twin_latest_loaded_at"] = datetime.now().strftime("%H:%M:%S")


def risk_level_from_pct(pct: float) -> str:
    if pct >= 70:
        return "HIGH"
    if pct >= 40:
        return "MEDIUM"
    return "LOW"


def get_history():
    r = requests.get(f"{API_BASE}/twin/history/{child_id}?limit=20", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_twin_events(child_id: str) -> list[dict]:
    raw = api_get(f"/twin/events/{child_id}")
    return normalize_events(raw)


def learning_timeline_view(child_id_value: str) -> None:
    """Render a learning risk timeline and event table for a child."""
    events = fetch_twin_events(child_id_value)
    st.write("events type:", type(events).__name__, "len:", len(events))

    if not events:
        st.info("No events found for this child yet. Run Growth Score or Learning Score once.")
        return

    # Combined timeline (learning + growth)
    rows = []
    for e in events:
        if not isinstance(e, dict):
            continue
        payload = e.get("payload", {}) or {}
        rows.append({
            "timestamp": e.get("timestamp"),
            "modality": e.get("modality"),
            "risk_level": payload.get("risk_level"),
            "learning_risk_pct": payload.get("learning_risk_pct"),
            "growth_overall_risk": payload.get("overall_risk"),
            "confidence": payload.get("confidence"),
            "model_version": payload.get("model_version"),
        })
    if not rows:
        st.info("No valid events found for this child yet.")
        return
    df = pd.DataFrame(rows)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp_dt")

    st.subheader("All Events (Twin Timeline)")
    st.dataframe(df[["timestamp", "modality", "risk_level", "learning_risk_pct", "growth_overall_risk", "confidence", "model_version"]], use_container_width=True)

    if (df["modality"] == "learning").any():
        st.subheader("Learning Risk Trend")
        st.line_chart(df[df["modality"] == "learning"].set_index("timestamp_dt")["learning_risk_pct"])

    if (df["modality"] == "growth").any():
        st.subheader("Growth Overall Risk Trend")
        st.line_chart(df[df["modality"] == "growth"].set_index("timestamp_dt")["growth_overall_risk"])

    learning_events = [e for e in events if isinstance(e, dict) and e.get("modality") == "learning"]
    if not learning_events:
        st.info("No learning events found yet. Run Learning Score once.")
        return

    rows_learning = []
    for e in learning_events:
        ts = e.get("timestamp")
        payload = e.get("payload", {}) or {}
        imputed = payload.get("imputed_fields", [])
        rows_learning.append(
            {
                "timestamp": ts,
                "learning_risk": payload.get("learning_risk"),
                "learning_risk_pct": payload.get("learning_risk_pct"),
                "risk_level": payload.get("risk_level"),
                "confidence": payload.get("confidence"),
                "model_version": payload.get("model_version"),
                "imputed_fields_count": len(imputed) if isinstance(imputed, list) else None,
            }
        )

    df_learning = pd.DataFrame(rows_learning)
    df_learning["timestamp_dt"] = pd.to_datetime(df_learning["timestamp"], errors="coerce", utc=True)
    df_learning = df_learning.sort_values("timestamp_dt")

    st.subheader("Learning Events")
    show_cols = [
        "timestamp",
        "learning_risk_pct",
        "risk_level",
        "confidence",
        "model_version",
        "imputed_fields_count",
    ]
    st.dataframe(df_learning[show_cols], use_container_width=True)

    # Latest payload for transparency
    latest = sorted(learning_events, key=lambda x: x.get("timestamp", ""))[-1]
    st.subheader("Latest Learning Payload")
    st.json(latest)


def show_twin_timeline(child_id_value: str) -> None:
    """Show a combined twin timeline across modalities using /twin/events."""
    try:
        r = requests.get(f"{API_BASE}/twin/events/{child_id_value}", timeout=30)
        r.raise_for_status()
        data = r.json()

        # Support both list and potential {"value": [...]} wrappers
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            events = data.get("value", []) if "value" in data else []
        else:
            events = []

        if not events:
            st.info("No events found for this child yet.")
            return

        rows = []
        for e in events:
            payload = e.get("payload", {}) or {}
            rows.append(
                {
                    "timestamp": e.get("timestamp"),
                    "modality": e.get("modality"),
                    "risk_level": payload.get("risk_level"),
                    "learning_risk_pct": payload.get("learning_risk_pct"),
                    "growth_overall_risk": payload.get("overall_risk")
                    or payload.get("growth_overall_risk"),
                    "confidence": payload.get("confidence"),
                    "model_version": payload.get("model_version"),
                }
            )

        df = pd.DataFrame(rows)
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp_dt")

        st.subheader("Twin Timeline (All Modalities)")
        st.dataframe(
            df[
                [
                    "timestamp",
                    "modality",
                    "risk_level",
                    "learning_risk_pct",
                    "growth_overall_risk",
                    "confidence",
                    "model_version",
                ]
            ],
            use_container_width=True,
        )

        if (df["modality"] == "learning").any():
            st.subheader("Learning Risk Trend")
            df_l = df[df["modality"] == "learning"].set_index("timestamp_dt")
            if "learning_risk_pct" in df_l.columns:
                st.line_chart(df_l["learning_risk_pct"])

        with st.expander("Raw events JSON"):
            st.json(events)

    except Exception as e:
        st.error(f"Loading history failed: {e}")


def fusion_view():
    st.subheader("Fusion Intelligence (Global Development Risk)")
    child_id = st.session_state.get("child_id", "").strip()
    st.caption(f"Active Child: {child_id}")
    if not child_id:
        st.warning("Set Active Child ID in sidebar.")
        st.stop()

    if st.button("Load Latest Snapshot", key="fusion_load_snapshot", use_container_width=True):
        raw = api_get(f"/twin/events/latest/{child_id}")
        if raw is not None:
            st.session_state["twin_latest"] = raw
            st.session_state["fusion_snapshot"] = raw
            st.session_state["twin_latest_loaded_at"] = datetime.now().strftime("%H:%M:%S")
            st.success("Latest snapshot loaded.")

    child_id_fusion = st.text_input("Child ID (override)", value=child_id, key="fusion_child_id")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Run Fusion Score", key="run_fusion"):
            out = api_post("/fusion/score", {"child_id": child_id_fusion})
            if out is not None:
                st.success("Fusion score computed and saved to Twin.")
                st.session_state["fusion_last"] = out

    with colB:
        if st.button("Fetch Latest Snapshot", key="fetch_latest_snapshot"):
            raw = api_get(f"/twin/events/latest/{child_id_fusion}")
            if raw is not None:
                st.session_state["fusion_snapshot"] = raw
                st.session_state["twin_latest"] = raw
                st.success("Latest snapshot fetched.")

    latest = st.session_state.get("twin_latest")
    if latest:
        st.markdown("### Latest Snapshot")
        st.json(latest.get("snapshot", latest))
    else:
        st.info("Click **Load Latest Snapshot** or **Fetch Latest Snapshot** to load snapshot.")

    # ---- Show latest fusion output ----
    out = st.session_state.get("fusion_last")
    if out:
        pct = float(out.get("global_development_risk_pct", 0.0))
        lvl = risk_level_from_pct(pct)

        c1, c2, c3 = st.columns(3)
        c1.metric("Global Risk (%)", f"{pct:.2f}")
        c2.metric("Risk Level", lvl)
        c3.metric("Dominant Modality", out.get("dominant_modality", "unknown"))

        st.write("Explainability method:", out.get("explainability_method", "unknown"))

        contrib = out.get("contributions", {}) or {}
        if contrib:
            dfc = pd.DataFrame(
                [{"modality": k, "contribution": float(v)} for k, v in contrib.items()]
            ).sort_values("contribution", ascending=False)
            st.caption("Modality Contribution (normalized)")
            st.bar_chart(dfc.set_index("modality"))

        st.caption("Fusion JSON")
        st.json(out)

    st.divider()

    # ---- Trend line from timeline events ----
    st.subheader("Fusion Risk Trend (Timeline)")
    try:
        events = normalize_events(api_get(f"/twin/events/{child_id_fusion}"))

        fusion_events = [e for e in events if isinstance(e, dict) and e.get("modality") == "fusion"]
        if not fusion_events:
            st.info("No fusion events yet. Click 'Run Fusion Score' to generate.")
            return

        rows = []
        for e in fusion_events:
            ts = e.get("timestamp")
            payload = e.get("payload", {}) or {}
            rows.append({
                "timestamp": ts,
                "risk_pct": float(payload.get("global_development_risk_pct", 0.0)),
                "dominant": payload.get("dominant_modality", "unknown"),
            })

        df = pd.DataFrame(rows)

        # parse timestamp safely
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        st.line_chart(df.set_index("timestamp")[["risk_pct"]])
        st.dataframe(df.tail(20), use_container_width=True)

    except Exception as e:
        st.error(f"Loading fusion history failed: {e}")

    st.divider()

    # ---- Optional: show latest modality risks from snapshot ----
    st.subheader("Latest Modality Signals")
    snap = st.session_state.get("fusion_snapshot")
    if snap and isinstance(snap, dict):
        snapshot = snap.get("snapshot", {}) or {}

        def _get_payload(mod):
            m = snapshot.get(mod)
            if isinstance(m, dict):
                return m.get("payload", {}) or {}
            return {}

        g = _get_payload("growth")
        l = _get_payload("learning")
        em = _get_payload("emotion")

        c1, c2, c3 = st.columns(3)
        c1.metric("Growth overall_risk", f"{float(g.get('overall_risk', 0.0)):.3f}")
        c2.metric("Learning learning_risk", f"{float(l.get('learning_risk', 0.0)):.3f}")
        c3.metric("Emotion distress_risk", f"{float(em.get('distress_risk', 0.0)):.3f}")

        st.caption("Snapshot JSON")
        st.json(snapshot)


def what_if_view() -> None:
    """What-If Simulator for policy / care planning based on fusion engine."""
    st.subheader("What-If Simulator (Policy / Care Planning)")
    child_id = st.session_state.get("child_id", "").strip()
    st.caption(f"Active Child: {child_id}")
    if not child_id:
        st.warning("Set Active Child ID in sidebar.")
        st.stop()

    if st.button("Load Latest Snapshot", key="whatif_load_snapshot", use_container_width=True):
        raw = api_get(f"/twin/events/latest/{child_id}")
        if raw is not None:
            st.session_state["twin_latest"] = raw
            st.session_state["twin_latest_loaded_at"] = datetime.now().strftime("%H:%M:%S")
            st.success("Latest snapshot loaded.")

    latest = st.session_state.get("twin_latest")
    if latest:
        st.markdown("### Latest Snapshot")
        st.json(latest.get("snapshot", latest))
    else:
        st.info("Click **Load Latest Snapshot** to load snapshot.")

    child_id_w = st.text_input("Child ID (for simulation)", value=child_id, key="whatif_child_id")

    st.caption(
        "Move sliders to simulate interventions. This uses the fusion engine with overrides "
        "for emotion, learning, and growth risk."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        emo = st.slider("Emotion risk override (distress_risk)", 0.0, 1.0, 0.25, 0.01)
    with col2:
        learn = st.slider("Learning risk override (learning_risk)", 0.0, 1.0, 0.40, 0.01)
    with col3:
        grow = st.slider("Growth risk override (overall_risk)", 0.0, 1.0, 0.10, 0.01)

    colA, colB = st.columns([1, 1])
    with colA:
        save_to_twin = st.checkbox("Save simulation to Twin timeline", value=True)
    with colB:
        max_items = st.number_input(
            "Max recommendations", min_value=1, max_value=10, value=5, step=1
        )

    if st.button("Run Simulation", key="run_sim"):
        out = api_post(
            "/simulate",
            {
                "child_id": child_id_w,
                "overrides": {
                    "emotion_risk": float(emo),
                    "learning_risk": float(learn),
                    "growth_risk": float(grow),
                },
                "max_items": int(max_items),
                "save_to_twin": bool(save_to_twin),
            },
        )
        if out is not None:
            st.session_state["whatif_last"] = out
            st.success("Simulation complete.")

    out = st.session_state.get("whatif_last")
    if not out:
        st.info("Click Run Simulation to generate baseline vs simulated comparison.")
        return

    base = out.get("baseline", {}) or {}
    sim = out.get("simulated", {}) or {}
    delta = out.get("delta", {}) or {}

    st.divider()
    st.subheader("Baseline vs Simulated Risk")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Baseline Global Risk (%)",
        f"{float(base.get('global_risk_pct', 0.0)):.2f}",
    )
    c2.metric(
        "Simulated Global Risk (%)",
        f"{float(sim.get('global_risk_pct', 0.0)):.2f}",
    )
    c3.metric(
        "Delta Risk (percentage points)",
        f"{float(delta.get('risk_change_pct_points', 0.0)):+.2f}",
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Baseline Priority", str(base.get("priority_level", "NA")))
    c5.metric("Simulated Priority", str(sim.get("priority_level", "NA")))
    c6.metric(
        "Dominant Modality (Simulated)", str(sim.get("dominant_modality", "NA"))
    )

    st.divider()
    st.subheader("Recommendations: Before vs After")

    left, right = st.columns(2)
    with left:
        st.markdown("### Baseline")
        recs = base.get("recommendations", []) or []
        if not recs:
            st.info("No baseline recommendations.")
        else:
            for r in recs:
                st.markdown(
                    f"**{r.get('title','')}**  \n"
                    f"Stakeholder: {r.get('stakeholder','')}  \n"
                    f"Priority: {r.get('priority','')}  \n"
                    f"Timeframe: {r.get('expected_timeframe_days','')} days"
                )
                for line in (r.get("rationale", []) or []):
                    st.write(f"- {line}")
                st.write("---")

    with right:
        st.markdown("### Simulated")
        recs = sim.get("recommendations", []) or []
        if not recs:
            st.info("No simulated recommendations.")
        else:
            for r in recs:
                st.markdown(
                    f"**{r.get('title','')}**  \n"
                    f"Stakeholder: {r.get('stakeholder','')}  \n"
                    f"Priority: {r.get('priority','')}  \n"
                    f"Timeframe: {r.get('expected_timeframe_days','')} days"
                )
                for line in (r.get("rationale", []) or []):
                    st.write(f"- {line}")
                st.write("---")

    st.divider()
    st.subheader("Simulation JSON (debug)")
    st.json(out)


def policy_view() -> None:
    """Policy View: top-risk children based on latest fusion global risk."""
    st.subheader("Policy View (Population Risk Monitor)")
    child_id = st.session_state.get("child_id", "").strip()
    st.caption(f"Active Child: {child_id}")

    if st.button("Load Latest Snapshot", key="policy_load_snapshot", use_container_width=True):
        raw = api_get(f"/twin/events/latest/{child_id}") if child_id else None
        if raw is not None:
            st.session_state["twin_latest"] = raw
            st.session_state["twin_latest_loaded_at"] = datetime.now().strftime("%H:%M:%S")
            st.success("Latest snapshot loaded.")

    latest = st.session_state.get("twin_latest")
    if latest:
        st.markdown("### Latest Snapshot")
        st.json(latest.get("snapshot", latest))
    else:
        st.info("Click **Load Latest Snapshot** to load snapshot for active child.")

    limit = st.slider("Show top N children", 5, 50, 10, 1)

    if st.button("Load Policy Dashboard", key="load_policy"):
        data = api_get(f"/policy/top?limit={limit}")
        if data is not None:
            st.session_state["policy_rows"] = data.get("rows", [])

    rows = st.session_state.get("policy_rows", [])
    if not rows:
        st.info("Click Load Policy Dashboard.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    try:
        st.bar_chart(df.set_index("child_id")["global_risk_pct"])
    except Exception:
        pass


def twin_timeline_tab() -> None:
    """Twin Timeline tab: load and show events for active child."""
    child_id = st.session_state.get("child_id", "").strip() or "demo_001"
    st.subheader("Twin Timeline (All Modalities)")
    st.caption(f"Active Child: {child_id}")

    if st.button("Load Timeline", key="twin_timeline_load", use_container_width=True):
        raw = api_get(f"/twin/events/{child_id}")
        st.session_state["twin_events"] = normalize_events(raw)

    events = st.session_state.get("twin_events", [])
    if not events:
        st.info("No timeline loaded yet. Click **Load Timeline**.")
        return

    st.success(f"Loaded {len(events)} events")
    st.json(events[-1])  # last event quick view

    # Optional: full list in expander
    with st.expander("All events (JSON)"):
        st.json(events)


def learning_tab_ui():
    child_id = ensure_child_id()
    st.subheader("Learning Risk (UCI Student)")
    st.caption(f"Active Child ID: {child_id}")

    c1, c2, c3 = st.columns([1, 1, 1])
    run = c1.button("Run Learning Score", use_container_width=True, key="learning_run_score_btn_unique")
    load_snap = c2.button(
        "Load Latest Snapshot",
        use_container_width=True,
        key="learning_load_snapshot_btn_unique"
    )
    load_events = c3.button("Load Learning Events", use_container_width=True, key="learning_load_events_btn_unique")

    # Minimal form (matches your API which imputes missing fields)
    with st.expander("Learning Inputs", expanded=True):
        f1, f2, f3 = st.columns(3)
        age = f1.number_input("age", 10, 22, 16)
        sex = f2.selectbox("sex", ["M", "F"])
        address = f3.selectbox("address", ["U", "R"])

        s1, s2, s3 = st.columns(3)
        studytime = s1.number_input("studytime", 1, 4, 2)
        failures = s2.number_input("failures", 0, 4, 1)
        absences = s3.number_input("absences", 0, 100, 8)

        b1, b2, b3 = st.columns(3)
        schoolsup = b1.selectbox("schoolsup", ["no", "yes"])
        famsup = b2.selectbox("famsup", ["no", "yes"])
        internet = b3.selectbox("internet", ["no", "yes"])

    if run:
        payload = {
            "child_id": child_id,
            "features": {
                "age": age, "sex": sex, "address": address,
                "studytime": studytime, "failures": failures, "absences": absences,
                "schoolsup": schoolsup, "famsup": famsup, "internet": internet,
            }
        }
        out = api_post("/learning/score", payload)
        if out:
            st.session_state["learning_last"] = out
            st.success("Learning score saved to Twin.")
            st.json(out)

    if load_snap:
        snap = api_get(f"/twin/events/latest/{child_id}")
        if snap:
            st.session_state["twin_latest"] = snap

    latest = st.session_state.get("twin_latest")
    st.markdown("### Latest Learning Snapshot")
    if latest and isinstance(latest, dict):
        learning = (latest.get("snapshot", {}) or {}).get("learning")
        if learning:
            st.json(learning)
        else:
            st.info("No learning snapshot found yet. Run Learning Score once.")
    else:
        st.info("Click Load Latest Snapshot to view learning snapshot.")

    if load_events:
        raw = api_get(f"/twin/events/{child_id}")
        st.session_state["twin_events"] = normalize_events(raw) or []

    events = st.session_state.get("twin_events") or []
    learning_events = [e for e in events if isinstance(e, dict) and e.get("modality") == "learning"]

    st.markdown("### Learning Events (Timeline)")
    if not learning_events:
        st.info("No learning events loaded. Click Load Learning Events.")
    else:
        st.write(f"Loaded {len(learning_events)} learning events.")
        st.json(learning_events[-1])


def emotion_tab_ui():
    child_id = ensure_child_id()
    st.subheader("Emotion Analysis")
    st.caption(f"Active Child ID: {child_id}")

    col1, col2 = st.columns(2)

    with col1:
        upload_img = st.file_uploader(
            "Upload Image",
            type=["jpg", "png", "jpeg"],
            key="emotion_upload_image"
        )

    with col2:
        load_snap = st.button(
            "Load Latest Snapshot",
            use_container_width=True,
            key="emotion_load_snapshot_btn_unique"
        )

    if load_snap:
        fetch_latest_snapshot()

    score = st.button("Score Uploaded Face Image", use_container_width=True, key="emotion_score_btn_unique")

    if score:
        if upload_img is None:
            st.warning("Upload an image first.")
        else:
            files = {"image": (upload_img.name, upload_img.getvalue(), upload_img.type)}
            data = {"child_id": child_id}
            out = api_post("/emotion/score", data, files=files)
            if out:
                st.session_state["emotion_last"] = out
                st.success("Emotion score saved to Twin.")
                st.json(out)

    latest = st.session_state.get("twin_latest")
    st.markdown("### Latest Emotion Snapshot")
    if latest and isinstance(latest, dict):
        emo = (latest.get("snapshot", {}) or {}).get("emotion")
        if emo:
            st.json(emo)
        else:
            st.info("No emotion snapshot found yet. Score an image once.")
    else:
        st.info("Click Load Latest Snapshot to view emotion snapshot.")


def interventions_tab_ui():
    child_id = ensure_child_id()
    st.subheader("Interventions")
    st.caption(f"Active Child ID: {child_id}")

    c1, c2 = st.columns([1, 1])
    run = c1.button("Generate Recommendations", use_container_width=True, key="intervention_generate_btn_unique")
    load_snap = c2.button(
        "Load Latest Snapshot",
        use_container_width=True,
        key="intervention_load_snapshot_btn_unique"
    )

    max_items = st.slider("Max items", 1, 10, 5)

    if run:
        out = api_post("/recommendations", {"child_id": child_id, "max_items": max_items})
        if out:
            st.session_state["reco_last"] = out
            st.success("Recommendations saved to Twin.")
            # Render nicely
            st.metric("Priority Level", out.get("priority_level", "—"))
            st.metric("Severity Score", f"{out.get('severity_score', 0):.3f}")
            st.metric("Dominant Modality", out.get("dominant_modality", "—"))

            recs = out.get("recommendations", []) or []
            for r in recs:
                with st.expander(f"{r.get('priority','—')} — {r.get('title','Recommendation')}"):
                    st.write("Stakeholder:", r.get("stakeholder", "—"))
                    st.write("Timeframe (days):", r.get("expected_timeframe_days", "—"))
                    st.write("Rationale:")
                    for line in (r.get("rationale", []) or []):
                        st.write("-", line)

            with st.expander("Raw JSON"):
                st.json(out)

    if load_snap:
        snap = api_get(f"/twin/events/latest/{child_id}")
        if snap:
            st.session_state["twin_latest"] = snap

    latest = st.session_state.get("twin_latest")
    st.markdown("### Latest Recommendations Snapshot")
    if latest and isinstance(latest, dict):
        rec = (latest.get("snapshot", {}) or {}).get("recommendations")
        if rec:
            st.json(rec)
        else:
            st.info("No recommendations snapshot found yet. Click Generate Recommendations.")
    else:
        st.info("Click Load Latest Snapshot to view recommendations snapshot.")


def growth_tab_ui() -> None:
    """Compact, operator-style Growth tab with two-column layout and minimal scrolling."""
    st.subheader("Growth & Nutrition (0–60 months)")
    st.caption("Enter basic anthropometrics to get z-scores, risk, and save to the Twin.")

    # Sidebar: child id only (simple NGO flow)
    with st.sidebar:
        st.header("Child Profile")
        child_id = st.text_input("Child ID", value="demo_001", key="child_id_main")
        st.caption("Use the same Child ID across modules to build the timeline.")
        st.divider()

    # Main two-column layout
    left, right = st.columns([1.05, 1.25], gap="large")

    # ---------- LEFT: Inputs + Actions ----------
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 1) Input")

        with st.form("growth_form", clear_on_submit=False):
            c1, c2, c3 = st.columns([1, 1, 1])
            sex = c1.selectbox("Sex", ["M", "F"], index=0)
            age_months = c2.number_input(
                "Age (months)", min_value=0.0, max_value=60.0, value=12.0, step=1.0
            )
            c3.write("")  # spacer / future field

            h, w = st.columns(2)
            height_cm = h.number_input(
                "Height (cm)",
                min_value=30.0,
                max_value=130.0,
                value=75.0,
                step=0.5,
            )
            weight_kg = w.number_input(
                "Weight (kg)",
                min_value=1.0,
                max_value=40.0,
                value=9.0,
                step=0.1,
            )

            # Buttons aligned (same row)
            b1, b2, b3 = st.columns([1, 1, 1])
            run = b1.form_submit_button("Run Growth Score", use_container_width=True, key="growth_run_score_btn_unique")
            fetch = b2.form_submit_button("Fetch Latest Twin", use_container_width=True, key="growth_fetch_latest_btn_unique")
            timeline = b3.form_submit_button("Show Timeline", use_container_width=True, key="growth_show_timeline_btn_unique")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RIGHT: Output Cards ----------
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 2) Output (Growth Engine)")
        st.caption("Clear summary for health worker / teacher / parent.")

        if "growth_last" not in st.session_state:
            st.session_state["growth_last"] = None
        if "twin_latest" not in st.session_state:
            st.session_state["twin_latest"] = None
        if "twin_events" not in st.session_state:
            st.session_state["twin_events"] = None

        # Handle actions
        if run:
            payload = {
                "child_id": child_id,
                "sex": sex,
                "age_months": float(age_months),
                "height_cm": float(height_cm),
                "weight_kg": float(weight_kg),
            }
            out = api_post("/growth/score", payload)
            if out is not None:
                st.session_state["growth_last"] = out
                st.success("Growth score computed and saved to Twin.")

        if fetch:
            raw = api_get(f"/twin/events/latest/{child_id}")
            if raw is not None:
                st.session_state["twin_latest"] = raw
                st.session_state["twin_latest_loaded_at"] = datetime.now().strftime("%H:%M:%S")
                st.success("Latest snapshot loaded.")

        if timeline:
            raw = api_get(f"/twin/events/{child_id}")
            st.session_state["twin_events"] = normalize_events(raw)
            st.success("Timeline loaded.")

        # Prefer showing last computed, else show snapshot growth
        growth_out = st.session_state.get("growth_last")
        if not growth_out:
            snap = st.session_state.get("twin_latest") or {}
            snap_payload = (
                (snap.get("snapshot", {}) or {}).get("growth", {}) or {}
            ).get("payload")
            if snap_payload:
                growth_out = snap_payload

        if growth_out:
            waz = float(growth_out.get("waz", 0.0) or 0.0)
            haz = float(growth_out.get("haz", 0.0) or 0.0)
            whz = float(growth_out.get("whz", 0.0) or 0.0)
            overall = float(growth_out.get("overall_risk", 0.0) or 0.0)
            conf = float(growth_out.get("confidence", 0.0) or 0.0)

            m1, m2, m3 = st.columns(3)
            m1.metric("WAZ", f"{waz:.2f}")
            m2.metric("HAZ", f"{haz:.2f}")
            m3.metric("WHZ", f"{whz:.2f}")

            st.metric("Overall Growth Risk (%)", f"{overall * 100:.2f}")
            st.progress(max(0.0, min(1.0, overall)))

            st.caption(f"Confidence: {conf:.2f}")

            with st.expander("Show details (for clinicians / debug)"):
                st.json(growth_out)
        else:
            st.info("Run Growth Score to see output here (no scrolling).")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- 3) Latest Digital Twin Snapshot ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 3) Latest Digital Twin Snapshot")

        latest = st.session_state.get("twin_latest")

        if not latest:
            st.info("Click **Fetch Latest Twin** to load the latest state here.")
        else:
            st.caption(f"Last loaded at: {st.session_state.get('twin_latest_loaded_at', '—')}")
            snap = latest.get("snapshot", {}) if isinstance(latest, dict) else {}
            cols = st.columns(4)

            def _risk_pct(x):
                try:
                    return float(x) * 100.0
                except Exception:
                    return None

            # Growth
            g = snap.get("growth", {}) or {}
            gp = (g.get("payload", {}) or {})
            if gp:
                rp = _risk_pct(gp.get("overall_risk"))
                cols[0].metric("Growth Risk", f"{rp:.2f}%" if rp is not None else "—")
            else:
                cols[0].metric("Growth", "—")

            # Learning
            l = snap.get("learning", {}) or {}
            lp = (l.get("payload", {}) or {})
            if lp:
                cols[1].metric("Learning Risk", f"{lp.get('learning_risk_pct', 0):.2f}%")
            else:
                cols[1].metric("Learning", "—")

            # Emotion
            e = snap.get("emotion", {}) or {}
            ep = (e.get("payload", {}) or {})
            if ep:
                pct = ep.get("distress_risk_pct")
                if pct is None:
                    pct = _risk_pct(ep.get("distress_risk")) or 0.0
                cols[2].metric("Distress Risk", f"{float(pct):.2f}%")
            else:
                cols[2].metric("Emotion", "—")

            # Fusion
            f = snap.get("fusion", {}) or {}
            fp = (f.get("payload", {}) or {})
            if fp:
                cols[3].metric("Global Risk", f"{fp.get('global_development_risk_pct', 0):.2f}%")
            else:
                cols[3].metric("Fusion", "—")

            with st.expander("Show full snapshot (debug / clinician view)"):
                st.json(latest)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Below: Timeline (collapsed by default) ----------
    st.divider()

    with st.expander("Twin Timeline (All Modalities)"):
        events = normalize_events(st.session_state.get("twin_events"))
        if not events:
            st.info("Click Show Timeline to load events.")
            return

        df = to_df_events(events)
        if df.empty:
            st.info("No events found yet.")
            return

        st.dataframe(
            df[["timestamp", "modality"]].tail(50), use_container_width=True
        )

        fusion_rows = []
        for _, row in df.iterrows():
            if row["modality"] == "fusion":
                p = row["payload"] or {}
                fusion_rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "risk_pct": float(
                            p.get("global_development_risk_pct", 0.0) or 0.0
                        ),
                    }
                )
        if fusion_rows:
            dff = (
                pd.DataFrame(fusion_rows)
                .dropna(subset=["timestamp"])
                .sort_values("timestamp")
            )
            st.line_chart(dff.set_index("timestamp")["risk_pct"])


def home_dashboard() -> None:
    """Judge-friendly home dashboard for the multimodal twin."""
    st.title("Multimodal Child Development Digital Twin")
    st.caption(
        "Growth • Learning • Emotion • Fusion • Recommendations • Simulation (What-If)"
    )

    # Sidebar: child selection + demo controls
    with st.sidebar:
        st.header("Control Panel")
        child_id = st.text_input("Child ID", value="demo_child_001", key="home_child_id")
        st.divider()
        st.subheader("Demo Mode")
        demo = st.toggle("Enable guided demo", value=True)
        st.caption("Guides judges through the full story.")
        st.divider()
        st.subheader("Quick Actions")
        run_fusion = st.button("Run Fusion Score", key="home_run_fusion")
        run_recs = st.button("Generate Recommendations", key="home_run_recs")
        fetch = st.button("Fetch Latest Snapshot", key="home_fetch_latest")

    if "latest_snapshot" not in st.session_state:
        st.session_state["latest_snapshot"] = None

    if fetch:
        try:
            st.session_state["latest_snapshot"] = api_get(
                f"/twin/events/latest/{child_id}"
            )
            st.success("Snapshot loaded.")
        except Exception as e:
            st.error(f"Snapshot load failed: {e}")

    if run_fusion:
        try:
            out = api_post("/fusion/score", {"child_id": child_id})
            st.session_state["fusion_last"] = out
            st.success("Fusion computed and saved to Twin.")
        except Exception as e:
            st.error(f"Fusion scoring failed: {e}")

    if run_recs:
        try:
            out = api_post("/recommendations", {"child_id": child_id, "max_items": 5})
            st.session_state["recs_last"] = out
            st.success("Recommendations generated and saved to Twin.")
        except Exception as e:
            st.error(f"Recommendations failed: {e}")

    # Pull snapshot if missing
    snap_obj = st.session_state.get("latest_snapshot")
    if not snap_obj:
        try:
            snap_obj = api_get(f"/twin/events/latest/{child_id}")
            st.session_state["latest_snapshot"] = snap_obj
        except Exception:
            snap_obj = {"child_id": child_id, "snapshot": {}}

    snapshot = (snap_obj or {}).get("snapshot", {}) or {}

    def _payload(mod: str) -> dict:
        m = snapshot.get(mod)
        if isinstance(m, dict):
            return m.get("payload", {}) or {}
        return {}

    g = _payload("growth")
    l = _payload("learning")
    e = _payload("emotion")
    f = _payload("fusion")
    r = _payload("recommendations")

    growth_risk = float(g.get("overall_risk", 0.0) or 0.0)
    learn_risk = float(l.get("learning_risk", 0.0) or 0.0)
    emo_risk = float(e.get("distress_risk", 0.0) or 0.0)
    fusion_risk = float(f.get("global_development_risk", 0.0) or 0.0)

    left, mid, right = st.columns([1.2, 1.2, 1.6])

    with left:
        st.subheader("Child Snapshot")
        st.metric("Child ID", child_id)
        st.metric("Dominant Modality", f.get("dominant_modality", "unknown"))

    with mid:
        st.subheader("Global Risk (Fusion)")
        st.metric("Global Risk (%)", f"{fusion_risk * 100:.2f}")
        gauge("Fusion Risk", fusion_risk, "overall development risk")

    with right:
        st.subheader("Modality Risks")
        c1, c2, c3 = st.columns(3)
        c1.metric("Growth", f"{growth_risk * 100:.1f}%")
        c2.metric("Learning", f"{learn_risk * 100:.1f}%")
        c3.metric("Emotion", f"{emo_risk * 100:.1f}%")

        contrib = f.get("contributions", {}) or {}
        if contrib:
            dfc = pd.DataFrame(
                [
                    {"modality": k, "contribution": float(v)}
                    for k, v in contrib.items()
                ]
            ).sort_values("contribution", ascending=False)
            st.bar_chart(dfc.set_index("modality"))

    if demo:
        card(
            "What this system does (judge summary)",
            """
            <div class='small'>
            <b>1)</b> Ingests multimodal signals (Growth/Learning/Emotion).<br/>
            <b>2)</b> Builds a longitudinal Digital Twin timeline.<br/>
            <b>3)</b> Computes modality risks and fusion global risk with explainability.<br/>
            <b>4)</b> Produces ranked interventions by stakeholder.<br/>
            <b>5)</b> Runs What-If simulation to compare outcomes before and after.
            </div>
            """,
        )

    st.divider()

    tabs = st.tabs(
        ["Fusion", "Recommendations", "What-If", "Timeline", "Raw Snapshot"]
    )

    with tabs[0]:
        st.subheader("Fusion Details")
        if f:
            st.json(f)
        else:
            st.info("No fusion payload yet. Use the sidebar: Run Fusion Score.")

    with tabs[1]:
        st.subheader("Recommendations")
        if r:
            st.metric("Priority Level", r.get("priority_level", "NA"))
            st.metric(
                "Severity Score",
                f"{float(r.get('severity_score', 0.0) or 0.0):.3f}",
            )
            recs = r.get("recommendations", []) or []
            if recs:
                for rec in recs:
                    st.markdown(f"### {rec.get('title','')}")
                    st.write(f"Stakeholder: **{rec.get('stakeholder','')}**")
                    st.write(
                        "Priority: **{}** | Timeframe: **{} days**".format(
                            rec.get("priority", ""),
                            rec.get("expected_timeframe_days", ""),
                        )
                    )
                    for line in rec.get("rationale", []) or []:
                        st.write(f"- {line}")
                    st.divider()
            else:
                st.info(
                    "No recommendations in snapshot yet. Use the sidebar: Generate Recommendations."
                )
        else:
            st.info(
                "No recommendations in snapshot yet. Use the sidebar: Generate Recommendations."
            )

    with tabs[2]:
        st.subheader("What-If Simulator")
        colA, colB, colC = st.columns(3)
        emo_v = colA.slider("Emotion risk", 0.0, 1.0, emo_risk, 0.01)
        learn_v = colB.slider("Learning risk", 0.0, 1.0, learn_risk, 0.01)
        grow_v = colC.slider("Growth risk", 0.0, 1.0, growth_risk, 0.01)

        save_sim = st.checkbox("Save simulation to Twin", True)

        if st.button("Run What-If Simulation"):
            try:
                out = api_post(
                    "/simulate",
                    {
                        "child_id": child_id,
                        "overrides": {
                            "emotion_risk": float(emo_v),
                            "learning_risk": float(learn_v),
                            "growth_risk": float(grow_v),
                        },
                        "max_items": 5,
                        "save_to_twin": bool(save_sim),
                    },
                )
                st.session_state["sim_last"] = out
            except Exception as e:
                st.error(f"Simulation failed: {e}")

        sim = st.session_state.get("sim_last")
        if sim:
            base = sim.get("baseline", {}) or {}
            simm = sim.get("simulated", {}) or {}
            delta = sim.get("delta", {}) or {}

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Baseline Risk (%)",
                f"{float(base.get('global_risk_pct', 0.0) or 0.0):.2f}",
            )
            c2.metric(
                "Simulated Risk (%)",
                f"{float(simm.get('global_risk_pct', 0.0) or 0.0):.2f}",
            )
            c3.metric(
                "Delta Risk (pp)",
                f"{float(delta.get('risk_change_pct_points', 0.0) or 0.0):+.2f}",
            )

            st.caption("Simulated recommendations")
            for rec in simm.get("recommendations", []) or []:
                st.markdown(
                    f"**{rec.get('title','')}** - {rec.get('stakeholder','')} ({rec.get('priority','')})"
                )

            with st.expander("Simulation JSON"):
                st.json(sim)
        else:
            st.info(
                "Run a simulation to compare baseline and simulated risk and interventions."
            )

    with tabs[3]:
        st.subheader("Twin Timeline")
        try:
            events = normalize_events(api_get(f"/twin/events/{child_id}"))
            df = to_df_events(events)
            if df.empty:
                st.info("No timeline events found.")
            else:
                fusion_rows = []
                for _, row in df.iterrows():
                    if row["modality"] == "fusion":
                        p = row["payload"] or {}
                        fusion_rows.append(
                            {
                                "timestamp": row["timestamp"],
                                "risk_pct": float(
                                    p.get("global_development_risk_pct", 0.0) or 0.0
                                ),
                            }
                        )
                if fusion_rows:
                    dff = (
                        pd.DataFrame(fusion_rows)
                        .dropna(subset=["timestamp"])
                        .sort_values("timestamp")
                    )
                    st.line_chart(dff.set_index("timestamp")["risk_pct"])

                st.dataframe(
                    df[["timestamp", "modality"]].tail(50), use_container_width=True
                )
                with st.expander("Raw events JSON"):
                    st.json(events)
        except Exception as ex:
            st.error(f"Timeline load failed: {ex}")

    with tabs[4]:
        st.subheader("Latest Snapshot JSON")
        st.json(snapshot)


# Tab content
with tabs[0]:
    growth_tab_ui()

with tabs[1]:
    learning_tab_ui()

with tabs[2]:
    emotion_tab_ui()

with tabs[3]:
    fusion_view()

with tabs[4]:
    interventions_tab_ui()

with tabs[5]:
    twin_timeline_tab()
