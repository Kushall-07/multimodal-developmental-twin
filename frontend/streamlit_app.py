import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Child Digital Twin (Prototype)", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8001")

st.title("AI Multimodal Child Development Digital Twin — Prototype")
st.caption("Currently enabled: Growth & Nutrition (WHO z-scores) + Digital Twin persistence")

# -----------------------
# Inputs
# -----------------------
st.subheader("1) Growth & Nutrition Input")

col1, col2, col3, col4 = st.columns(4)
with col1:
    child_id = st.text_input("Child ID", value="demo_001")
with col2:
    sex = st.selectbox("Sex", ["M", "F"])
with col3:
    age_months = st.number_input("Age (months)", min_value=0.0, value=12.0, step=1.0)
with col4:
    st.write("")  # spacing

col5, col6 = st.columns(2)
with col5:
    height_cm = st.number_input("Height (cm)", min_value=1.0, value=75.0, step=0.5)
with col6:
    weight_kg = st.number_input("Weight (kg)", min_value=0.5, value=9.0, step=0.1)

# -----------------------
# Actions
# -----------------------
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

with btn_col1:
    run_score = st.button("Run Growth Score", type="primary")
with btn_col2:
    fetch_latest = st.button("Fetch Latest Twin State")
with btn_col3:
    generate_demo = st.button("Generate Demo Trajectory (3–5 points)")

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


def risk_badge(r):
    # simple display categories for judges (not used for model logic)
    if r < 0.2:
        return "Low"
    if r < 0.5:
        return "Moderate"
    return "High"


def get_history():
    r = requests.get(f"{API_BASE}/twin/history/{child_id}?limit=20", timeout=30)
    r.raise_for_status()
    return r.json()


# -----------------------
# Results
# -----------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("2) Model Output (Growth Engine)")

    if run_score:
        try:
            out = post_growth_score()
            st.success("Growth scoring successful. (Result auto-saved to Digital Twin DB.)")
            st.json(out)

            # Clean visualization cards
            c1, c2, c3 = st.columns(3)
            c1.metric("WAZ", f"{out['waz']:.2f}")
            c2.metric("HAZ", f"{out['haz']:.2f}")
            c3.metric("WHZ", f"{out['whz']:.2f}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Stunting Risk", f"{out['stunting_risk']:.3f}", help=risk_badge(out["stunting_risk"]))
            r2.metric("Wasting Risk", f"{out['wasting_risk']:.3f}", help=risk_badge(out["wasting_risk"]))
            r3.metric("Underweight Risk", f"{out['underweight_risk']:.3f}", help=risk_badge(out["underweight_risk"]))
            r4.metric("Overall Growth Risk", f"{out['overall_risk']:.3f}")

            # Narrative risk summary for judges
            overall_level = risk_badge(out["overall_risk"])
            st.markdown(
                f"**Risk Summary:** Overall growth risk: **{overall_level}** "
                f"({out['overall_risk']:.3f}). Confidence: {out['confidence']:.2f}."
            )

        except Exception as e:
            st.error(f"Growth scoring failed: {e}")

with right:
    st.subheader("3) Digital Twin Snapshot (Latest State)")

    if fetch_latest:
        try:
            twin = get_latest_twin()
            if twin.get("status") == "not_found":
                st.warning("No twin state found yet for this child. Run Growth Score first.")
            else:
                st.success("Latest Twin State loaded.")
                st.write("**Child ID:**", twin["child_id"])
                st.write("**Timestamp:**", twin["created_at"])

                growth = twin.get("growth", {})
                g1, g2 = st.columns(2)
                g1.metric("Overall Growth Risk", f"{growth.get('overall_risk', 0):.3f}")
                g2.metric("Confidence", f"{growth.get('confidence', 0):.2f}")

                st.write("**Stored Snapshot:**")
                st.json(twin.get("snapshot", {}))

        except Exception as e:
            st.error(f"Fetch latest twin failed: {e}")

    st.subheader("4) Twin Timeline (Overall Growth Risk)")

    if st.button("Show Twin Timeline"):
        try:
            hist = get_history()
            if hist:
                df = pd.DataFrame(hist)
                df["created_at"] = pd.to_datetime(df["created_at"])
                df = df.sort_values("created_at")
                st.line_chart(df.set_index("created_at")["growth_overall_risk"])
            else:
                st.warning("No history yet. Run Growth Score multiple times.")
        except Exception as e:
            st.error(f"Loading history failed: {e}")

    if generate_demo:
        try:
            st.info("Generating 3 demo growth points for this child_id…")
            # Simple synthetic variations around current inputs (no training, just API calls)
            for delta in (-1.0, 0.0, 1.0):
                demo_payload = {
                    "child_id": child_id,
                    "sex": sex,
                    "age_months": float(age_months + delta),
                    "height_cm": float(height_cm + 0.5 * delta),
                    "weight_kg": float(weight_kg + 0.2 * delta),
                }
                r = requests.post(f"{API_BASE}/growth/score", json=demo_payload, timeout=30)
                r.raise_for_status()
            st.success("Demo trajectory generated. Use 'Show Twin Timeline' to visualize.")
        except Exception as e:
            st.error(f"Demo generation failed: {e}")

st.divider()

