import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
from omegaconf import OmegaConf

from src.ingestion.reader import get_reader
from src.models.cityeye import CityEye
from src.utils.draw import annotate_frame

st.set_page_config(page_title="CityEye — Traffic Intelligence", layout="wide")

st.title("CityEye")
st.caption("AI-powered urban traffic intelligence — works on any city's existing CCTV feed")

cfg = OmegaConf.load("configs/default.yaml")

# --- Sidebar controls ---
st.sidebar.header("Configuration")
source = st.sidebar.text_input("Video source", value=cfg.inference.source)
run_button = st.sidebar.button("Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Congestion thresholds")
st.sidebar.write(f"Free flow: ≤ {cfg.tasks.congestion.thresholds.free_flow} vehicles")
st.sidebar.write(f"Moderate: ≤ {cfg.tasks.congestion.thresholds.moderate} vehicles")
st.sidebar.write(f"Dense: ≤ {cfg.tasks.congestion.thresholds.dense} vehicles")
st.sidebar.write("Gridlock: above that")

# --- Main layout ---
col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    st.subheader("Live Stats")
    congestion_metric = st.empty()
    vehicle_metric = st.empty()
    violation_metric = st.empty()
    anomaly_metric = st.empty()

st.markdown("---")
st.subheader("Congestion Over Time")
chart_placeholder = st.empty()

st.subheader("Event Log")
log_placeholder = st.empty()

if run_button:
    try:
        st.write("Step 1: Opening video source...")
        reader = get_reader(source, fps_target=cfg.inference.fps_target)
        st.write("Step 2: Video opened successfully. Loading model...")
        model = CityEye(cfg)
        st.write("Step 3: Model loaded. Starting frame processing...")

        history = []
        events = []
        frame_count = 0

        for frame_id, ts, frame in reader.read():
            frame_count += 1
            result = model.process_frame(frame, frame_id, ts)
            annotated = annotate_frame(frame, result)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            congestion_metric.metric("Congestion Level", result.congestion_level.upper())
            vehicle_metric.metric("Vehicles Detected", result.vehicle_count)
            violation_metric.metric("Violations", len(result.violations))
            anomaly_metric.metric("Anomalies", len(result.anomalies))

            history.append({
                "frame": frame_id,
                "timestamp": round(ts, 1),
                "vehicle_count": result.vehicle_count,
                "congestion_level": result.congestion_level,
            })
            events.extend(result.violations)
            events.extend(result.anomalies)

            if len(history) % 5 == 0:
                df = pd.DataFrame(history)
                fig = px.line(df, x="timestamp", y="vehicle_count",
                              title="Vehicle Count Over Time")
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                if events:
                    log_placeholder.dataframe(pd.DataFrame(events), use_container_width=True)

        reader.release()
        st.success(f"Analysis complete. Processed {frame_count} frames.")

    except Exception as e:
        st.error(f"Error occurred: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("Set a video source in the sidebar and click 'Run Analysis' to begin.")