## CityEye — AI-Powered Urban Traffic Intelligence

CityEye turns any city's existing CCTV infrastructure into an intelligent traffic monitoring system. It classifies congestion, tracks vehicles, flags violations, and detects anomalies — all from standard video feeds, with zero new hardware required.

Built for **Flipkart Gridlock Hackathon 2.0** and **InnovateZ — The Gen Z Challenge**.

## 🔴 Live Demo
**[https://cityeye-mqgkwj33xvbzgtgrxalvbi.streamlit.app](https://cityeye-mqgkwj33xvbzgtgrxalvbi.streamlit.app)**

Upload any traffic video and see real-time detection, congestion classification, violation flags, and anomaly alerts.

## 📁 GitHub
**[https://github.com/vashika2/cityeye](https://github.com/vashika2/cityeye)**

---

## What it does

- **Vehicle detection & tracking** — cars, motorcycles, buses, trucks, bicycles via YOLOv8 + ByteTrack
- **Congestion classification** — Free Flow / Moderate / Dense / Gridlock based on configurable thresholds
- **Violation detection** — helmet compliance detection (heuristic, upgradeable to trained classifier)
- **Anomaly detection** — stalled/stopped vehicles via track-position history across frames
- **Live dashboard** — Streamlit real-time monitoring with live stats, charts, and event log
- **REST API** — FastAPI backend for programmatic video analysis
- **City-agnostic by design** — every threshold and class lives in `configs/default.yaml`, not hardcoded

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection & Tracking | YOLOv8n (Ultralytics) + ByteTrack |
| Backend Logic | Python, OpenCV |
| Config System | OmegaConf (YAML-driven) |
| Dashboard | Streamlit + Plotly |
| REST API | FastAPI + Uvicorn |
| Deployment | Streamlit Community Cloud |

---

## Project Structure
cityeye/

├── configs/

│   └── default.yaml          # All tunable params — no hardcoding

├── src/

│   ├── ingestion/

│   │   └── reader.py         # Video/stream/RTSP input handling

│   ├── models/

│   │   └── cityeye.py        # Core detection + analysis pipeline

│   ├── tracking/             # ByteTrack integration

│   ├── preprocessing/        # Frame pipeline

│   └── utils/

│       ├── metrics.py        # Congestion/violation/anomaly logic

│       └── draw.py           # Visualization overlays

├── scripts/

│   └── infer_video.py        # CLI entrypoint

├── api/

│   └── main.py               # FastAPI REST server

├── dashboard/

│   └── app.py                # Streamlit live dashboard

└── requirements.txt

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place a traffic video at `data/sample.mp4`, then run the CLI:

```bash
python scripts/infer_video.py
```

Output video saved to `runs/inference/output.mp4` and event log to `runs/inference/events.json`.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Upload a traffic video in the sidebar and click **Run Analysis**.

---

## API

```bash
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for the auto-generated API docs.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/analyze/video` | Upload video, get full JSON analysis |

---

## Why YOLOv8 for the MVP

The production architecture uses a custom ResNet-50 + Feature Pyramid Network with four dedicated task heads trained on India-specific datasets (IDD). For this MVP, YOLOv8 pretrained weights demonstrate the full system end-to-end within hackathon constraints. The modular design means the custom backbone can be swapped in without changing any other part of the pipeline.

---

## Roadmap

- Fine-tune on IDD (India Driving Dataset) for Indian vehicle classes (autos, e-rickshaws)
- Replace heuristic violation detection with a trained classifier head
- Edge deployment via ONNX export on NVIDIA Jetson
- Multi-camera city-wide dashboard
- Signal-jump and wrong-way detection
