## CityEye — Urban Traffic Intelligence from CCTV Video

CityEye turns traffic video into structured data: vehicle counts with persistent IDs, a congestion level per frame, a basic helmet check for motorcycles, and stalled-vehicle alerts. It runs on standard video, with no special hardware.

Originally started for **Flipkart Gridlock Hackathon 2.0** and **InnovateZ — The Gen Z Challenge**.

## Live Demo
**[https://cityeye-mqgkwj33xvbzgtgrxalvbi.streamlit.app](https://cityeye-mqgkwj33xvbzgtgrxalvbi.streamlit.app)**

Upload a traffic video and the app runs detection, tracking, congestion classification and stalled-vehicle detection on it. The demo runs on CPU and processes uploaded files, not live streams.

## GitHub
**[https://github.com/vashika2/cityeye](https://github.com/vashika2/cityeye)**

---

## What it does

- **Vehicle detection and tracking:** cars, motorcycles, buses, trucks and bicycles via YOLOv8 + ByteTrack
- **Congestion classification:** Free Flow / Moderate / Dense / Gridlock, rule-based on the number of vehicles per frame, with thresholds set in config
- **Helmet check (heuristic):** a proxy based on motorcycle bounding-box shape, not a trained classifier. Expect false positives and misses
- **Stalled-vehicle detection:** flags a tracked car, bus or truck that barely moves over a run of consecutive processed frames
- **Dashboard:** Streamlit app that analyses an uploaded video and shows stats, charts and an event log
- **REST API:** FastAPI backend for analysing single frames and full videos
- **Config-driven:** thresholds and class lists live in `configs/default.yaml`, so adapting to a new camera or city means editing config rather than code

## Current status and limitations

Being upfront about what is and isn't done:

- The demo, dashboard and API all run the **YOLOv8n + ByteTrack pipeline**. COCO-pretrained YOLOv8 has no auto-rickshaw class, so autos appear as cars or trucks.
- I also built a **multi-task model** (ResNet-50 backbone + FPN, with detection, congestion, violation and anomaly heads). **Only the congestion head has been trained.** The detection head had no box targets, and the violation and anomaly heads were trained on empty labels, so those three are not usable yet. The multi-task model is not used by the dashboard or API.
- The congestion head was trained on **87 frames (69 train / 18 val) from a single ~17-second video**, with pseudo-labels from YOLOv8 boxes and count-based congestion levels. It reached **83.3% accuracy (15/18)** on the validation split. Because the validation frames come from the same video and the same labelling rule, this shows the training pipeline works, not that the model generalises.
- **Not yet measured:** FPS, mAP, MOTA.
- **Not yet implemented:** signal-jump, wrong-way and accident detection, ONNX export, Jetson deployment, training on IDD / UA-DETRAC / BDD100K.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection and tracking | YOLOv8n (Ultralytics) + ByteTrack |
| Multi-task model (experimental) | PyTorch, PyTorch Lightning, ResNet-50 + FPN |
| Video processing | Python, OpenCV |
| Config | OmegaConf (YAML) |
| Dashboard | Streamlit + Plotly |
| REST API | FastAPI + Uvicorn |
| Deployment | Streamlit Community Cloud (dashboard only) |

---

## Project Structure

```
cityeye/
├── configs/
│   └── default.yaml          # Tunable parameters
├── src/
│   ├── ingestion/            # Video / stream input handling
│   ├── models/               # Detector pipeline and multi-task model (backbone, FPN, heads)
│   ├── tracking/             # ByteTrack integration
│   ├── preprocessing/        # Frame transforms
│   ├── losses/               # Multi-task loss
│   ├── datasets/             # Dataset and dataloader code
│   ├── training/             # Lightning trainer
│   └── utils/                # Congestion / anomaly logic, drawing overlays
├── scripts/
│   ├── infer_video.py        # CLI inference
│   ├── prepare_dataset.py    # Frame extraction and pseudo-labelling
│   ├── train.py              # Train the multi-task model
│   └── evaluate.py           # Congestion accuracy evaluation
├── api/
│   └── main.py               # FastAPI server
├── dashboard/
│   └── app.py                # Streamlit dashboard
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place a traffic video at `data/sample.mp4`, then run the CLI:

```bash
python scripts/infer_video.py
```

The annotated video is saved to `runs/inference/output.mp4` and the event log to `runs/inference/events.json`.

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
| GET | `/` | API info |
| GET | `/health` | Health check and model name |
| GET | `/config` | Current system config |
| POST | `/analyze/frame` | Upload one image, get detections and congestion level as JSON |
| POST | `/analyze/video` | Upload a video, get full JSON analysis |

---

## Training the multi-task model

```bash
python scripts/prepare_dataset.py --video_dir data --output_dir data/processed
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint <path-to-checkpoint>
```

---

## Why YOLOv8 for the demo

The multi-task ResNet-50 + FPN model was built to explore a shared backbone serving several tasks at once. Since only its congestion head is trained, the demo uses pretrained YOLOv8 so the full pipeline works end to end. The design is modular, so a trained custom model could replace the detector later.

---

## Roadmap

- Add more videos and hold one out entirely for validation
- Fine-tune on IDD, UA-DETRAC or similar for Indian vehicle classes (autos, e-rickshaws)
- Label data for, and train, the detection, violation and anomaly heads
- Replace the helmet heuristic with a trained classifier
- Signal-jump and wrong-way detection
- Benchmark FPS, mAP and MOTA
- ONNX export and edge deployment (e.g. NVIDIA Jetson)
- Multi-camera dashboard
