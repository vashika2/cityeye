import os
import tempfile
import threading
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from omegaconf import OmegaConf

from src.ingestion.reader import get_reader
from src.models.cityeye import CityEye

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

state = {}
# The model and its tracker are shared, so only one request may use them at a time.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = OmegaConf.load(CONFIG_PATH)
    state["cfg"] = cfg
    state["engine"] = CityEye(cfg)   # load the model once, not per request
    yield
    state.clear()


app = FastAPI(
    title="CityEye API",
    version="1.0.0",
    description="Traffic analysis from images and video: vehicle counts, congestion level, "
                "helmet heuristic and stalled-vehicle alerts.",
    lifespan=lifespan,
)


def _detection_to_dict(d):
    return {
        "class": d.class_name,
        "confidence": round(d.confidence, 3),
        "bbox": [round(v, 1) for v in d.bbox],
    }


def _summarise(events, truncated):
    counts = [e["vehicle_count"] for e in events]
    return {
        "frames_processed": len(events),
        "truncated": truncated,
        "max_vehicle_count": max(counts, default=0),
        "avg_vehicle_count": round(sum(counts) / len(counts), 2) if counts else 0,
        "congestion_levels": dict(Counter(e["congestion_level"] for e in events)),
        "frames_with_violations": sum(1 for e in events if e["violations"]),
        "frames_with_anomalies": sum(1 for e in events if e["anomalies"]),
    }


def _save_upload(file: UploadFile, suffix: str, limit: int) -> str:
    """Copy an upload to a temp file, enforcing a size limit. Returns the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    size = 0
    try:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"File too large (limit {limit // (1024 * 1024)} MB)")
            tmp.write(chunk)
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise
    tmp.close()
    if size == 0:
        os.unlink(tmp.name)
        raise HTTPException(400, "Uploaded file is empty")
    return tmp.name


@app.get("/")
def root():
    return {
        "name": "CityEye API",
        "version": "1.0.0",
        "description": "Traffic analysis from images and video",
        "endpoints": [
            "/analyze/frame  - POST - analyze a single image",
            "/analyze/video  - POST - analyze a video file",
            "/health         - GET  - health check",
            "/config         - GET  - current system config",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": state["cfg"].model.backbone}


@app.get("/config")
def get_config():
    return OmegaConf.to_container(state["cfg"], resolve=True)


@app.post("/analyze/frame")
def analyze_frame(file: UploadFile = File(...)):
    """One image in, detections, vehicle count and congestion level out. Stateless."""
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (limit 20 MB)")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode the upload as an image")

    engine = state["engine"]
    with lock:
        engine.reset()
        result = engine.process_frame(frame, 0, 0.0)
        engine.reset()

    return {
        "vehicle_count": result.vehicle_count,
        "congestion_level": result.congestion_level,
        "detections": [_detection_to_dict(d) for d in result.detections],
        "violations": result.violations,
        "note": "Single frame: stalled-vehicle detection needs video and is not run here.",
    }


@app.post("/analyze/video")
def analyze_video(
        file: UploadFile = File(...),
        max_frames: int = Query(300, ge=1, le=5000, description="Stop after this many processed frames"),
):
    """Video in, per-frame events and a summary out."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type; use one of {sorted(VIDEO_SUFFIXES)}")

    path = _save_upload(file, suffix, MAX_VIDEO_BYTES)
    engine = state["engine"]
    cfg = state["cfg"]
    events = []
    truncated = False
    try:
        try:
            reader = get_reader(path, fps_target=cfg.inference.fps_target)
        except (FileNotFoundError, ValueError):
            raise HTTPException(400, "Could not open the upload as a video")

        with lock:
            engine.reset()
            try:
                for n, (frame_id, ts, frame) in enumerate(reader.read()):
                    if n >= max_frames:
                        truncated = True
                        break
                    events.append(engine.process_frame(frame, frame_id, ts).to_dict())
            finally:
                reader.release()
                engine.reset()
    finally:
        os.unlink(path)

    return {"summary": _summarise(events, truncated), "events": events}