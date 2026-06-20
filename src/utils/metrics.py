from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Detection:
    bbox: List[float]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    vehicle_count: int = 0
    congestion_level: str = "unknown"
    violations: List[dict] = field(default_factory=list)
    anomalies: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 2),
            "vehicle_count": self.vehicle_count,
            "congestion_level": self.congestion_level,
            "violations": self.violations,
            "anomalies": self.anomalies,
        }


# COCO class IDs YOLOv8 already knows, mapped to our traffic vocabulary
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
COUNTABLE_IDS = set(VEHICLE_CLASSES.keys())


def classify_congestion(vehicle_count: int, thresholds) -> str:
    if vehicle_count <= thresholds.free_flow:
        return "free_flow"
    if vehicle_count <= thresholds.moderate:
        return "moderate"
    if vehicle_count <= thresholds.dense:
        return "dense"
    return "gridlock"


def detect_helmet_violations(detections: List[Detection], aspect_threshold: float) -> List[dict]:
    """
    MVP proxy: a motorcycle bounding box that's unusually 'flat' (wide relative
    to tall) suggests the rider's head/helmet isn't prominently visible.
    Production version: a dedicated classifier head trained on labeled helmet data.
    """
    violations = []
    for d in detections:
        if d.class_name != "motorcycle":
            continue
        x1, y1, x2, y2 = d.bbox
        h, w = (y2 - y1), (x2 - x1)
        ratio = h / (w + 1e-6)
        if ratio < aspect_threshold and d.confidence > 0.5:
            violations.append({"type": "no_helmet", "track_id": d.track_id, "bbox": d.bbox})
    return violations


class AnomalyTracker:
    """Stateful tracker across frames — flags vehicles stopped too long."""

    def __init__(self, stopped_frames_threshold: int, min_movement_px: float):
        self.threshold = stopped_frames_threshold
        self.min_movement = min_movement_px
        self._stopped_counts: Dict[int, int] = {}
        self._prev_positions: Dict[int, Tuple[float, float]] = {}

    def update(self, detections: List[Detection]) -> List[dict]:
        anomalies = []
        for d in detections:
            if d.track_id is None:
                continue
            cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2

            if d.track_id in self._prev_positions:
                px, py = self._prev_positions[d.track_id]
                moved = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if moved > self.min_movement:
                    self._stopped_counts[d.track_id] = 0
                else:
                    self._stopped_counts[d.track_id] = self._stopped_counts.get(d.track_id, 0) + 1

            self._prev_positions[d.track_id] = (cx, cy)

            if (self._stopped_counts.get(d.track_id, 0) >= self.threshold
                    and d.class_name in ("car", "bus", "truck")):
                anomalies.append({
                    "type": "stopped_vehicle",
                    "track_id": d.track_id,
                    "bbox": d.bbox,
                })
        return anomalies