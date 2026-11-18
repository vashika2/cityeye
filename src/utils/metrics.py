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
    """
    Flags vehicles that stay in one place for too long.

    Movement is measured from the spot where the vehicle first stopped (its
    anchor), not from the previous frame, so slowly crawling traffic is not
    mistaken for a stalled vehicle. The allowed wobble scales with box size,
    so near and far vehicles are treated alike.

    update() returns every vehicle that is currently stalled, so overlays can
    keep drawing it. Each entry has "new": True only on the first frame of that
    stall, so counting events means counting the new ones.
    """

    WATCHED = ("car", "bus", "truck")

    def __init__(self, stopped_frames_threshold: int, min_movement_px: float,
                 size_ratio: float = 0.05, forget_after_frames: int = 60):
        self.threshold = stopped_frames_threshold
        self.min_movement = min_movement_px
        self.size_ratio = size_ratio
        self.forget_after = forget_after_frames
        self._frame = 0
        self._tracks: Dict[int, dict] = {}

    def update(self, detections: List[Detection]) -> List[dict]:
        self._frame += 1
        active = []
        for d in detections:
            if d.track_id is None or d.class_name not in self.WATCHED:
                continue
            x1, y1, x2, y2 = d.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            radius = max(self.min_movement, self.size_ratio * max(x2 - x1, y2 - y1))

            tr = self._tracks.get(d.track_id)
            if tr is None:
                self._tracks[d.track_id] = {
                    "anchor": (cx, cy), "still": 0, "alerted": False, "seen": self._frame,
                }
                continue

            tr["seen"] = self._frame
            ax, ay = tr["anchor"]
            if ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 > radius:
                tr["anchor"], tr["still"], tr["alerted"] = (cx, cy), 0, False
            else:
                tr["still"] += 1

            if tr["still"] >= self.threshold:
                active.append({
                    "type": "stopped_vehicle",
                    "track_id": d.track_id,
                    "bbox": d.bbox,
                    "new": not tr["alerted"],
                })
                tr["alerted"] = True

        self._prune()
        return active

    def _prune(self):
        cutoff = self._frame - self.forget_after
        for tid in [t for t, tr in self._tracks.items() if tr["seen"] < cutoff]:
            del self._tracks[tid]