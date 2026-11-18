from collections import deque
from statistics import median

import torch
from ultralytics import YOLO

from src.utils.metrics import (
    Detection, FrameResult, VEHICLE_CLASSES, COUNTABLE_IDS,
    classify_congestion, detect_helmet_violations, AnomalyTracker,
)


class CityEye:
    """Multi-task traffic intelligence wrapper around YOLOv8 + ByteTrack."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.model.device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(f"{cfg.model.backbone}.pt")
        self.conf = cfg.tasks.detection.confidence
        self.iou = cfg.tasks.detection.iou_threshold
        self.anomaly_tracker = AnomalyTracker(
            stopped_frames_threshold=cfg.tasks.anomaly.stopped_vehicle_frames,
            min_movement_px=cfg.tasks.anomaly.min_movement_pixels,
        )

        # Congestion smoothing: median over a window, plus a hold time before
        # the displayed level is allowed to change.
        cong = cfg.tasks.congestion
        self.count_window = deque(maxlen=cong.get("smoothing_window", 15))
        self.hold_frames = cong.get("hold_frames", 20)
        self.congestion_state = None
        self.pending_level = None
        self.pending_frames = 0

    def _stable_congestion(self, count: int) -> str:
        self.count_window.append(count)
        level = classify_congestion(
            median(self.count_window), self.cfg.tasks.congestion.thresholds
        )
        if self.congestion_state is None:
            self.congestion_state = level
        elif level == self.congestion_state:
            self.pending_level, self.pending_frames = None, 0
        else:
            if level == self.pending_level:
                self.pending_frames += 1
            else:
                self.pending_level, self.pending_frames = level, 1
            if self.pending_frames >= self.hold_frames:
                self.congestion_state = level
                self.pending_level, self.pending_frames = None, 0
        return self.congestion_state
    def reset(self):
        """Clear all per-stream state: smoothing, anomaly history and the tracker."""
        self.count_window.clear()
        self.congestion_state = None
        self.pending_level = None
        self.pending_frames = 0
        self.anomaly_tracker = AnomalyTracker(
            stopped_frames_threshold=self.cfg.tasks.anomaly.stopped_vehicle_frames,
            min_movement_px=self.cfg.tasks.anomaly.min_movement_pixels,
        )
        # Start ByteTrack with no known tracks
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers and all(hasattr(t, "reset") for t in trackers):
            for t in trackers:
                t.reset()
        else:
            self.model.predictor = None
    def process_frame(self, frame, frame_id: int, timestamp: float) -> FrameResult:
        result = FrameResult(frame_id=frame_id, timestamp=timestamp)

        outputs = self.model.track(
            frame, persist=True, conf=self.conf, iou=self.iou,
            device=self.device, tracker="bytetrack.yaml", verbose=False,
        )

        if outputs and outputs[0].boxes is not None:
            for box in outputs[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                result.detections.append(Detection(
                    bbox=box.xyxy[0].cpu().numpy().tolist(),
                    confidence=float(box.conf[0]),
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES[cls_id],
                    track_id=int(box.id[0]) if box.id is not None else None,
                ))

        result.vehicle_count = sum(1 for d in result.detections if d.class_id in COUNTABLE_IDS)
        result.congestion_level = self._stable_congestion(result.vehicle_count)
        result.violations = detect_helmet_violations(
            result.detections, self.cfg.tasks.violation.helmet_aspect_ratio_threshold
        )
        result.anomalies = self.anomaly_tracker.update(result.detections)
        return result