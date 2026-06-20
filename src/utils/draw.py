import cv2

COLOR_MAP = {
    "car": (0, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 0, 0),
    "truck": (255, 0, 255),
    "bicycle": (0, 255, 255),
}

CONGESTION_COLORS = {
    "free_flow": (0, 200, 0),
    "moderate": (0, 200, 200),
    "dense": (0, 100, 255),
    "gridlock": (0, 0, 255),
}


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = COLOR_MAP.get(d.class_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        if d.track_id is not None:
            label += f" #{d.track_id}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def draw_violations(frame, violations):
    for v in violations:
        x1, y1, x2, y2 = map(int, v["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "NO HELMET", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def draw_anomalies(frame, anomalies):
    for a in anomalies:
        x1, y1, x2, y2 = map(int, a["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
        cv2.putText(frame, "STOPPED", (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    return frame


def draw_stats_overlay(frame, result):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 110), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    color = CONGESTION_COLORS.get(result.congestion_level, (255, 255, 255))
    cv2.putText(frame, f"Congestion: {result.congestion_level.upper()}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Vehicles: {result.vehicle_count}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Violations: {len(result.violations)}",
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    cv2.putText(frame, f"Anomalies: {len(result.anomalies)}",
                (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return frame


def annotate_frame(frame, result):
    frame = draw_detections(frame, result.detections)
    frame = draw_violations(frame, result.violations)
    frame = draw_anomalies(frame, result.anomalies)
    frame = draw_stats_overlay(frame, result)
    return frame