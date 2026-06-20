import json
from pathlib import Path

import cv2

from src.ingestion.reader import get_reader
from src.models.cityeye import CityEye
from src.utils.draw import annotate_frame


def run_inference(cfg):
    source = cfg.inference.source
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = get_reader(source, fps_target=cfg.inference.fps_target)
    model = CityEye(cfg)

    writer = None
    events = []

    print(f"Running inference on: {source}")
    for frame_id, ts, frame in reader.read():
        result = model.process_frame(frame, frame_id, ts)
        annotated = annotate_frame(frame, result)

        if writer is None:
            h, w = annotated.shape[:2]
            out_path = output_dir / "output.mp4"
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
            )
            print(f"Writing annotated video to: {out_path}")

        writer.write(annotated)
        events.append(result.to_dict())

        if frame_id % 30 == 0:
            print(f"  frame {frame_id} | {result.congestion_level} | "
                  f"{result.vehicle_count} vehicles | "
                  f"{len(result.violations)} violations | "
                  f"{len(result.anomalies)} anomalies")

    reader.release()
    if writer:
        writer.release()

    json_path = output_dir / "events.json"
    with open(json_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved event log to: {json_path}")

    return events