import json
from pathlib import Path

import cv2

from src.ingestion.reader import get_reader
from src.models.cityeye import CityEye
from src.utils.draw import annotate_frame


def run_inference(cfg, on_event=None, write_video=True, max_frames=None):
    """
    on_event: optional callable, called with each frame's event dict as it is produced.
    write_video: set False to skip drawing/encoding (events only).
    max_frames: stop after N processed frames (useful for testing live sources).
    """
    source = cfg.inference.source
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = get_reader(source, fps_target=cfg.inference.fps_target)
    model = CityEye(cfg)

    writer = None
    events = []
    jsonl = open(output_dir / "events.jsonl", "w")

    print(f"Running inference on: {source}")
    try:
        for n, (frame_id, ts, frame) in enumerate(reader.read()):
            if max_frames is not None and n >= max_frames:
                break

            result = model.process_frame(frame, frame_id, ts)
            event = result.to_dict()
            events.append(event)

            # emit immediately, one JSON object per line
            jsonl.write(json.dumps(event) + "\n")
            jsonl.flush()
            if on_event:
                on_event(event)

            if write_video:
                annotated = annotate_frame(frame, result)
                if writer is None:
                    h, w = annotated.shape[:2]
                    out_path = output_dir / "output.mp4"
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
                    )
                    print(f"Writing annotated video to: {out_path}")
                writer.write(annotated)

            if frame_id % 30 == 0:
                print(f"  frame {frame_id} | {result.congestion_level} | "
                      f"{result.vehicle_count} vehicles | "
                      f"{len(result.violations)} violations | "
                      f"{len(result.anomalies)} anomalies")
    finally:
        reader.release()
        if writer:
            writer.release()
        jsonl.close()

    with open(output_dir / "events.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved event log to: {output_dir / 'events.json'}")

    return events