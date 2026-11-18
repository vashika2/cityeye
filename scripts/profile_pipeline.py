import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from omegaconf import OmegaConf
from src.ingestion.reader import get_reader
from src.models.cityeye import CityEye
from src.utils.draw import annotate_frame

cfg = OmegaConf.load("configs/default.yaml")
source = sys.argv[1] if len(sys.argv) > 1 else cfg.inference.source
warmup = 3

reader = get_reader(source, fps_target=cfg.inference.fps_target)
model = CityEye(cfg)

# wrap the YOLO call so we can separate detect+track from the rest of process_frame
track_times = []
orig_track = model.model.track
def timed_track(*a, **k):
    s = time.perf_counter()
    out = orig_track(*a, **k)
    track_times.append(time.perf_counter() - s)
    return out
model.model.track = timed_track

t_read, t_proc, t_draw, t_event = [], [], [], []
it = reader.read()
n = 0
while True:
    t0 = time.perf_counter()
    try:
        frame_id, ts, frame = next(it)
    except StopIteration:
        break
    t1 = time.perf_counter()
    result = model.process_frame(frame, frame_id, ts)
    t2 = time.perf_counter()
    annotated = annotate_frame(frame, result)
    t3 = time.perf_counter()
    result.to_dict()
    t4 = time.perf_counter()
    if n >= warmup:
        t_read.append(t1 - t0); t_proc.append(t2 - t1)
        t_draw.append(t3 - t2); t_event.append(t4 - t3)
    n += 1
reader.release()

track = np.array(track_times[warmup:])
proc = np.array(t_proc)
logic = proc - track[:len(proc)]
def ms(x): return f"{np.mean(x)*1000:7.1f} ms (p95 {np.percentile(x,95)*1000:.1f})"
total = np.array(t_read) + proc + np.array(t_draw) + np.array(t_event)

print(f"\nsource={source}  processed frames={len(total)}  (reader targets {cfg.inference.fps_target} FPS)")
print("read (incl. skipped frames)", ms(t_read))
print("detect + track             ", ms(track))
print("congestion/helmet/anomaly  ", ms(logic))
print("draw overlays              ", ms(t_draw))
print("to_dict                    ", ms(t_event))
print(f"total per processed frame   {np.mean(total)*1000:7.1f} ms  ->  {1/np.mean(total):.1f} processed FPS")
print(f"budget at {cfg.inference.fps_target} FPS: {1000/cfg.inference.fps_target:.0f} ms per frame")