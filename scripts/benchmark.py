import sys, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

video = sys.argv[1] if len(sys.argv) > 1 else "data/sample.mp4"
weights = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640
max_frames, warmup = 200, 5

model = YOLO(weights)
cap = cv2.VideoCapture(video)
t_read, t_track, t_draw = [], [], []

n = 0
while n < max_frames + warmup:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    t1 = time.perf_counter()
    if not ok:
        break
    res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                      imgsz=imgsz, verbose=False)
    t2 = time.perf_counter()
    annotated = res[0].plot()
    t3 = time.perf_counter()
    if n >= warmup:
        t_read.append(t1 - t0)
        t_track.append(t2 - t1)
        t_draw.append(t3 - t2)
    n += 1

def ms(x): return f"{np.mean(x)*1000:7.1f} ms (p95 {np.percentile(x,95)*1000:.1f})"

total = np.array(t_read) + np.array(t_track) + np.array(t_draw)
print(f"\nvideo={video}  weights={weights}  imgsz={imgsz}  frames={len(total)}")
print("read          ", ms(t_read))
print("detect+track  ", ms(t_track))
print("draw          ", ms(t_draw))
print(f"total/frame    {np.mean(total)*1000:7.1f} ms  ->  {1/np.mean(total):.1f} FPS")