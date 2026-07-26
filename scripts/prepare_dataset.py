import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json, random, cv2
from pathlib import Path


def extract_frames(video_path, output_dir, fps_target=5, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, int(fps / fps_target))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved, frame_id, count = [], 0, 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if frame_id % skip == 0:
            fname = f"frame_{count:05d}.jpg"
            cv2.imwrite(str(Path(output_dir) / fname), frame)
            saved.append(fname)
            count += 1
        frame_id += 1
    cap.release()
    print(f"Saved {count} frames to {output_dir}")
    return saved


def prepare_from_videos(video_dir, output_dir, train_split=0.8, fps_target=5, max_frames_per_video=300):
    from ultralytics import YOLO
    video_dir, output_dir = Path(video_dir), Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    if not videos: raise FileNotFoundError(f"No videos found in {video_dir}")
    print(f"Found {len(videos)} video(s)")
    yolo = YOLO("yolov8n.pt")
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
    all_images, all_annotations, image_id, ann_id = [], [], 0, 0
    for video_path in videos:
        frame_names = extract_frames(str(video_path), str(images_dir), fps_target, max_frames_per_video)
        for fname in frame_names:
            frame = cv2.imread(str(images_dir / fname))
            if frame is None: continue
            h, w = frame.shape[:2]
            results = yolo(frame, verbose=False, conf=0.4)
            boxes, labels, vehicle_count = [], [], 0
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES: continue
                    boxes.append(box.xyxy[0].cpu().numpy().tolist())
                    labels.append(cls_id)
                    vehicle_count += 1
            congestion = "free_flow" if vehicle_count <= 5 else "moderate" if vehicle_count <= 15 else "dense" if vehicle_count <= 30 else "gridlock"
            all_images.append({"id": image_id, "file_name": fname, "width": w, "height": h, "congestion_label": congestion, "violations": [], "anomalies": []})
            for bbox, label in zip(boxes, labels):
                all_annotations.append({"id": ann_id, "image_id": image_id, "category_id": label, "bbox": bbox, "area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1])})
                ann_id += 1
            image_id += 1
    print(f"Total: {len(all_images)} images, {len(all_annotations)} annotations")
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_split)
    train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]
    train_ids = {img["id"] for img in train_imgs}
    for split, imgs, anns in [("train", train_imgs, [a for a in all_annotations if a["image_id"] in train_ids]), ("val", val_imgs, [a for a in all_annotations if a["image_id"] not in train_ids])]:
        with open(output_dir / f"{split}.json", "w") as f:
            json.dump({"images": imgs, "annotations": anns}, f, indent=2)
        print(f"{split}: {len(imgs)} images")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()
    prepare_from_videos(args.video_dir, args.output_dir, args.train_split, args.fps, args.max_frames)
