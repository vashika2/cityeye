from pathlib import Path
import cv2


class VideoReader:
    """Reads a video file, downsampled to fps_target frames per second."""

    def __init__(self, path: str, fps_target: int = 10):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.skip = max(1, int(fps / fps_target))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        """Yields (frame_id, timestamp_seconds, frame) tuples."""
        frame_id = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_id % self.skip == 0:
                ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield frame_id, ts, frame
            frame_id += 1

    def release(self):
        self.cap.release()


def get_reader(source: str, fps_target: int = 10):
    """Factory: pass a video path, '0' for webcam, or rtsp:// URL."""
    if source.startswith("rtsp://") or source.startswith("rtmp://"):
        return VideoReader(source, fps_target)
    if source.isdigit():
        return VideoReader(int(source), fps_target)
    if Path(source).exists():
        return VideoReader(source, fps_target)
    raise ValueError(f"Source not found: {source}")