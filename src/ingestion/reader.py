import os
import threading
import time
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


class LiveReader:
    """
    Reads a live source (RTSP/RTMP stream or webcam) on a background thread.

    Only the newest frame is kept. If processing is slower than the camera,
    old frames are dropped instead of piling up, so output always reflects
    what the camera sees now. If the stream drops, it reconnects.
    """

    def __init__(self, source, fps_target: int = 10, reconnect_delay: float = 2.0):
        self.source = source
        self.interval = 1.0 / fps_target
        self.reconnect_delay = reconnect_delay

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None
        self._count = 0

        self._cap = self._open()
        if not self._cap.isOpened():
            self._cap.release()
            raise ConnectionError(f"Cannot open live source: {source}")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open(self):
        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            # TCP avoids the corrupted frames that UDP packet loss can cause
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._count += 1
                continue
            # stream dropped: keep retrying until it comes back
            self._cap.release()
            while not self._stop.is_set():
                time.sleep(self.reconnect_delay)
                cap = self._open()
                if cap.isOpened():
                    self._cap = cap
                    break
                cap.release()

    def read(self):
        """
        Yields (frame_id, timestamp_seconds, frame) at about fps_target.
        frame_id is the camera's capture counter, so gaps in it show how
        many frames were skipped to stay current.
        """
        start = time.monotonic()
        next_time = start
        last_count = 0
        while not self._stop.is_set():
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                frame, count = self._frame, self._count
            if frame is None or count == last_count:
                time.sleep(0.005)  # nothing new yet
                continue
            last_count = count
            yield count, time.monotonic() - start, frame
            next_time = max(next_time + self.interval, time.monotonic())

    def release(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self._cap.release()


def get_reader(source: str, fps_target: int = 10):
    """Factory: pass a video path, '0' for webcam, or an rtsp:// / rtmp:// URL."""
    if source.startswith(("rtsp://", "rtmp://")):
        return LiveReader(source, fps_target)
    if source.isdigit():
        return LiveReader(int(source), fps_target)
    if Path(source).exists():
        return VideoReader(source, fps_target)
    raise ValueError(f"Source not found: {source}")