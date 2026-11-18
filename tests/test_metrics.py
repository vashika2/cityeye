from types import SimpleNamespace

import pytest

from src.utils.metrics import (
    Detection, AnomalyTracker, classify_congestion, detect_helmet_violations,
)

THRESH = SimpleNamespace(free_flow=5, moderate=15, dense=30)


def det(cls="car", box=(100, 100, 200, 180), conf=0.9, tid=1):
    return Detection(bbox=list(box), confidence=conf, class_id=0, class_name=cls, track_id=tid)


def shifted(box, dx):
    return (box[0] + dx, box[1], box[2] + dx, box[3])


# ---------- congestion thresholds ----------

@pytest.mark.parametrize("count,expected", [
    (0, "free_flow"), (5, "free_flow"), (6, "moderate"), (15, "moderate"),
    (16, "dense"), (30, "dense"), (31, "gridlock"), (5.5, "moderate"),
])
def test_congestion_boundaries(count, expected):
    assert classify_congestion(count, THRESH) == expected


# ---------- helmet heuristic ----------

def test_flat_motorcycle_is_flagged():
    d = det("motorcycle", box=(100, 100, 200, 150))          # h/w = 0.5
    assert len(detect_helmet_violations([d], 0.6)) == 1

def test_upright_motorcycle_is_not_flagged():
    d = det("motorcycle", box=(100, 100, 150, 200))          # h/w = 2.0
    assert detect_helmet_violations([d], 0.6) == []

def test_low_confidence_motorcycle_is_ignored():
    d = det("motorcycle", box=(100, 100, 200, 150), conf=0.4)
    assert detect_helmet_violations([d], 0.6) == []

def test_non_motorcycles_are_never_flagged():
    d = det("car", box=(100, 100, 200, 150))
    assert detect_helmet_violations([d], 0.6) == []


# ---------- stalled-vehicle tracker ----------

def test_stationary_car_is_flagged_after_threshold():
    t = AnomalyTracker(stopped_frames_threshold=30, min_movement_px=5.0)
    flagged_at = None
    for i in range(60):
        if t.update([det()]) and flagged_at is None:
            flagged_at = i
    assert flagged_at == 30          # first frame only sets position, then 30 still frames

def test_moving_car_is_not_flagged():
    t = AnomalyTracker(30, 5.0)
    box = (100, 100, 200, 180)
    for i in range(100):
        assert t.update([det(box=shifted(box, i * 20))]) == []

def test_motorcycle_is_never_flagged_as_stalled():
    t = AnomalyTracker(30, 5.0)
    for _ in range(60):
        assert t.update([det("motorcycle")]) == []

def test_untracked_detections_are_ignored():
    t = AnomalyTracker(30, 5.0)
    for _ in range(60):
        assert t.update([det(tid=None)]) == []


# ---------- known weaknesses (xfail until fixed) ----------

@pytest.mark.xfail(reason="compares to previous frame, so slow creeping counts as stopped")
def test_slowly_creeping_car_is_not_flagged():
    t = AnomalyTracker(30, 5.0)
    box = (100, 100, 200, 180)
    flagged = False
    for i in range(100):                       # 4 px per frame = 400 px in total
        if t.update([det(box=shifted(box, i * 4))]):
            flagged = True
    assert not flagged

@pytest.mark.xfail(reason="alerts every frame once the threshold is passed")
def test_alert_is_not_repeated_every_frame():
    t = AnomalyTracker(30, 5.0)
    alerts = sum(len(t.update([det()])) for _ in range(100))
    assert alerts == 1

@pytest.mark.xfail(reason="state for vanished tracks is never cleaned up")
def test_state_does_not_grow_forever():
    t = AnomalyTracker(30, 5.0)
    for i in range(1000):                      # 1000 different vehicles, each seen once
        t.update([det(tid=i)])
    assert len(t._prev_positions) < 100