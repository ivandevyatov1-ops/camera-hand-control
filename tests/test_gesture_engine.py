"""Camera-free tests for the JARVIS gesture port."""
from __future__ import annotations

import sys
import pytest

from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gesture_engine import (  # noqa: E402
    COMMAND_POSES,
    FeatureTracker,
    JarvisGestureEngine,
    classify_fingers,
)


def test_fit_affine_and_map_calibrated_grid():
    from src.calibration import fit_affine, map_calibrated, target_grid

    targets = target_grid(3, 3)
    # Synthetic camera coordinates are inverted/scaled relative to screen targets.
    samples = [{"raw": [0.9 - x * 0.7, 0.1 + y * 0.8], "target": [x, y]}
               for x, y in targets]
    fitted = fit_affine(samples)
    mapped = map_calibrated(samples, [0.55, 0.5])

    assert fitted is not None
    assert mapped == pytest.approx([0.5, 0.5], abs=0.02)
    assert len(targets) == 9


def test_affine_rejects_collinear_calibration_points():
    from src.calibration import fit_affine

    samples = [{"raw": [x, x], "target": [x, x]} for x in (0.1, 0.5, 0.9)]
    assert fit_affine(samples) is None


def test_blink_series_maps_two_and_three_blinks_but_ignores_single():
    from src.calibration import BlinkSeries

    def blink(series, at):
        series.update(0.1, at)
        series.update(0.1, at + 0.08)
        return series.update(0.3, at + 0.16)

    series = BlinkSeries()
    blink(series, 0.0)
    assert series.update(0.3, 1.0) is None

    series = BlinkSeries()
    blink(series, 0.0)
    blink(series, 0.3)
    assert series.update(0.3, 1.3) == "left_click"

    series = BlinkSeries()
    blink(series, 0.0)
    blink(series, 0.3)
    blink(series, 0.6)
    assert series.update(0.3, 1.6) == "double_click"


@pytest.mark.parametrize("open_ratio", [0.18, 0.24, 0.30, 0.36])
def test_blink_series_counts_relative_closed_ratio(open_ratio):
    from src.calibration import BlinkSeries

    series = BlinkSeries()
    # Learn a user's natural open-eye ratio, then close both eyes briefly.
    for i in range(20):
        series.update(open_ratio, i * 0.03)
    close_ratio = open_ratio * 0.48
    series.update(close_ratio, 1.0)
    series.update(close_ratio, 1.08)
    series.update(open_ratio, 1.16)
    series.update(open_ratio, 1.30)
    series.update(close_ratio, 1.40)
    series.update(close_ratio, 1.48)
    series.update(open_ratio, 1.56)
    assert series.update(open_ratio, 2.40) == "left_click"


def test_gaze_controller_moves_without_calibration_and_clicks_on_blinks():
    from src.calibration import GazeController

    mouse = FakeMouse()
    controller = GazeController(mouse)
    mapped, event = controller.update([0.5, 0.5], 0.3, 1.0)
    assert mapped is not None
    assert mouse.moves

    for at, ratio in ((1.1, 0.3), (1.18, 0.12), (1.26, 0.12), (1.34, 0.3),
                      (1.45, 0.12), (1.53, 0.12), (1.61, 0.3)):
        _, event = controller.update([0.5, 0.5], ratio, at)
    controller.update([0.5, 0.5], 0.3, 2.5)
    assert any(call[1].get("button") == "left" for call in mouse.clicks)


def test_gaze_features_reject_missing_iris_points():
    from src.calibration import face_gaze_features

    assert face_gaze_features([SimpleNamespace(x=0.5, y=0.5)] * 468) is None


def test_classify_jarvis_finger_poses():
    assert classify_fingers([0, 1, 0, 0, 0]) == "point"
    assert classify_fingers([0, 1, 1, 0, 0]) == "victory"
    assert classify_fingers([1, 1, 1, 1, 1]) == "palm"
    assert classify_fingers([0, 0, 0, 0, 0]) == "fist"
    assert classify_fingers([0, 1, 1, 1, 1]) == "four"
    assert classify_fingers([1, 1, 0, 0, 1]) == "ily"
    assert {"three", "horns", "shaka", "four"}.issubset(COMMAND_POSES)


def test_feature_tracker_returns_required_local_features():
    # A deterministic, non-degenerate landmark cloud; this verifies the data contract,
    # not gesture accuracy (which requires an actual camera frame).
    points = [SimpleNamespace(x=0.4 + (i % 5) * 0.025, y=0.3 + (i // 5) * 0.035, z=0.0)
              for i in range(21)]
    tracker = FeatureTracker()
    features = tracker.update(points, None, 4 / 3)

    assert set(("fingers", "pinch", "pinch_palm", "index_ang", "mid_pinch",
                "mid_palm", "mid_ang", "mid_far", "spread", "anchor", "palm",
                "size", "aspect", "thumb_ratio")).issubset(features)
    assert len(features["fingers"]) == 5
    assert features["aspect"] == 4 / 3


class FakeMouse:
    def __init__(self):
        self.moves = []
        self.clicks = []
        self.scrolls = []
        self.releases = 0
        self.position_value = SimpleNamespace(x=50, y=40)

    def size(self):
        return 100, 80

    def position(self):
        return self.position_value

    def moveTo(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        if args:
            self.position_value = SimpleNamespace(x=args[0], y=args[1])

    def mouseUp(self, *args, **kwargs):
        self.releases += 1

    def mouseDown(self, *args, **kwargs):
        pass

    def click(self, *args, **kwargs):
        self.clicks.append((args, kwargs))

    def scroll(self, amount):
        self.scrolls.append(amount)


def test_point_pose_drives_cursor_without_camera_or_os_calls():
    mouse = FakeMouse()
    engine = JarvisGestureEngine(mouse)
    features = {
        "fingers": [0, 1, 0, 0, 0], "size": 0.16, "aspect": 4 / 3,
        "anchor": (0.58, 0.46), "spread": 0.0, "palm": (0.5, 0.5),
        "pinch": 1.0, "pinch_palm": 0.0, "index_ang": 170.0,
        "mid_pinch": 1.0, "mid_palm": 0.0, "mid_ang": 170.0,
        "mid_far": 0.0, "thumb_ratio": 0.0,
    }
    engine.update(features, 1.0)
    engine.update(features, 1.04)
    assert mouse.moves
    assert mouse.moves[-1][0][0] >= 2 and mouse.moves[-1][0][1] >= 2
    assert engine.last_label == "point"
    assert not mouse.clicks
    engine.shutdown()


def test_unbound_command_poses_are_recognized_but_never_execute_external_actions():
    mouse = FakeMouse()
    engine = JarvisGestureEngine(mouse)
    engine._command_pose("shaka", 1.0, 0.5)
    assert engine.last_label == "shaka"
    assert not mouse.clicks
    assert not mouse.scrolls
    engine.shutdown()


def test_shutdown_always_releases_mouse_buttons():
    mouse = FakeMouse()
    engine = JarvisGestureEngine(mouse)
    engine.shutdown()
    assert mouse.releases >= 1



def _features_for_pose(fingers):
    return {
        "fingers": fingers, "size": 0.16, "aspect": 4 / 3,
        "anchor": (0.58, 0.46), "spread": 0.0, "palm": (0.5, 0.5),
        "pinch": 1.0, "pinch_palm": 0.0, "index_ang": 170.0,
        "mid_pinch": 1.0, "mid_palm": 0.0, "mid_ang": 170.0,
        "mid_far": 0.0, "thumb_ratio": 0.0,
    }


def test_custom_binding_fires_once_and_disabled_custom_binding_does_not_fire():
    mouse = FakeMouse()
    settings = {
        "enabled": {"cursor": True, "clicks": True, "scroll": True, "window": True, "minimize": True},
        "custom": {"shaka": "left_click"},
        "custom_enabled": {"shaka": True},
    }
    engine = JarvisGestureEngine(mouse, settings)
    features = _features_for_pose([1, 0, 0, 0, 1])

    engine.update(features, 1.0)
    engine.update(features, 1.04)
    engine.update(features, 1.08)
    assert len(mouse.clicks) == 1

    settings["custom_enabled"]["shaka"] = False
    engine.gesture_settings = settings
    engine.custom_fired = None
    engine.update(features, 1.12)
    assert len(mouse.clicks) == 1
    assert engine.last_label == "shaka"
    engine.shutdown()


def test_disabled_scroll_and_minimize_switches_block_their_handlers():
    mouse = FakeMouse()
    settings = {"enabled": {"scroll": False, "minimize": False}, "custom": {}}
    engine = JarvisGestureEngine(mouse, settings)
    calls = []
    engine._victory = lambda features, now: calls.append("scroll")
    engine._fist = lambda features, now: calls.append("minimize")

    victory = _features_for_pose([0, 1, 1, 0, 0])
    fist = _features_for_pose([0, 0, 0, 0, 0])
    engine.update(victory, 1.0)
    engine.update(victory, 1.04)
    engine.update(fist, 1.08)
    engine.update(fist, 1.12)

    assert calls == []
    assert engine.last_label == "fist"
    engine.shutdown()
