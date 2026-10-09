"""Tests for the camera hand-control tools and safe startup behavior."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra_plugin_sdk.testing import Harness, fuzz_configs  # noqa: E402

from src.plugin import (  # noqa: E402
    CameraHandControl,
    _current_pid,
    _frame_is_usable,
    _open_camera,
    _pid_is_running,
)


def test_save_calibration_rejects_too_few_points(tmp_path):
    import asyncio
    from src.plugin import CameraHandControl

    with patch("src.plugin.CALIBRATION_FILE", tmp_path / "calibration.json"):
        result = asyncio.run(CameraHandControl().ui_save_calibration(
            mode="gaze", samples=[{"raw": [0.1, 0.2], "target": [0.0, 0.0]}]
        ))
    assert "хотя бы 3" in result["error"]


def test_camera_tools_are_exposed_with_expected_names_and_schemas():
    with Harness(CameraHandControl()) as h:
        assert set(h.tool_names()) == {
            "start_camera_control",
            "stop_camera_control",
            "camera_control_status",
        }
        for name in h.tool_names():
            assert h.schema(name).get("type") == "object"


def test_face_model_is_present_and_readable():
    from src.plugin import FACE_MODEL_FILE

    assert FACE_MODEL_FILE.is_file()
    assert FACE_MODEL_FILE.stat().st_size > 1_000_000


def test_start_refuses_missing_local_model_without_starting_controller(tmp_path):
    with (
        patch("src.plugin.MODEL_FILE", tmp_path / "missing.task"),
        patch("src.plugin.subprocess.Popen") as popen,
    ):
        result = __import__("asyncio").run(CameraHandControl().start_camera_control())
    assert "модель" in result.lower()
    popen.assert_not_called()


def test_status_reports_stopped_when_no_live_pid(tmp_path):
    with patch("src.plugin.PID_FILE", tmp_path / "controller.pid"):
        assert _current_pid() is None


def test_pid_probe_handles_nonexistent_process():
    assert _pid_is_running(-123456789) is False


def test_pid_probe_detects_current_process():
    assert _pid_is_running(__import__("os").getpid()) is True


def test_dashboard_converts_stale_starting_state_to_error_and_clears_pid(tmp_path):
    pid_file = tmp_path / "controller.pid"
    pid_file.write_text("4294967294", encoding="ascii")
    status_file = tmp_path / "status.json"
    status_file.write_text('{"state":"starting"}', encoding="utf-8")
    with (
        patch("src.plugin.PID_FILE", pid_file),
        patch("src.plugin.STATUS_FILE", status_file),
    ):
        status = CameraHandControl._dashboard_state(include_preview=False)

    assert status["running"] is False
    assert status["state"] == "error"
    assert "завершился" in status["message"]
    assert not pid_file.exists()


def test_dashboard_preserves_last_camera_open_error_and_diagnostics(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        '{"state":"camera_unavailable","message":"camera open failed",'
        '"camera_diagnostics":"0/msmf: did not open"}', encoding="utf-8"
    )
    with (
        patch("src.plugin.PID_FILE", tmp_path / "controller.pid"),
        patch("src.plugin.STATUS_FILE", status_file),
    ):
        status = CameraHandControl._dashboard_state(include_preview=False)

    assert status["state"] == "camera_unavailable"
    assert status["running"] is False
    assert "camera open failed" in status["message"]
    assert "0/msmf" in status["camera_diagnostics"]


def test_status_text_includes_backend_diagnostics():
    from src.plugin import _dashboard_status_text

    text = _dashboard_status_text({
        "state": "camera_unavailable",
        "message": "camera unavailable",
        "camera_diagnostics": "0/msmf: did not open",
    })
    assert "camera unavailable" in text
    assert "0/msmf" in text
def test_no_config_payload_crashes_plugin():
    with Harness(CameraHandControl()) as h:
        for payload in fuzz_configs():
            h.set_config(payload)


def test_open_camera_uses_jarvis_backend_order_and_keeps_best_camera():
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    yy, xx = np.indices(frame.shape[:2])
    frame[:, :, 0] = xx % 255
    frame[:, :, 1] = yy % 255
    frame[:, :, 2] = (xx + yy) % 255

    class Capture:
        def __init__(self, index, backend):
            self.index = index
            self.backend = backend
            self.released = False
            self.settings = []

        def set(self, prop, value):
            self.settings.append((prop, value))
            return True

        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            self.released = True

    class FakeCV2:
        CAP_MSMF = 1400
        CAP_DSHOW = 700
        CAP_PROP_BUFFERSIZE = 38

        def __init__(self):
            self.calls = []
            self.captures = []

        def VideoCapture(self, index, backend):
            self.calls.append((index, backend))
            capture = Capture(index, backend)
            self.captures.append(capture)
            return capture

    cv2 = FakeCV2()
    cap, index, backend, diagnostics = _open_camera(cv2, max_index=1, warmup_s=0.02)

    # Jarvis accepts the first usable backend (MSMF before DirectShow).
    assert cv2.calls == [(0, cv2.CAP_MSMF)]
    assert cv2.captures[0].settings == [(cv2.CAP_PROP_BUFFERSIZE, 1)]
    assert cap is cv2.captures[0]
    assert index == 0
    assert backend == "msmf"
    assert "selected camera 0 via msmf" in diagnostics


def test_open_camera_scans_past_gaps_and_selects_available_camera():
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    yy, xx = np.indices(frame.shape[:2])
    frame[:, :, 0] = xx % 255
    frame[:, :, 1] = yy % 255
    frame[:, :, 2] = (xx + yy) % 255

    class Capture:
        def __init__(self, index, works):
            self.index = index
            self.works = works
            self.released = False

        def isOpened(self):
            return self.works

        def read(self):
            return (True, frame) if self.works else (False, None)

        def release(self):
            self.released = True

    class FakeCV2:
        CAP_MSMF = 1
        CAP_DSHOW = 2

        def __init__(self):
            self.calls = []
            self.captures = []

        def VideoCapture(self, index, backend):
            works = index == 3
            cap = Capture(index, works)
            self.calls.append((index, backend))
            self.captures.append(cap)
            return cap

    cv2 = FakeCV2()
    cap, index, backend, diagnostics = _open_camera(cv2, max_index=8, warmup_s=0.02)

    assert index == 3
    assert backend == "msmf"
    assert cap is not None
    assert all(candidate.released for candidate in cv2.captures if candidate is not cap)
    assert (3, cv2.CAP_MSMF) in cv2.calls
    assert "selected camera 3 via msmf" in diagnostics


def test_open_camera_falls_back_from_msmf_to_dshow():
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    yy, xx = np.indices(frame.shape[:2])
    frame[:, :, 0] = xx % 255
    frame[:, :, 1] = yy % 255
    frame[:, :, 2] = (xx + yy) % 255

    class Capture:
        def __init__(self, backend):
            self.backend = backend
            self.released = False

        def isOpened(self):
            return self.backend == 700

        def read(self):
            return (True, frame) if self.isOpened() else (False, None)

        def release(self):
            self.released = True

    class FakeCV2:
        CAP_MSMF = 1400
        CAP_DSHOW = 700

        def __init__(self):
            self.calls = []
            self.captures = []

        def VideoCapture(self, index, backend):
            self.calls.append((index, backend))
            cap = Capture(backend)
            self.captures.append(cap)
            return cap

    cv2 = FakeCV2()
    cap, index, backend, diagnostics = _open_camera(cv2, max_index=1, warmup_s=0.02)

    assert cv2.calls[:2] == [(0, cv2.CAP_MSMF), (0, cv2.CAP_DSHOW)]
    assert cap is cv2.captures[1]
    assert index == 0
    assert backend == "dshow"
    assert cv2.captures[0].released
    assert "selected camera 0 via dshow" in diagnostics


def test_frame_is_usable_rejects_empty_and_flat_frames():
    import numpy as np

    assert not _frame_is_usable(None)
    assert not _frame_is_usable(np.zeros((16, 16), dtype=np.uint8))





def test_gesture_settings_load_and_validate_custom_enabled(tmp_path):
    import json
    from src.plugin import _load_gesture_settings

    settings_file = tmp_path / "gestures.json"
    settings_file.write_text(json.dumps({
        "enabled": {"cursor": False, "clicks": "false", "unknown": False},
        "custom": {"shaka": "left_click", "unknown": "right_click", "point": "bad_action"},
        "custom_enabled": {"shaka": False, "point": False, "ily": False},
    }), encoding="utf-8")
    with patch("src.plugin.GESTURES_FILE", settings_file):
        settings = _load_gesture_settings()

    assert settings["enabled"]["cursor"] is False
    assert settings["enabled"]["clicks"] is True
    assert settings["custom"] == {"shaka": "left_click"}
    assert settings["custom_enabled"] == {"shaka": False}


def test_save_gesture_settings_merges_single_toggle_without_reenabling_others(tmp_path):
    import asyncio
    from src.plugin import DEFAULT_GESTURE_SETTINGS, GESTURES_FILE

    settings_file = tmp_path / "gestures.json"
    payload = {**DEFAULT_GESTURE_SETTINGS, "enabled": {**DEFAULT_GESTURE_SETTINGS["enabled"], "cursor": False}}
    settings_file.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with patch("src.plugin.GESTURES_FILE", settings_file), patch("src.plugin.STATE_DIR", tmp_path):
        result = asyncio.run(CameraHandControl().ui_save_gesture_settings(enabled={"clicks": False}))
        saved = __import__("json").loads(settings_file.read_text(encoding="utf-8"))

    assert result["gesture_settings"]["enabled"]["cursor"] is False
    assert result["gesture_settings"]["enabled"]["clicks"] is False
    assert saved["enabled"]["cursor"] is False


def test_delete_and_reset_custom_gesture_settings(tmp_path):
    import asyncio
    import json
    from src.plugin import DEFAULT_GESTURE_SETTINGS

    settings_file = tmp_path / "gestures.json"
    settings_file.write_text(json.dumps({
        **DEFAULT_GESTURE_SETTINGS,
        "custom": {"shaka": "left_click"},
        "custom_enabled": {"shaka": False},
    }), encoding="utf-8")
    with patch("src.plugin.GESTURES_FILE", settings_file), patch("src.plugin.STATE_DIR", tmp_path):
        plugin = CameraHandControl()
        deleted = asyncio.run(plugin.ui_delete_custom_gesture("shaka"))
        assert "shaka" not in json.loads(settings_file.read_text(encoding="utf-8"))["custom"]
        reset = asyncio.run(plugin.ui_reset_gesture_settings())

    assert "shaka" not in deleted["gesture_settings"]["custom"]
    assert not settings_file.exists()
    assert reset["gesture_settings"] == DEFAULT_GESTURE_SETTINGS


def test_controller_creationflags_suppress_console_window_on_windows():
    from src.plugin import _controller_creationflags

    with patch("src.plugin.os.name", "nt"), patch("src.plugin.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True):
        assert _controller_creationflags() == 0x08000000



def test_controller_receives_and_reloads_gesture_settings_without_touching_camera_fps():
    from src.plugin import _load_gesture_settings

    text = Path(__file__).resolve().parents[1].joinpath("src", "plugin.py").read_text(encoding="utf-8")
    assert "JarvisGestureEngine(pyautogui, _load_gesture_settings())" in text
    assert "current_settings = engine.gesture_settings" in text
    assert text.index("current_settings = engine.gesture_settings") < text.index("while not STOP_FILE.exists():")
    assert "engine.gesture_settings = current_settings" in text
    assert "preview_interval = 1.0 / 30.0" in text
    ui_text = Path(__file__).resolve().parents[1].joinpath("ui", "camera.html").read_text(encoding="utf-8")
    assert 'id="captureCountdown">3<' in ui_text
    assert 'beginCaptureButton' in ui_text
    assert "captureAvailable.includes(pose)" in ui_text
    assert "DASHBOARD_POLL_MS=120" in ui_text
    assert "Promise.race([request,timeout])" in ui_text
    assert "@keyframes aurora-breathe" in ui_text
    assert "#111 !important" not in ui_text
    assert "window.prompt" not in ui_text


def test_dashboard_exposes_gesture_settings_choices(tmp_path):
    import asyncio
    import json
    from src.plugin import DEFAULT_GESTURE_SETTINGS

    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"state": "stopped"}), encoding="utf-8")
    settings_file = tmp_path / "gestures.json"
    settings_file.write_text(json.dumps(DEFAULT_GESTURE_SETTINGS), encoding="utf-8")
    with patch("src.plugin.STATUS_FILE", status_file), patch("src.plugin.PID_FILE", tmp_path / "controller.pid"), patch("src.plugin.GESTURES_FILE", settings_file):
        state = asyncio.run(CameraHandControl().ui_dashboard_state())

    assert state["gesture_settings"] == DEFAULT_GESTURE_SETTINGS
    assert "shaka" in state["gesture_poses"]
    assert "left_click" in state["gesture_actions"]


def test_control_mode_switch_is_exclusive_and_migrates_legacy_gaze(tmp_path):
    import asyncio
    import json
    from src.plugin import DEFAULT_GESTURE_SETTINGS, GESTURES_FILE, _load_gesture_settings

    path=tmp_path/"gestures.json"
    path.write_text(json.dumps({**DEFAULT_GESTURE_SETTINGS,"gaze_enabled":True,"enabled":dict(DEFAULT_GESTURE_SETTINGS["enabled"])}),encoding="utf-8")
    with patch("src.plugin.GESTURES_FILE",path), patch("src.plugin.STATE_DIR",tmp_path):
        assert _load_gesture_settings()["control_mode"] == "gaze"
        result=asyncio.run(CameraHandControl().ui_save_gesture_settings(enabled={"control_mode":"hand"}))
        assert result["gesture_settings"]["gaze_enabled"] is False
        assert result["gesture_settings"]["enabled"]["cursor"] is True
        result=asyncio.run(CameraHandControl().ui_save_gesture_settings(enabled={"control_mode":"gaze"}))
        assert result["gesture_settings"]["gaze_enabled"] is True
        assert result["gesture_settings"]["enabled"]["cursor"] is False


def test_dashboard_keeps_gaze_backend_but_ui_is_hand_only():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    controller=(root/"src"/"plugin.py").read_text(encoding="utf-8")
    gaze=(root/"src"/"calibration.py").read_text(encoding="utf-8")
    ui=(root/"ui"/"camera.html").read_text(encoding="utf-8")
    assert "sample_id=frame_id" in controller
    assert "control_mode=control_mode" in controller or "control_mode=current_settings.get" in controller
    assert 'gaze_controller.update(' in controller
    assert 'state_name = "gaze_active" if face_features and mapped is not None else "gaze_calibrating"' in controller
    assert "self.pg.moveTo(max(2, min(sw - 3" in gaze
    assert 'id="controlMode"' not in ui
    assert 'data-calibrate="gaze"' not in ui
    assert 'id="calibrationStage"' not in ui
    assert 'ensureHandOnly' in ui
    assert "preview_interval = 1.0 / 30.0" in controller


def test_preview_uses_high_contrast_hand_skeleton_without_camera_model_banner(tmp_path):
    from types import SimpleNamespace
    from src.plugin import _write_preview

    class Frame:
        shape = (270, 480, 3)

    class FakeCV2:
        INTER_AREA = 1
        LINE_AA = 2
        IMWRITE_JPEG_QUALITY = 3

        def __init__(self):
            self.lines = []
            self.circles = []

        def flip(self, frame, axis):
            return frame

        def resize(self, frame, size, interpolation):
            return frame

        def line(self, image, start, end, color, thickness, line_type):
            self.lines.append((color, thickness))

        def circle(self, image, center, radius, color, fill, line_type):
            self.circles.append(color)

        def imencode(self, extension, image, params):
            return True, SimpleNamespace(tobytes=lambda: b"jpeg")

    import src.plugin as plugin_module

    fake = FakeCV2()
    landmarks = [SimpleNamespace(x=index / 20, y=index / 20) for index in range(21)]
    with patch.object(plugin_module, "PREVIEW_FILE", tmp_path / "preview.jpg"):
        _write_preview(fake, Frame(), landmarks, True, False, 0, "point")

    assert len(fake.lines) == 42  # dark outline plus bright cyan for 21 bones
    assert fake.lines.count(((12, 16, 28), 8)) == 21
    assert fake.lines.count(((255, 220, 72), 4)) == 21
    assert set(fake.circles) == {(12, 16, 28), (245, 250, 255), (105, 255, 100)}
    assert (tmp_path / "preview.jpg").read_bytes() == b"jpeg"


def test_hand_tracking_status_is_cleared_when_no_hand_is_visible():
    text = Path(__file__).resolve().parents[1].joinpath("src", "plugin.py").read_text(encoding="utf-8")
    no_hand_branch = text.split("            if hand is None:", 1)[1].split("            world_set =", 1)[0]
    assert '_write_status("waiting_for_hand"' in no_hand_branch
    assert "hand_detected=False" in no_hand_branch


def test_mediapipe_timestamps_are_strictly_monotonic_in_milliseconds():
    from src.plugin import _next_timestamp_ms

    first = _next_timestamp_ms(12.3451, 0)
    duplicate_clock_tick = _next_timestamp_ms(12.3452, first)
    clock_catches_up = _next_timestamp_ms(12.3470, duplicate_clock_tick)

    assert (first, duplicate_clock_tick, clock_catches_up) == (12345, 12346, 12347)
    assert _next_timestamp_ms(12.4000, clock_catches_up) == 12400
