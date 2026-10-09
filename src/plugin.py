"""Local camera hand control for Astra.

The controller only uses the bundled MediaPipe model and local camera frames.
No image, landmark, or mouse data is sent over the network.
"""
from __future__ import annotations

import base64
import importlib
import queue
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from astra_plugin_sdk import Plugin, tool, ui_call, ui_page

PLUGIN_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Astra" / "camera-hand-control"
PID_FILE = STATE_DIR / "controller.pid"
STOP_FILE = STATE_DIR / "stop.request"
STATUS_FILE = STATE_DIR / "status.json"
LOG_FILE = STATE_DIR / "controller.log"
PREVIEW_FILE = STATE_DIR / "preview.jpg"
GESTURES_FILE = STATE_DIR / "gestures.json"
MODEL_FILE = PLUGIN_DIR / "src" / "models" / "hand_landmarker.task"
FACE_MODEL_FILE = PLUGIN_DIR / "src" / "models" / "face_landmarker.task"
CALIBRATION_FILE = STATE_DIR / "calibration.json"
CUSTOM_POSES = frozenset({"thumb", "four", "three", "horns", "shaka", "ily", "victory", "fist", "palm", "point"})
GESTURE_ACTIONS = {
    "left_click": "Левый щелчок",
    "right_click": "Правый щелчок",
    "double_click": "Двойной щелчок",
    "scroll_up": "Прокрутить вверх",
    "scroll_down": "Прокрутить вниз",
    "minimize_window": "Свернуть активное окно",
}
DEFAULT_GESTURE_SETTINGS = {
    "enabled": {"cursor": True, "clicks": True, "scroll": True, "window": True, "minimize": True},
    "custom": {},
    "custom_enabled": {},
    "gaze_enabled": False,
    "control_mode": "hand",
    "gaze_clicks_enabled": True,
}
DEFAULT_CALIBRATION = {"hand": [], "gaze": []}
FFMPEG_CANDIDATES = (
    Path(r"C:\Program Files\JARVIS\tools\ffmpeg\ffmpeg.exe"),
    Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
)
CAMERA_DEVICE_NAME = "DV20 USB CAMERA"


def _controller_creationflags() -> int:
    """Run the background controller without opening a console window on Windows."""
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _ffmpeg_executable() -> str | None:
    """Find FFmpeg for Windows capture when OpenCV's camera backends cannot open devices."""
    for candidate in FFMPEG_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def _controller_python() -> str:
    """Prefer this plugin's dependency environment over Astra's host Python."""
    if os.name == "nt":
        candidate = PLUGIN_DIR / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PLUGIN_DIR / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def _create_hand_landmarker(vision, mp_python, options):
    """Create MediaPipe's native task, working around its missing Windows CRT free export."""
    if os.name != "nt":
        return vision.HandLandmarker.create_from_options(options)

    import ctypes
    bindings = importlib.import_module("mediapipe.tasks.python.core.mediapipe_c_bindings")

    original_cdll = ctypes.CDLL
    crt = original_cdll("ucrtbase.dll")
    crt_free = crt.free
    crt_free.argtypes = [ctypes.c_void_p]
    crt_free.restype = None

    def load_with_crt_free(path, *args, **kwargs):
        library = original_cdll(path, *args, **kwargs)
        if str(path).lower().endswith("libmediapipe.dll"):
            library.free = crt_free
        return library

    # MediaPipe 0.10.30 asks libmediapipe.dll for a bare `free` export, which
    # that DLL doesn't provide on Windows. Its allocations use the Windows CRT.
    bindings.ctypes.CDLL = load_with_crt_free
    try:
        return vision.HandLandmarker.create_from_options(options)
    finally:
        bindings.ctypes.CDLL = original_cdll


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a reliable liveness probe on Windows: it can
        # succeed for a PID whose process has already exited. Wait on a real
        # process handle instead; WAIT_TIMEOUT means it is still running.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
            if result == 0x00000102:  # WAIT_TIMEOUT
                return True
            if result == 0x00000000:  # WAIT_OBJECT_0
                return False
        finally:
            kernel32.CloseHandle(handle)
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _current_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
        return pid if _pid_is_running(pid) else None
    except (OSError, ValueError):
        return None


def _load_gesture_settings() -> dict:
    settings = json.loads(json.dumps(DEFAULT_GESTURE_SETTINGS))
    try:
        stored = json.loads(GESTURES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    if not isinstance(stored, dict):
        return settings
    if isinstance(stored.get("enabled"), dict):
        for name in settings["enabled"]:
            if isinstance(stored["enabled"].get(name), bool):
                settings["enabled"][name] = stored["enabled"][name]
        for key in ("gaze_enabled", "gaze_clicks_enabled", "control_mode"):
            value = stored["enabled"].get(key)
            if key == "control_mode":
                if value in ("hand", "gaze"):
                    settings[key] = value
            elif isinstance(value, bool):
                settings[key] = value
    custom = stored.get("custom")
    if isinstance(custom, dict):
        settings["custom"] = {
            pose: action for pose, action in custom.items()
            if pose in CUSTOM_POSES and action in GESTURE_ACTIONS
        }
    custom_enabled = stored.get("custom_enabled")
    if isinstance(custom_enabled, dict):
        settings["custom_enabled"] = {
            pose: value for pose, value in custom_enabled.items()
            if pose in CUSTOM_POSES and isinstance(value, bool)
        }
    # An enabled flag has meaning only while its pose has a custom binding.
    settings["custom_enabled"] = {
        pose: value for pose, value in settings["custom_enabled"].items()
        if pose in settings["custom"]
    }
    for key in ("gaze_enabled", "gaze_clicks_enabled"):
        if isinstance(stored.get(key), bool):
            settings[key] = stored[key]
    if stored.get("control_mode") in ("hand", "gaze"):
        settings["control_mode"] = stored["control_mode"]
    if stored.get("gaze_enabled") is True:
        settings["control_mode"] = "gaze"
    if settings.get("control_mode") == "gaze":
        settings["gaze_enabled"] = True
        settings["enabled"]["cursor"] = False
    else:
        settings["gaze_enabled"] = False
    return settings


def _load_calibration() -> dict:
    try:
        stored = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_CALIBRATION))
    if not isinstance(stored, dict):
        return json.loads(json.dumps(DEFAULT_CALIBRATION))
    clean = {}
    for mode in ("hand", "gaze"):
        samples = stored.get(mode, [])
        clean[mode] = [s for s in samples if isinstance(s, dict) and len(s.get("raw", [])) == 2 and len(s.get("target", [])) == 2][:16] if isinstance(samples, list) else []
    return clean


def _save_calibration(calibration: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CALIBRATION_FILE)


def _save_gesture_settings(settings: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GESTURES_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, GESTURES_FILE)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _reset_gesture_settings() -> dict:
    GESTURES_FILE.unlink(missing_ok=True)
    return json.loads(json.dumps(DEFAULT_GESTURE_SETTINGS))


@ui_page("camera-hand-control", "Управление камерой", "camera.html")
class CameraHandControl(Plugin):
    """Opt-in hand tracking and mouse control using a local webcam."""

    @tool("Запусти локальное управление курсором рукой через камеру. Видео не отправляется в интернет.")
    async def start_camera_control(self):
        """Запустить отслеживание кисти. Щипок большим и указательным пальцами нажимает; удержание перетаскивает."""
        if not MODEL_FILE.is_file():
            return "Не найдена локальная модель распознавания руки. Переустанови папку плагина."
        if _load_gesture_settings().get("gaze_enabled") and not FACE_MODEL_FILE.is_file():
            return "Не найдена локальная модель лица. Скачай face_landmarker.task в папку models."
        running = _current_pid()
        if running:
            return f"Управление камерой уже работает (процесс {running}). Чтобы остановить, вызови stop_camera_control."
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STOP_FILE.unlink(missing_ok=True)
        log = LOG_FILE.open("ab", buffering=0)
        flags = _controller_creationflags()
        try:
            proc = subprocess.Popen(
                [_controller_python(), "-m", "src.plugin", "--controller"],
                cwd=str(PLUGIN_DIR),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=True,
            )
        finally:
            log.close()
        PID_FILE.write_text(str(proc.pid), encoding="ascii")
        await self.log_info(f"Local camera controller started: pid={proc.pid}")
        return "Запускаю локальное управление рукой. Наведи указательный палец, щипок большим с указательным нажимает, удержание щипка перетаскивает. Для остановки скажи: останови управление камерой."

    @ui_call("start_camera")
    async def ui_start_camera(self):
        message = await self.start_camera_control()
        return {"message": message, **self._dashboard_state(include_preview=False)}

    @ui_call("stop_camera")
    async def ui_stop_camera(self):
        message = await self.stop_camera_control()
        return {"message": message, **self._dashboard_state(include_preview=False)}

    @ui_call("dashboard_state")
    async def ui_dashboard_state(self):
        status = self._dashboard_state(include_preview=True)
        status["gesture_settings"] = _load_gesture_settings()
        if status.get("state") in {"waiting_for_hand", "active", "dragging", "gaze_active", "gaze_calibrating"}:
            status["state"] = "gaze_active" if status.get("gesture_settings", {}).get("control_mode") == "gaze" and status.get("gaze_detected") else ("gaze_calibrating" if status.get("gesture_settings", {}).get("control_mode") == "gaze" else status.get("state"))
        status["gesture_actions"] = GESTURE_ACTIONS
        status["gesture_poses"] = sorted(CUSTOM_POSES)
        status["calibration"] = _load_calibration()
        status["face_model_available"] = FACE_MODEL_FILE.is_file()
        return status

    @ui_call("save_calibration")
    async def ui_save_calibration(self, mode=None, samples=None):
        if mode == "hand":
            return {"error": "Калибровка руки отключена. Управление рукой работает без неё."}
        if mode != "gaze" or not isinstance(samples, list):
            return {"error": "Некорректные данные калибровки."}
        calibration = _load_calibration()
        clean = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            raw, target = sample.get("raw"), sample.get("target")
            if not isinstance(raw, list) or not isinstance(target, list) or len(raw) != 2 or len(target) != 2:
                continue
            try:
                raw = [max(0.0, min(1.0, float(v))) for v in raw]
                target = [max(0.0, min(1.0, float(v))) for v in target]
            except (TypeError, ValueError):
                continue
            clean.append({"raw": raw, "target": target})
        if len(clean) < 3:
            return {"error": "Для калибровки нужны хотя бы 3 устойчиво распознанные точки."}
        calibration[mode] = clean[:16]
        _save_calibration(calibration)
        return {"message": "Калибровка сохранена локально.", "calibration": calibration}

    @ui_call("save_gesture_settings")
    async def ui_save_gesture_settings(self, enabled=None, custom=None, custom_enabled=None):
        settings = _load_gesture_settings()
        if isinstance(enabled, dict):
            for name in settings["enabled"]:
                if name in enabled and isinstance(enabled[name], bool):
                    settings["enabled"][name] = enabled[name]
            for key in ("gaze_enabled", "gaze_clicks_enabled", "control_mode"):
                value = enabled.get(key)
                if key == "control_mode":
                    if value in ("hand", "gaze"):
                        settings[key] = value
                elif isinstance(value, bool):
                    settings[key] = value
        if settings.get("control_mode") == "gaze":
            settings["gaze_enabled"] = True
            settings["enabled"]["cursor"] = False
        else:
            settings["gaze_enabled"] = False
            if isinstance(enabled, dict) and enabled.get("control_mode") == "hand":
                settings["enabled"]["cursor"] = True
        if isinstance(custom, dict):
            clean = {}
            for pose, action in custom.items():
                if pose not in CUSTOM_POSES or action not in GESTURE_ACTIONS:
                    return {"error": "Выбери распознаваемую позу и действие из списка."}
                clean[pose] = action
            settings["custom"] = clean
        if isinstance(custom_enabled, dict):
            for pose, value in custom_enabled.items():
                if pose in settings["custom"] and isinstance(value, bool):
                    settings["custom_enabled"][pose] = value
        _save_gesture_settings(settings)
        return {"message": "Настройки жестов сохранены.", "gesture_settings": settings}

    @ui_call("delete_custom_gesture")
    async def ui_delete_custom_gesture(self, pose=None):
        if pose not in CUSTOM_POSES:
            return {"error": "Неизвестный жест."}
        settings = _load_gesture_settings()
        if pose not in settings["custom"]:
            return {"error": "Для этой позы нет пользовательской привязки."}
        settings["custom"].pop(pose, None)
        settings["custom_enabled"].pop(pose, None)
        _save_gesture_settings(settings)
        return {"message": "Пользовательский жест удалён.", "gesture_settings": settings}

    @ui_call("reset_gesture_settings")
    async def ui_reset_gesture_settings(self):
        settings = _reset_gesture_settings()
        return {"message": "Настройки жестов восстановлены по умолчанию.", "gesture_settings": settings}

    @staticmethod
    def _dashboard_state(include_preview: bool) -> dict:
        pid = _current_pid()
        try:
            status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {"state": "starting"} if pid else {"state": "stopped"}
        status["running"] = pid is not None
        if pid is not None:
            status["pid"] = pid
            if status.get("state") == "starting":
                try:
                    age = time.time() - float(status.get("updated", time.time()))
                except (TypeError, ValueError):
                    age = 0
                if age > 45:
                    status["state"] = "error"
                    status["message"] = "Контроллер слишком долго запускается. Останови его и попробуй снова."
        elif status.get("state") in {
            "starting", "active", "waiting_for_hand", "dragging"
        }:
            # Do not let a stale controller status leave the embedded page in
            # an endless loading state after the child process has exited.
            status["state"] = "error"
            status["message"] = "Контроллер завершился. Нажми «Запустить камеру», чтобы попробовать снова."
            PID_FILE.unlink(missing_ok=True)
        if include_preview and pid is not None:
            try:
                data = PREVIEW_FILE.read_bytes()
                if len(data) <= 400_000:
                    status["preview"] = base64.b64encode(data).decode("ascii")
            except OSError:
                pass
        return status

    @tool("Останови запущенное управление мышью через камеру и освободи кнопку мыши.")
    async def stop_camera_control(self):
        """Безопасно остановить управление и завершить возможное перетаскивание."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STOP_FILE.write_text("stop", encoding="ascii")
        pid = _current_pid()
        if pid is None:
            PID_FILE.unlink(missing_ok=True)
            return "Управление камерой уже остановлено."
        return "Отправила сигнал остановки. Управление мышью остановится в ближайший момент."

    @tool("Проверь, включено ли сейчас локальное управление курсором через камеру.")
    async def camera_control_status(self):
        """Return controller or last camera-open diagnostic without opening the camera."""
        pid = _current_pid()
        details = {}
        try:
            details = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if pid is None:
            state = details.get("state", "stopped")
            if state in {"camera_unavailable", "error"}:
                reason = details.get("message", details.get("camera_diagnostics", "проверь подключение и разрешение Windows"))
                return f"Управление камерой выключено. Последняя ошибка: {reason}"
            return "Управление камерой выключено."
        return f"Управление камерой работает (процесс {pid}). Состояние: {details.get('state', 'запуск')}."


def _write_status(state: str, **extra) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "updated": time.time(), **extra}
    tmp = STATUS_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATUS_FILE)
    except OSError:
        tmp.unlink(missing_ok=True)


def _write_preview(cv2, frame, hand, controlling: bool, pinching: bool, camera_index: int, gesture: str = "") -> None:
    """Publish a compact mirrored preview with a high-contrast, Quest-inspired hand skeleton."""
    preview = cv2.flip(frame, 1)
    preview = cv2.resize(preview, (480, 270), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = 480 / frame.shape[1], 270 / frame.shape[0]
    if hand:
        # The detector runs on the unmirrored camera frame; reflect x to match preview.
        points = [(int((1.0 - p.x) * scale_x), int(p.y * scale_y)) for p in hand]
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
            (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        )
        # Dark under-strokes keep cyan bones visible over bright clothing;
        # larger white joints and lime fingertips make the tracked pose obvious.
        bone_color = (255, 220, 72)       # bright cyan in BGR
        joint_color = (245, 250, 255)
        fingertip_color = (105, 255, 100)
        for a, b in connections:
            cv2.line(preview, points[a], points[b], (12, 16, 28), 8, cv2.LINE_AA)
            cv2.line(preview, points[a], points[b], bone_color, 4, cv2.LINE_AA)
        for i, point in enumerate(points):
            tip = i in (4, 8, 12, 16, 20)
            cv2.circle(preview, point, 8 if tip else 7, (12, 16, 28), -1, cv2.LINE_AA)
            cv2.circle(preview, point, 5 if tip else 4, fingertip_color if tip else joint_color, -1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return
    tmp = PREVIEW_FILE.with_suffix(".tmp")
    try:
        tmp.write_bytes(encoded.tobytes())
        os.replace(tmp, PREVIEW_FILE)
    except OSError:
        tmp.unlink(missing_ok=True)


def _frame_is_usable(frame) -> bool:
    """Reject empty, flat or privacy-shutter frames, as JARVIS does."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    try:
        small = frame[::8, ::8]
        gray = small.mean(axis=2) if small.ndim == 3 else small
        mean = float(gray.mean())
        std = float(gray.std())
    except Exception:
        return False
    if std < 1.5:
        return False
    if std < 4.0 and (mean < 8.0 or mean > 247.0):
        return False
    return True


def _score_camera_frame(frame, fps: float = 0.0) -> float:
    """Prefer a fast RGB webcam over IR or virtual cameras, like JARVIS."""
    if not _frame_is_usable(frame):
        return -1.0
    h, w = frame.shape[:2]
    score = float(min(w, 1920)) / 32.0 + float(min(h, 1080)) / 48.0
    if 480 <= min(h, w) <= 1080:
        score += 12.0
    if frame.ndim == 3 and frame.shape[2] >= 3:
        score += 80.0
        small = frame[::4, ::4, :3].astype("int16")
        blue, green, red = small[:, :, 0], small[:, :, 1], small[:, :, 2]
        chroma = float(abs(red - green).mean() + abs((red + green) // 2 - blue).mean())
        score += chroma * 0.35
        score += float(small.mean(axis=2).std()) * 0.15
        if chroma < 6.0:
            score -= 55.0
    else:
        score += 8.0
    score += min(max(fps, 0.0), 30.0) * 2.0
    return score


def _open_ffmpeg_camera(cv2, executable: str | None = None):
    """Capture the named DirectShow webcam through FFmpeg and expose OpenCV-like frames."""
    executable = executable or _ffmpeg_executable()
    if not executable or os.name != "nt":
        return None, None

    import numpy as np

    # The webcam supports MJPEG up to 1920x1080 at 30 fps. Decode at full
    # detail, then let the preview scale to its widget size.
    width, height, fps = 1280, 720, 30
    command = [
        executable, "-hide_banner", "-loglevel", "error",
        "-f", "dshow", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-vcodec", "mjpeg", "-i", f"video={CAMERA_DEVICE_NAME}",
        "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=width * height * 3 * 2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, ValueError) as exc:
        return None, f"ffmpeg dshow: {exc}"

    class FFmpegCapture:
        def __init__(self):
            self.process = process
            self.width = width
            self.height = height
            self.frame_bytes = width * height * 3
            self.frames = queue.Queue(maxsize=1)
            self.reader = threading.Thread(target=self._read_frames, daemon=True, name="camera-ffmpeg-reader")
            self.reader.start()

        def _read_frames(self):
            # Never let a silent camera block controller startup or UI updates.
            try:
                while self.process.poll() is None and self.process.stdout is not None:
                    chunks = bytearray()
                    while len(chunks) < self.frame_bytes:
                        part = self.process.stdout.read(self.frame_bytes - len(chunks))
                        if not part:
                            return
                        chunks.extend(part)
                    frame = np.frombuffer(chunks, dtype=np.uint8).reshape((self.height, self.width, 3))
                    try:
                        self.frames.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            pass
            except (OSError, ValueError):
                return

        def isOpened(self):
            return self.process.poll() is None and self.process.stdout is not None

        def read(self):
            if not self.isOpened():
                return False, None
            try:
                return True, self.frames.get(timeout=5.0)
            except queue.Empty:
                return False, None

        def get(self, prop):
            if prop == getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3):
                return self.width
            if prop == getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4):
                return self.height
            return 0

        def release(self):
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            if self.process.stdout:
                try:
                    self.process.stdout.close()
                except OSError:
                    pass
            if self.reader.is_alive():
                self.reader.join(timeout=1)

    cap = FFmpegCapture()
    ok, frame = cap.read()
    if not ok or not _frame_is_usable(frame):
        cap.release()
        return None, "ffmpeg dshow: устройство открылось, но не поступил пригодный кадр"
    details = f"ffmpeg dshow {CAMERA_DEVICE_NAME} ({width}x{height} @ {fps} fps)"
    return cap, details


def _open_camera(cv2, max_index: int = 16, warmup_s: float = 1.6):
    """Scan cameras in JARVIS order and keep the best frame-producing device."""
    backends = []
    for name in ("MSMF", "DSHOW", "ANY"):
        api = getattr(cv2, f"CAP_{name}", None)
        if api is not None and all(existing_api != api for _, existing_api in backends):
            backends.append((name.lower(), api))
    if not backends:
        backends.append(("any", getattr(cv2, "CAP_ANY", 0)))

    diagnostics = []

    def open_backend(index, backend_name, api):
        try:
            cap = cv2.VideoCapture(index, api)
            # Keep the freshest frame available; stale backend buffering adds
            # visible cursor lag when inference briefly falls behind 30 fps.
            buffer_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
            if buffer_prop is not None:
                cap.set(buffer_prop, 1)
        except Exception as exc:
            diagnostics.append(f"{index}/{backend_name}: open error: {exc}")
            return None
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            diagnostics.append(f"{index}/{backend_name}: did not open")
            return None

        frame = None
        last_any = None
        deadline = time.monotonic() + max(0.0, warmup_s)
        while time.monotonic() < deadline:
            try:
                ok, current = cap.read()
            except Exception as exc:
                diagnostics.append(f"{index}/{backend_name}: read error: {exc}")
                ok, current = False, None
            if ok and current is not None and getattr(current, "size", 0):
                last_any = current
                if _frame_is_usable(current):
                    frame = current
                    break
            else:
                time.sleep(0.03)
        if frame is None and last_any is None:
            cap.release()
            diagnostics.append(f"{index}/{backend_name}: no frames")
            return None

        dark = frame is None
        if dark:
            frame = last_any
        count = 0
        started = time.monotonic()
        while count < 6 and time.monotonic() - started < 0.9:
            try:
                ok, current = cap.read()
            except Exception:
                ok, current = False, None
            if ok and current is not None and getattr(current, "size", 0):
                count += 1
                if _frame_is_usable(current):
                    frame = current
                    dark = False
        fps = count / max(1e-3, time.monotonic() - started)
        height, width = frame.shape[:2]
        score = _score_camera_frame(frame, fps) if not dark else -50.0 + fps
        return {"cap": cap, "index": index, "backend": backend_name,
                "width": int(width), "height": int(height), "fps": fps,
                "score": score, "dark": dark}

    def open_index(index):
        dark = None
        for backend_name, api in backends:
            candidate = open_backend(index, backend_name, api)
            if candidate is None:
                continue
            if not candidate["dark"]:
                if dark is not None:
                    dark["cap"].release()
                return candidate
            if dark is None:
                dark = candidate
            else:
                candidate["cap"].release()
        return dark

    best = None
    misses = 0
    for index in range(max_index):
        candidate = open_index(index)
        if candidate is None:
            misses += 1
            if misses >= 8:
                break
            continue
        misses = 0
        if best is None or candidate["score"] > best["score"]:
            if best is not None:
                best["cap"].release()
            best = candidate
        else:
            candidate["cap"].release()

    if best is None:
        ffmpeg_cap, ffmpeg_details = _open_ffmpeg_camera(cv2)
        if ffmpeg_cap is not None:
            diagnostics.append(f"selected {ffmpeg_details}")
            return ffmpeg_cap, CAMERA_DEVICE_NAME, "ffmpeg-dshow", "; ".join(diagnostics[-12:])
        if ffmpeg_details:
            diagnostics.append(ffmpeg_details)
        return None, None, None, "; ".join(diagnostics[-12:])
    diagnostics.append(
        f"selected camera {best['index']} via {best['backend']} "
        f"({best['width']}x{best['height']}, {best['fps']:.1f} fps, score {best['score']:.1f})"
    )
    if best["dark"]:
        diagnostics.append("camera image is black/flat; check the privacy shutter or lens cover")
    return best["cap"], best["index"], best["backend"], "; ".join(diagnostics[-12:])

def _dashboard_status_text(status: dict) -> str:
    cursor = {
        "stopped": "Остановлено — камера сейчас не используется",
        "starting": "Запускается…",
        "active": "Управление включено · покажи лицо в кадр для взгляда",
        "waiting_for_hand": "Камера работает · покажи руку или лицо в кадр",
        "gaze_active": "Курсор следует за взглядом",
        "gaze_calibrating": "Взгляд отслеживается · калибровка необязательна",
        "dragging": "Перетаскивание · кнопка мыши удерживается",
        "camera_unavailable": "Камера не открылась — подробности ниже",
        "error": f"Ошибка контроллера: {status.get('message', 'неизвестная ошибка')}",
    }
    state = status.get("state", "stopped")
    text = cursor.get(state, f"Состояние: {state}")
    if status.get("camera_name"):
        text += f"  ·  {status['camera_name']}"
    elif status.get("camera") is not None:
        text += f"  ·  Камера {status['camera']}"
    if status.get("hand_detected"):
        text += "  ·  рука обнаружена"
    if status.get("gesture"):
        text += f"  ·  жест: {status['gesture']}"
    if status.get("pinch"):
        text += "  ·  щипок"
    if state in {"camera_unavailable", "error"} and status.get("message"):
        text += f"  ·  {status['message']}"
    if status.get("camera_diagnostics"):
        text += f"  ·  {status['camera_diagnostics']}"
    return text


def _run_dashboard() -> int:
    """Show the opt-in local camera preview and live hand-tracking diagnostics."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Управление ПК через камеру — диагностика")
    root.geometry("720x590")
    root.minsize(560, 480)
    root.configure(bg="#171a22")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#171a22")
    style.configure("TLabel", background="#171a22", foreground="#edf1f7", font=("Segoe UI", 10))
    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#8df0c0")
    style.configure("Hint.TLabel", foreground="#aab3c2")
    style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Диагностика отслеживания руки", style="Title.TLabel").pack(anchor="w")
    ttk.Label(frame, text="Всё обрабатывается локально на этом компьютере.", style="Hint.TLabel").pack(anchor="w", pady=(3, 12))
    image_label = tk.Label(frame, text="Камера выключена\\nНажми «Запустить», чтобы включить предпросмотр.",
                           bg="#0d1016", fg="#aab3c2", font=("Segoe UI", 12), width=60, height=17)
    image_label.pack(fill="both", expand=True)
    status_label = ttk.Label(frame, text="Остановлено — камера сейчас не используется", wraplength=660)
    status_label.pack(anchor="w", pady=(12, 6))
    detail_label = ttk.Label(frame, text="Курсор: —    Щипок: —    Жест: —", style="Hint.TLabel", wraplength=660)
    detail_label.pack(anchor="w", pady=(0, 12))
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")

    def launch_controller():
        if _current_pid():
            return
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STOP_FILE.unlink(missing_ok=True)
        log = LOG_FILE.open("ab", buffering=0)
        flags = _controller_creationflags()
        try:
            proc = subprocess.Popen(
                [_controller_python(), "-m", "src.plugin", "--controller"], cwd=str(PLUGIN_DIR),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, close_fds=True,
            )
        finally:
            log.close()
        PID_FILE.write_text(str(proc.pid), encoding="ascii")

    def stop_controller():
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STOP_FILE.write_text("stop", encoding="ascii")

    ttk.Button(buttons, text="▶  Запустить", command=launch_controller).pack(side="left")
    ttk.Button(buttons, text="■  Остановить", command=stop_controller).pack(side="left", padx=8)
    last_preview_mtime = None

    def refresh():
        nonlocal last_preview_mtime
        status = {}
        try:
            status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        status_label.configure(text=_dashboard_status_text(status))
        cursor = status.get("cursor")
        cursor_text = f"{cursor[0]}, {cursor[1]}" if cursor else "—"
        detail_label.configure(text=(
            f"Курсор: {cursor_text}    Щипок: {'да' if status.get('pinch') else 'нет'}    "
            f"Жест: {status.get('gesture') or '—'}"
        ))
        try:
            mtime = PREVIEW_FILE.stat().st_mtime_ns
            if mtime != last_preview_mtime:
                photo = tk.PhotoImage(file=str(PREVIEW_FILE))
                image_label.configure(image=photo, text="")
                image_label.image = photo
                last_preview_mtime = mtime
        except (OSError, tk.TclError):
            if not _current_pid():
                image_label.configure(image="", text="Камера выключена\\nНажми «Запустить», чтобы включить предпросмотр.")
                image_label.image = None
        root.after(250, refresh)

    root.after(100, refresh)
    root.mainloop()
    return 0


def _create_face_landmarker(vision, mp_python):
    """Create optional local face/iris tracker; no blendshapes or network output."""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=FACE_MODEL_FILE.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    if os.name != "nt":
        return vision.FaceLandmarker.create_from_options(options)
    # Reuse MediaPipe's Windows CRT binding workaround used by the hand task.
    bindings = importlib.import_module("mediapipe.tasks.python.core.mediapipe_c_bindings")
    import ctypes
    original_cdll = ctypes.CDLL
    crt = original_cdll("ucrtbase.dll")
    crt_free = crt.free
    crt_free.argtypes = [ctypes.c_void_p]
    crt_free.restype = None
    def load_with_crt_free(path, *args, **kwargs):
        library = original_cdll(path, *args, **kwargs)
        if str(path).lower().endswith("libmediapipe.dll"):
            library.free = crt_free
        return library
    bindings.ctypes.CDLL = load_with_crt_free
    try:
        return vision.FaceLandmarker.create_from_options(options)
    finally:
        bindings.ctypes.CDLL = original_cdll


def _next_timestamp_ms(now: float, previous_ms: int) -> int:
    """MediaPipe VIDEO tasks reject duplicate millisecond timestamps."""
    return max(int(now * 1000), previous_ms + 1)


def _run_controller() -> int:
    """Track one hand and run the Astra-adapted JARVIS gesture engine."""
    import cv2
    import mediapipe as mp
    import pyautogui
    from src.gesture_engine import FeatureTracker, JarvisGestureEngine, classify_fingers

    mp_python = importlib.import_module("mediapipe.tasks.python")
    vision = importlib.import_module("mediapipe.tasks.python.vision")
    from src.calibration import GazeController, face_gaze_features, hand_landmark_anchor
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True
    pyautogui.mouseUp()
    _write_status("starting")
    detector = _create_hand_landmarker(
        vision,
        mp_python,
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_FILE.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        ),
    )
    face_detector = None
    initial_settings = _load_gesture_settings()
    if initial_settings.get("control_mode") == "gaze" and FACE_MODEL_FILE.is_file():
        face_detector = _create_face_landmarker(vision, mp_python)
    current_calibration = _load_calibration()
    gaze_controller = GazeController(pyautogui, current_calibration.get("gaze", [])) if face_detector else None
    cap, camera_index, camera_backend, camera_diagnostics = _open_camera(cv2)
    if cap is None:
        detector.close()
        if face_detector is not None:
            face_detector.close()
        PID_FILE.unlink(missing_ok=True)
        details = camera_diagnostics or "OpenCV не смог получить изображение с камеры."
        message = (
            "Камеру не удалось открыть или получить кадр через MSMF/DirectShow/FFmpeg. "
            "Закрой приложения, занявшие камеру, проверь разрешение Windows и повтори запуск. "
            f"Диагностика: {details}"
        )
        _write_status("camera_unavailable", message=message, camera_backends=[
            name for name, api in (("msmf", getattr(cv2, "CAP_MSMF", None)),
                                   ("dshow", getattr(cv2, "CAP_DSHOW", None))) if api is not None
        ])
        raise RuntimeError(message)

    tracker = FeatureTracker()
    engine = JarvisGestureEngine(pyautogui, _load_gesture_settings())
    current_settings = engine.gesture_settings
    last_settings_reload = time.monotonic()
    last_status = 0.0
    last_preview = 0.0
    preview_interval = 1.0 / 30.0
    frame_id = 0
    hand_timestamp_ms = 0
    face_timestamp_ms = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    aspect = width / max(1, height)
    current_calibration = _load_calibration()
    last_calibration_reload = time.monotonic()
    _write_status("waiting_for_hand", camera=camera_index, camera_backend=camera_backend,
                  camera_name="DV20 USB CAMERA", hand_detected=False,
                  gesture="waiting_for_hand", camera_diagnostics=camera_diagnostics)
    try:
        while not STOP_FILE.exists():
            settings_now = time.monotonic()
            if settings_now - last_settings_reload >= 0.25:
                current_settings = _load_gesture_settings()
                if current_settings != engine.gesture_settings:
                    engine.gesture_settings = current_settings
                    engine.custom_fired = None
                if current_settings.get("control_mode") == "gaze" and face_detector is None and FACE_MODEL_FILE.is_file():
                    face_detector = _create_face_landmarker(vision, mp_python)
                    gaze_controller = GazeController(pyautogui, current_calibration.get("gaze", []))
                elif current_settings.get("control_mode") != "gaze" and face_detector is not None:
                    face_detector.close()
                    face_detector = None
                    gaze_controller = None
                if gaze_controller is not None:
                    gaze_controller.calibration = current_calibration.get("gaze", [])
                last_settings_reload = settings_now
            if settings_now - last_calibration_reload >= 1.0:
                current_calibration = _load_calibration()
                last_calibration_reload = settings_now
                if gaze_controller is not None:
                    gaze_controller.calibration = current_calibration.get("gaze", [])
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                engine.no_hand(now)
                tracker.reset()
                if now - last_status > 0.6:
                    _write_status("waiting_for_hand", camera=camera_index, camera_backend=camera_backend,
                                  hand_detected=False, gaze_detected=False,
                                  gesture="waiting_for_hand", camera_diagnostics=camera_diagnostics)
                    last_status = now
                time.sleep(0.04)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_timestamp_ms = _next_timestamp_ms(now, hand_timestamp_ms)
            result = detector.detect_for_video(image, hand_timestamp_ms)
            face_features = None
            mapped, blink_event = None, None
            if face_detector is not None:
                try:
                    face_timestamp_ms = _next_timestamp_ms(now, face_timestamp_ms)
                    face_result = face_detector.detect_for_video(image, face_timestamp_ms)
                    face_features = face_gaze_features(face_result.face_landmarks[0]) if face_result.face_landmarks else None
                except Exception:
                    face_features = None
                if face_features and current_settings.get("control_mode") == "gaze":
                    mapped, blink_event = gaze_controller.update(
                        face_features["gaze"], face_features["eye_open"], now,
                        clicks_enabled=bool(current_settings.get("gaze_clicks_enabled", True)))
                    if mapped is not None:
                        screen_w, screen_h = pyautogui.size()
                        cursor = [max(2,min(screen_w-3,round(mapped[0]*(screen_w-1)))),
                                  max(2,min(screen_h-3,round(mapped[1]*(screen_h-1))))]
                    else:
                        cursor = None
                elif current_settings.get("control_mode") == "gaze":
                    blink_event = gaze_controller.blinks.update(None, now)
                    cursor = None
            frame_id += 1
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            hand_anchor = hand_landmark_anchor(hand) if hand else None
            control_mode = current_settings.get("control_mode", "hand")
            if control_mode == "gaze" and gaze_controller is not None:
                state_name = "gaze_active" if face_features and mapped is not None else "gaze_calibrating"
                cursor = None
                try:
                    pos = pyautogui.position()
                    cursor = [int(pos.x), int(pos.y)]
                except Exception:
                    pass
                if mapped is not None:
                    screen_w, screen_h = pyautogui.size()
                    cursor = [max(2,min(screen_w-3,round(mapped[0]*(screen_w-1)))),
                              max(2,min(screen_h-3,round(mapped[1]*(screen_h-1))))]
                if now - last_status > 0.18:
                    _write_status(state_name, sample_id=frame_id, control_mode=control_mode,
                                  camera=camera_index, camera_backend=camera_backend,
                                  camera_name="DV20 USB CAMERA", hand_detected=bool(hand),
                                  gaze_detected=bool(face_features), gaze=face_features["gaze"] if face_features else None,
                                  gaze_cursor=[round(v, 3) for v in mapped] if mapped else None,
                                  eye_open=round(face_features["eye_open"], 3) if face_features else None,
                                  blink=blink_event, cursor=cursor,
                                  gesture=blink_event or ("Лицо обнаружено" if face_features else "Ищу лицо"),
                                  camera_diagnostics=camera_diagnostics)
                    last_status = now
                if now - last_preview >= preview_interval:
                    _write_preview(cv2, frame, None, True, False, camera_index,
                                   "Взгляд отслеживается" if face_features else "Ищу лицо")
                    last_preview = now
                time.sleep(0.001)
                continue
            if hand is None:
                engine.no_hand(now)
                tracker.reset()
                label = engine.last_label
                if now - last_status > 0.18:
                    _write_status("waiting_for_hand", sample_id=frame_id, control_mode=control_mode,
                                  camera=camera_index, camera_backend=camera_backend,
                                  camera_name="DV20 USB CAMERA", hand_detected=False,
                                  gaze_detected=bool(face_features), gesture="waiting_for_hand",
                                  camera_diagnostics=camera_diagnostics)
                    last_status = now
                if now - last_preview >= preview_interval:
                    _write_preview(cv2, frame, None, True, False, camera_index, label)
                    last_preview = now
                time.sleep(0.002)
                continue

            world_set = getattr(result, "hand_world_landmarks", None)
            world = world_set[0] if world_set else None
            features = tracker.update(hand, world, aspect)
            from src.calibration import map_calibrated
            hand_cal = _load_calibration().get("hand", [])
            mapped_hand = None
            if current_settings.get("control_mode") == "hand":
                engine.update(features, now)
            label = engine.last_label
            busy = engine.left.get("state") == "drag" or engine.right.get("state") != "up"
            is_pinching = engine.left.get("state") != "up" or engine.right.get("state") != "up"
            cursor = None
            try:
                pos = pyautogui.position()
                cursor = [int(pos.x), int(pos.y)]
            except Exception:
                pass
            state = ("gaze_active" if current_settings.get("control_mode") == "gaze" else "dragging" if busy else "active")
            if now - last_status > 0.18:
                _write_status(state, sample_id=frame_id, control_mode=current_settings.get("control_mode", "hand"),
                              camera=camera_index, camera_backend=camera_backend,
                              camera_name="DV20 USB CAMERA", hand_detected=True,
                              gesture=label, pose=classify_fingers(features["fingers"]),
                              fingers=features["fingers"], pinch=bool(is_pinching), cursor=cursor,
                              hand_anchor=list(features["anchor"]), spread=round(float(features["spread"]), 3),
                              gaze_detected=bool(face_features), gaze=face_features["gaze"] if face_features else None,
                              gaze_cursor=[round(v, 3) for v in mapped] if face_features and mapped else None,
                              camera_diagnostics=camera_diagnostics)
                last_status = now
            if now - last_preview >= preview_interval:
                _write_preview(cv2, frame, hand, True, bool(is_pinching), camera_index, label)
                last_preview = now
            time.sleep(0.001)
    finally:
        try:
            engine.shutdown()
        except Exception:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        cap.release()
        detector.close()
        if face_detector is not None:
            face_detector.close()
        STOP_FILE.unlink(missing_ok=True)
        PID_FILE.unlink(missing_ok=True)
        _write_status("stopped")
    return 0

if __name__ == "__main__":
    if "--controller" in sys.argv:
        try:
            raise SystemExit(_run_controller())
        except Exception as exc:
            _write_status("error", message=str(exc))
            print(f"Controller error: {exc}", flush=True)
            raise
    CameraHandControl().run()
