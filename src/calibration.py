"""Local calibration and gaze controls for webcam input."""
from __future__ import annotations

import math
import time


def _solve3(matrix, values):
    """Solve a tiny 3x3 system using pivoted Gaussian elimination."""
    a = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, values)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-9:
            raise ValueError("Недостаточно разных точек калибровки.")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [a[row][i] - factor * a[col][i] for i in range(4)]
    return [a[row][3] for row in range(3)]


def fit_affine(samples):
    """Fit raw (x,y) points to normalized target coordinates using least squares."""
    samples = [s for s in samples if isinstance(s, dict) and len(s.get("raw", [])) == 2 and len(s.get("target", [])) == 2]
    if len(samples) < 3:
        return None
    # Normal equations for [x, y, 1]; three-point calibration is enough, more
    # points reduce jitter from camera noise through a least-squares fit.
    rows = [[float(s["raw"][0]), float(s["raw"][1]), 1.0] for s in samples]
    gram = [[sum(row[i] * row[j] for row in rows) for j in range(3)] for i in range(3)]
    try:
        bx = [sum(row[i] * float(s["target"][0]) for row, s in zip(rows, samples)) for i in range(3)]
        by = [sum(row[i] * float(s["target"][1]) for row, s in zip(rows, samples)) for i in range(3)]
        return _solve3(gram, bx), _solve3(gram, by)
    except ValueError:
        return None


def map_calibrated(samples, raw):
    model = fit_affine(samples)
    if not model or not raw or len(raw) != 2:
        return None
    x, y = map(float, raw)
    out = [sum(c * v for c, v in zip(coeff, (x, y, 1.0))) for coeff in model]
    return [max(0.0, min(1.0, value)) for value in out]


def face_gaze_features(landmarks):
    """Return normalized iris position and eye-closed ratio from Face Landmarker points."""
    if not landmarks or len(landmarks) < 478:
        return None

    def xy(index):
        point = landmarks[index]
        return float(point.x), float(point.y)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    eyes = ((33, 133, 159, 145, (468, 469, 470, 471, 472)),
            (362, 263, 386, 374, (473, 474, 475, 476, 477)))
    positions, ears = [], []
    for outer, inner, upper, lower, iris_indices in eyes:
        a, b = xy(outer), xy(inner)
        lo, hi = sorted((a[0], b[0]))
        ux = max(1e-5, hi - lo)
        iris = tuple(sum(xy(i)[axis] for i in iris_indices) / len(iris_indices) for axis in (0, 1))
        top, bottom = xy(upper), xy(lower)
        positions.append(((iris[0] - lo) / ux, (iris[1] - min(top[1], bottom[1])) / max(1e-5, abs(bottom[1] - top[1]))))
    for indices in ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380)):
        p = [xy(i) for i in indices]
        width = max(1e-5, dist(p[0], p[3]))
        ears.append((dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * width))
    gaze = [sum(p[i] for p in positions) / 2.0 for i in range(2)]
    # Blinks close both eyes; using the smaller EAR prevents one noisy eye from
    # hiding a clear simultaneous closure and is robust to slight asymmetry.
    return {"gaze": gaze, "eye_open": min(ears), "eye_ears": ears}


class BlinkSeries:
    """Count deliberate 2/3-blink bursts; ignore ordinary single blinks."""
    def __init__(self, close_threshold=0.17, min_closed_s=0.025, group_s=0.72):
        self.close_threshold = close_threshold
        self.min_closed_s = min_closed_s
        self.group_s = group_s
        self.open_reference = 0.26
        self.closed_since = None
        self.was_closed = False
        self.count = 0
        self.last_blink = None

    def update(self, eye_open, now):
        if eye_open is None:
            closed = False
        else:
            value = float(eye_open)
            threshold = max(0.085, min(0.24, self.open_reference * 0.65))
            closed = value < threshold
            if not closed:
                self.open_reference = 0.96 * self.open_reference + 0.04 * value
        event = None
        if closed and not self.was_closed:
            self.closed_since = now
        elif not closed and self.was_closed:
            if self.closed_since is not None and now - self.closed_since >= self.min_closed_s:
                self.count += 1
                self.last_blink = now
        self.was_closed = closed
        self.closed_since = self.closed_since if closed else None
        if self.count and not closed and self.last_blink is not None and now - self.last_blink > self.group_s:
            if self.count == 2:
                event = "left_click"
            elif self.count == 3:
                event = "double_click"
            self.count = 0
            self.last_blink = None
        return event


class GazeController:
    """Maps calibrated iris position to the desktop and dispatches blink clicks."""
    def __init__(self, pyautogui, calibration=None):
        from src.gesture_engine import OneEuro2D
        self.pg = pyautogui
        self.calibration = list(calibration or [])
        self.filter = OneEuro2D()
        self.blinks = BlinkSeries()
        self.last_click = 0.0

    def update(self, gaze, eye_open, now=None, clicks_enabled=True):
        now = time.monotonic() if now is None else now
        # The iris position is already normalized inside each eye. Use it directly
        # for a usable default; an optional gaze calibration can refine mapping.
        normalized = map_calibrated(self.calibration, gaze) if len(self.calibration) >= 3 else [
            max(0.0, min(1.0, 1.0 - float(gaze[0]))),
            max(0.0, min(1.0, float(gaze[1]))),
        ] if gaze and len(gaze) == 2 else None
        if normalized is not None:
            sw, sh = self.pg.size()
            x, y = self.filter((normalized[0] * (sw - 1), normalized[1] * (sh - 1)), now, float(max(sw, sh)))
            self.pg.moveTo(max(2, min(sw - 3, int(round(x)))), max(2, min(sh - 3, int(round(y)))), _pause=False)
        event = self.blinks.update(eye_open, now) if clicks_enabled else None
        if event and now - self.last_click > 0.25:
            try:
                if event == "left_click":
                    self.pg.click(button="left")
                else:
                    self.pg.doubleClick(button="left")
                self.last_click = now
            except Exception:
                pass
        return normalized, event


def hand_landmark_anchor(landmarks):
    """Same mirrored palm anchor used by hand cursor control."""
    if not landmarks or len(landmarks) < 18:
        return None
    return [1.0 - (float(landmarks[5].x) * 0.7 + float(landmarks[9].x) * 0.3),
            float(landmarks[5].y) * 0.7 + float(landmarks[9].y) * 0.3]


def target_grid(rows=3, columns=3):
    return [[x / (columns - 1), y / (rows - 1)] for y in range(rows) for x in range(columns)]
