from __future__ import annotations

import json
import re
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from motion_cut.vision.capture import CameraBusyError, CameraStream
from motion_cut.vision.hand_tracker import HandTracker

# ── module-level constants ─────────────────────────────────────────────────────

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
_DEPTH_WEIGHT = 0.4
_SEQUENCE_POINTS_PREFIX = "sequence_points"
_DEFAULT_TOLERANCE = 0.23
_DEFAULT_MAX_GAP = 2.0
_DEFAULT_COLOR_TOLERANCE = 0.24
_STABLE_FRAMES = 3
_MID_STABLE_FRAMES = 1  # the middle is passed through, not held

# Shared by the wizard and the main window mapping panel.
SYSTEM_ACTIONS = (
    ("🔊  Volume Up", "volumeup"),
    ("🔉  Volume Down", "volumedown"),
    ("🔇  Mute", "mute"),
    ("⏯  Play / Pause", "playpause"),
    ("⏭  Next Track", "nexttrack"),
    ("⏮  Previous Track", "prevtrack"),
    ("☀  Brightness Up", "brightnessup"),
    ("🌙  Brightness Down", "brightnessdown"),
)

# ── shared utilities (minimal duplication to avoid circular imports) ───────────

def _build_pose_vector(hand_landmarks) -> np.ndarray:
    vec = np.array(
        [[float(p.x), float(p.y), float(p.z)] for p in hand_landmarks.landmark],
        dtype=np.float32,
    )
    wrist = vec[0]
    scale = float(np.linalg.norm(vec[9, :2] - wrist[:2]))
    if scale < 1e-4:
        scale = float(np.linalg.norm(vec[:, :2] - wrist[:2], axis=1).max(initial=1e-4))
    normalized = (vec - wrist) / max(scale, 1e-4)
    normalized[:, 2] *= _DEPTH_WEIGHT
    return normalized.astype(np.float32, copy=False)


def _build_color_signature(frame_bgr: np.ndarray, hand_landmarks) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    indices = (0, 4, 8, 12, 16, 20)
    signature = np.zeros((len(indices), 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        p = hand_landmarks.landmark[idx]
        cx = int(np.clip(p.x * w, 0, max(w - 1, 0)))
        cy = int(np.clip(p.y * h, 0, max(h - 1, 0)))
        r = 3
        patch = hsv[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
        if patch.size == 0:
            continue
        med = np.median(patch.reshape(-1, 3), axis=0)
        signature[i] = [med[0] / 180.0, med[1] / 255.0, med[2] / 255.0]
    return signature


def _pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b, axis=1).mean())


def _preview_points(pose_vector: np.ndarray, width: int = 220, height: int = 92) -> np.ndarray:
    points = pose_vector[:, :2].astype(np.float32)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) * 0.5
    span_x = float(max(maxs[0] - mins[0], 1e-5))
    span_y = float(max(maxs[1] - mins[1], 1e-5))
    scale = min(width * 0.76 / span_x, height * 0.76 / span_y)
    pixels = np.empty((21, 2), dtype=np.float32)
    pixels[:, 0] = (points[:, 0] - center[0]) * scale + width * 0.5
    pixels[:, 1] = (points[:, 1] - center[1]) * scale + height * 0.5
    norm = np.empty_like(pixels)
    norm[:, 0] = np.clip(pixels[:, 0] / width, 0.0, 1.0)
    norm[:, 1] = np.clip(pixels[:, 1] / height, 0.0, 1.0)
    return norm


def _snapshot_pixmap(
    pose_vector: np.ndarray,
    color_sig: np.ndarray | None,
    preview_pts: np.ndarray | None,
    width: int = 220,
    height: int = 92,
) -> QPixmap:
    px = QPixmap(width, height)
    px.fill(QColor("#101520"))
    if pose_vector.shape != (21, 3):
        return px

    if isinstance(preview_pts, np.ndarray) and preview_pts.shape == (21, 2):
        mapped = np.empty((21, 2), dtype=np.float32)
        mapped[:, 0] = np.clip(preview_pts[:, 0], 0.0, 1.0) * width
        mapped[:, 1] = np.clip(preview_pts[:, 1], 0.0, 1.0) * height
    else:
        points = pose_vector[:, :2].astype(np.float32)
        mins, maxs = points.min(axis=0), points.max(axis=0)
        center = (mins + maxs) * 0.5
        scale = min(width * 0.76 / float(max(maxs[0] - mins[0], 1e-5)),
                    height * 0.76 / float(max(maxs[1] - mins[1], 1e-5)))
        mapped = np.empty((21, 2), dtype=np.float32)
        mapped[:, 0] = (points[:, 0] - center[0]) * scale + width * 0.5
        mapped[:, 1] = (points[:, 1] - center[1]) * scale + height * 0.5

    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor("#4f6d95"), 2))
    for a, b in _HAND_CONNECTIONS:
        p.drawLine(int(mapped[a, 0]), int(mapped[a, 1]), int(mapped[b, 0]), int(mapped[b, 1]))

    tip_indices = {4, 8, 12, 16, 20}
    for idx in range(21):
        if idx == 0:
            color, r = QColor("#ffffff"), 4
        elif idx in tip_indices:
            color, r = QColor("#9de7b8"), 3
        else:
            color, r = QColor("#9ec2ff"), 2
        p.setPen(QPen(color, 1))
        p.setBrush(color)
        p.drawEllipse(int(mapped[idx, 0]) - r, int(mapped[idx, 1]) - r, r * 2, r * 2)
    p.end()
    return px


def _encode_sample(
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float,
    start_color: np.ndarray | None,
    end_color: np.ndarray | None,
    start_pts: np.ndarray | None,
    end_pts: np.ndarray | None,
    mid: np.ndarray | None = None,
    mid_color: np.ndarray | None = None,
    mid_pts: np.ndarray | None = None,
    hold_repeat: bool = False,
) -> str:
    payload: dict = {
        "start": np.round(start.reshape(-1), 5).tolist(),
        "end": np.round(end.reshape(-1), 5).tolist(),
        "max_gap": _DEFAULT_MAX_GAP,
        "start_tolerance": float(np.clip(tolerance, 0.08, 0.60)),
        "end_tolerance": float(np.clip(tolerance, 0.08, 0.60)),
        "mid_tolerance": float(np.clip(tolerance, 0.08, 0.60)),
        "start_color_tolerance": _DEFAULT_COLOR_TOLERANCE,
        "end_color_tolerance": _DEFAULT_COLOR_TOLERANCE,
        "hold_repeat": bool(hold_repeat),
    }
    if isinstance(mid, np.ndarray) and mid.shape == (21, 3):
        payload["mid"] = np.round(mid.reshape(-1), 5).tolist()
    if isinstance(mid_color, np.ndarray) and mid_color.shape == (6, 3):
        payload["mid_color"] = np.round(mid_color.reshape(-1), 5).tolist()
    if isinstance(mid_pts, np.ndarray) and mid_pts.shape == (21, 2):
        payload["mid_preview_points"] = np.round(mid_pts.reshape(-1), 5).tolist()
    if isinstance(start_color, np.ndarray) and start_color.shape == (6, 3):
        payload["start_color"] = np.round(start_color.reshape(-1), 5).tolist()
    if isinstance(end_color, np.ndarray) and end_color.shape == (6, 3):
        payload["end_color"] = np.round(end_color.reshape(-1), 5).tolist()
    if isinstance(start_pts, np.ndarray) and start_pts.shape == (21, 2):
        payload["start_preview_points"] = np.round(start_pts.reshape(-1), 5).tolist()
    if isinstance(end_pts, np.ndarray) and end_pts.shape == (21, 2):
        payload["end_preview_points"] = np.round(end_pts.reshape(-1), 5).tolist()
    return f"{_SEQUENCE_POINTS_PREFIX}:{json.dumps(payload, separators=(',', ':'))}"


def _frame_to_pixmap(frame_bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = QImage(resized.data, nw, nh, resized.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


def _hand_only_frame(landmarks: list, width: int, height: int, palette=()) -> np.ndarray:
    """Render just the hand skeleton, centred and zoomed, on a plain backdrop."""
    frame = np.full((height, width, 3), (28, 19, 14), dtype=np.uint8)

    # faint centre crosshair so the user can judge placement
    cv2.line(frame, (width // 2, 0), (width // 2, height), (48, 38, 32), 1)
    cv2.line(frame, (0, height // 2), (width, height // 2), (48, 38, 32), 1)

    if not landmarks:
        cv2.putText(
            frame, "no hand detected", (width // 2 - 78, height // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 122, 140), 1, cv2.LINE_AA,
        )
        return frame

    points = getattr(landmarks[0], "landmark", landmarks[0])
    xy = np.array(
        [
            [float(p[0]), float(p[1])] if isinstance(p, (tuple, list)) else [float(p.x), float(p.y)]
            for p in points
        ],
        dtype=np.float32,
    )
    if xy.shape[0] < 2:
        return frame

    mins, maxs = xy.min(axis=0), xy.max(axis=0)
    span = float(np.max(maxs - mins))
    if span <= 1e-5:
        return frame

    center = (mins + maxs) * 0.5
    scale = (min(width, height) * 0.68) / span
    mapped = np.empty_like(xy)
    mapped[:, 0] = (xy[:, 0] - center[0]) * scale + width * 0.5
    mapped[:, 1] = (xy[:, 1] - center[1]) * scale + height * 0.5

    for a, b in _HAND_CONNECTIONS:
        cv2.line(
            frame,
            (int(mapped[a, 0]), int(mapped[a, 1])),
            (int(mapped[b, 0]), int(mapped[b, 1])),
            (80, 220, 255), 2, cv2.LINE_AA,
        )

    for idx in range(mapped.shape[0]):
        color = palette[idx] if idx < len(palette) else (255, 255, 255)
        cx, cy = int(mapped[idx, 0]), int(mapped[idx, 1])
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

    return frame


# ── Helper widgets ─────────────────────────────────────────────────────────────

class _FeedLabel(QLabel):
    def __init__(self, w: int, h: int) -> None:
        super().__init__("Starting camera…")
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background:#0e131c; border:1px solid #2a3447; border-radius:10px; color:#6a7a9a;"
        )


class _SnapshotLabel(QLabel):
    def __init__(self, w: int, h: int, placeholder: str = "") -> None:
        super().__init__(placeholder)
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background:#101520; border:1px solid #2a3447; border-radius:8px;"
            " color:#8ea2c1; font-size:11px;"
        )


# ── Wizard ─────────────────────────────────────────────────────────────────────

class RecordMotionWizard(QDialog):
    """6-step wizard: start -> middle (optional) -> end -> review/test -> name -> shortcut."""

    # (gesture_name, shortcut_or_empty, sample_path, tolerance, hold_repeat)
    gesture_ready = Signal(str, str, str, float, bool)

    _PAGE_START = 0
    _PAGE_MID = 1
    _PAGE_END = 2
    _PAGE_REVIEW = 3
    _PAGE_NAME = 4
    _PAGE_SHORTCUT = 5

    _STEP_TITLES = [
        "Start Pose",
        "Middle Pose",
        "End Pose",
        "Review & Test",
        "Name",
        "Shortcut",
    ]

    def __init__(self, stream: CameraStream, tracker: HandTracker, parent=None) -> None:
        super().__init__(parent)
        self._stream = stream
        self._tracker = tracker
        self._stream_was_open = stream._cap is not None

        # Pose data
        self._start_vec: np.ndarray | None = None
        self._mid_vec: np.ndarray | None = None
        self._end_vec: np.ndarray | None = None
        self._start_color: np.ndarray | None = None
        self._mid_color: np.ndarray | None = None
        self._end_color: np.ndarray | None = None
        self._start_pts: np.ndarray | None = None
        self._mid_pts: np.ndarray | None = None
        self._end_pts: np.ndarray | None = None
        self._current_vec: np.ndarray | None = None
        self._current_color: np.ndarray | None = None
        self._tolerance = _DEFAULT_TOLERANCE
        self._hand_only = True

        # Live detection state (review page mirrors the runtime state machine)
        self._phase = 0
        self._streak = 0
        self._deadline = 0.0
        self._departed = False
        self._last_hit_at = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

        self.setWindowTitle("Record New Motion")
        self.setModal(True)
        self.resize(860, 620)
        self._apply_theme()
        self._build_ui()
        self._go_to_page(self._PAGE_START)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Progress bar row
        progress_row = QHBoxLayout()
        progress_row.setSpacing(4)
        self._step_dots: list[QLabel] = []
        for i, title in enumerate(self._STEP_TITLES):
            dot = QLabel(f" {i + 1}. {title} ")
            dot.setObjectName("stepDotInactive")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._step_dots.append(dot)
            progress_row.addWidget(dot, 1)
        root.addLayout(progress_row)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#2a3447;")
        root.addWidget(sep)

        # Page stack
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._stack.addWidget(self._page_start())
        self._stack.addWidget(self._page_mid())
        self._stack.addWidget(self._page_end())
        self._stack.addWidget(self._page_review())
        self._stack.addWidget(self._page_name())
        self._stack.addWidget(self._page_shortcut())

        # Nav row
        nav = QHBoxLayout()
        nav.setSpacing(10)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._back_btn = QPushButton("← Back")
        self._back_btn.setObjectName("secondaryButton")
        self._next_btn = QPushButton("Next →")
        self._next_btn.setObjectName("primaryButton")
        self._finish_btn = QPushButton("✔  Save Gesture")
        self._finish_btn.setObjectName("primaryButton")
        self._finish_btn.hide()
        self._cancel_btn.clicked.connect(self.reject)
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn.clicked.connect(self._go_next)
        self._finish_btn.clicked.connect(self._finish)

        self._view_btn = QPushButton("👋  Hand Only")
        self._view_btn.setObjectName("secondaryButton")
        self._view_btn.setCheckable(True)
        self._view_btn.setChecked(True)
        self._view_btn.setToolTip(
            "Switch between the isolated hand skeleton and the raw camera image."
        )
        self._view_btn.toggled.connect(self._on_view_toggled)

        nav.addWidget(self._cancel_btn)
        nav.addWidget(self._view_btn)
        nav.addStretch(1)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._finish_btn)
        root.addLayout(nav)

    def _on_view_toggled(self, checked: bool) -> None:
        self._hand_only = checked
        self._view_btn.setText("👋  Hand Only" if checked else "🎥  Camera")

    # ── Pages ──────────────────────────────────────────────────────────────────

    def _page_start(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        desc = QLabel(
            "Position your hand in the <b>starting pose</b> and hold it still, "
            "then click <b>Capture Start Pose</b>."
        )
        desc.setWordWrap(True)
        desc.setObjectName("descLabel")
        lay.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(16)

        self._feed_start = _FeedLabel(430, 286)
        row.addWidget(self._feed_start)

        right = QVBoxLayout()
        right.setSpacing(10)

        self._start_status = QLabel("● Show your hand in frame")
        self._start_status.setObjectName("statusHint")

        self._btn_cap_start = QPushButton("📸   Capture Start Pose")
        self._btn_cap_start.setObjectName("captureButton")
        self._btn_cap_start.setFixedHeight(42)
        self._btn_cap_start.setEnabled(False)
        self._btn_cap_start.clicked.connect(self._capture_start)

        self._lbl_start_snap = _SnapshotLabel(220, 100, "No snapshot yet")

        right.addWidget(QLabel("Captured pose preview:"))
        right.addWidget(self._lbl_start_snap)
        right.addSpacing(6)
        right.addWidget(self._btn_cap_start)
        right.addWidget(self._start_status)
        right.addStretch(1)

        row.addLayout(right)
        lay.addLayout(row, 1)
        return w

    def _page_mid(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        desc = QLabel(
            "Optional: capture a <b>middle pose</b> from partway through the motion.<br>"
            "This makes detection much more precise and prevents accidental triggers. "
            "Click <b>Skip</b> to use only start and end."
        )
        desc.setWordWrap(True)
        desc.setObjectName("descLabel")
        lay.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(16)
        self._feed_mid = _FeedLabel(430, 286)
        row.addWidget(self._feed_mid)

        right = QVBoxLayout()
        right.setSpacing(8)

        self._mid_status = QLabel("● Show your hand in frame")
        self._mid_status.setObjectName("statusHint")

        self._btn_cap_mid = QPushButton("📸   Capture Middle Pose")
        self._btn_cap_mid.setObjectName("captureButton")
        self._btn_cap_mid.setFixedHeight(42)
        self._btn_cap_mid.setEnabled(False)
        self._btn_cap_mid.clicked.connect(self._capture_mid)

        self._btn_skip_mid = QPushButton("Skip this step")
        self._btn_skip_mid.setObjectName("secondaryButton")
        self._btn_skip_mid.clicked.connect(self._skip_mid)

        snap_row = QHBoxLayout()
        sc = QVBoxLayout()
        sc.setSpacing(4)
        sc.addWidget(QLabel("Start pose:"))
        self._lbl_start_ref_mid = _SnapshotLabel(148, 82, "—")
        sc.addWidget(self._lbl_start_ref_mid)

        mc = QVBoxLayout()
        mc.setSpacing(4)
        mc.addWidget(QLabel("Middle pose:"))
        self._lbl_mid_snap = _SnapshotLabel(148, 82, "Not captured")
        mc.addWidget(self._lbl_mid_snap)

        snap_row.addLayout(sc)
        snap_row.addSpacing(8)
        snap_row.addLayout(mc)

        right.addLayout(snap_row)
        right.addSpacing(6)
        right.addWidget(self._btn_cap_mid)
        right.addWidget(self._btn_skip_mid)
        right.addWidget(self._mid_status)
        right.addStretch(1)

        row.addLayout(right)
        lay.addLayout(row, 1)
        return w

    def _page_end(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        desc = QLabel(
            "Now move your hand to the <b>end pose</b>, hold it still, "
            "then click <b>Capture End Pose</b>."
        )
        desc.setWordWrap(True)
        desc.setObjectName("descLabel")
        lay.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(16)

        self._feed_end = _FeedLabel(430, 286)
        row.addWidget(self._feed_end)

        right = QVBoxLayout()
        right.setSpacing(8)

        self._end_status = QLabel("● Show your hand in frame")
        self._end_status.setObjectName("statusHint")

        self._btn_cap_end = QPushButton("📸   Capture End Pose")
        self._btn_cap_end.setObjectName("captureButton")
        self._btn_cap_end.setFixedHeight(42)
        self._btn_cap_end.setEnabled(False)
        self._btn_cap_end.clicked.connect(self._capture_end)

        snap_row = QHBoxLayout()
        sc = QVBoxLayout()
        sc.setSpacing(4)
        sc.addWidget(QLabel("Start pose:"))
        self._lbl_start_ref = _SnapshotLabel(148, 82, "—")
        sc.addWidget(self._lbl_start_ref)

        ec = QVBoxLayout()
        ec.setSpacing(4)
        ec.addWidget(QLabel("End pose:"))
        self._lbl_end_snap = _SnapshotLabel(148, 82, "No snapshot yet")
        ec.addWidget(self._lbl_end_snap)

        snap_row.addLayout(sc)
        snap_row.addSpacing(8)
        snap_row.addLayout(ec)

        right.addLayout(snap_row)
        right.addSpacing(6)
        right.addWidget(self._btn_cap_end)
        right.addWidget(self._end_status)
        right.addStretch(1)
        row.addLayout(right)
        lay.addLayout(row, 1)
        return w

    def _page_review(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        desc = QLabel(
            "Adjust the sensitivity slider, then perform the gesture to verify "
            "detection before continuing."
        )
        desc.setWordWrap(True)
        desc.setObjectName("descLabel")
        lay.addWidget(desc)

        main_row = QHBoxLayout()
        main_row.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(8)
        self._feed_review = _FeedLabel(390, 258)
        self._detection_label = QLabel("Perform the gesture to test…")
        self._detection_label.setObjectName("detectionIdle")
        self._detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detection_label.setFixedHeight(36)
        left.addWidget(self._feed_review)
        left.addWidget(self._detection_label)
        main_row.addLayout(left)

        right = QVBoxLayout()
        right.setSpacing(12)

        snaps = QHBoxLayout()
        snaps.setSpacing(6)
        for title, attr in (("Start", "_review_start"), ("Middle", "_review_mid"), ("End", "_review_end")):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(QLabel(title))
            label = _SnapshotLabel(122, 82)
            setattr(self, attr, label)
            col.addWidget(label)
            snaps.addLayout(col)
        right.addLayout(snaps)

        right.addWidget(QLabel("Sensitivity (lower = stricter)"))
        self._tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self._tolerance_slider.setRange(8, 60)
        self._tolerance_slider.setValue(int(_DEFAULT_TOLERANCE * 100))
        self._tolerance_lbl = QLabel()
        self._tolerance_slider.valueChanged.connect(self._on_tolerance_changed)
        self._on_tolerance_changed(self._tolerance_slider.value())
        right.addWidget(self._tolerance_slider)
        right.addWidget(self._tolerance_lbl)
        right.addStretch(1)
        main_row.addLayout(right)
        lay.addLayout(main_row, 1)
        return w

    def _page_name(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(14)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        desc = QLabel(
            "Give your gesture a short, unique name.<br>"
            "Use only letters, numbers, and underscores."
        )
        desc.setWordWrap(True)
        desc.setObjectName("descLabel")
        lay.addWidget(desc)
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g.  swipe_left")
        self._name_input.setFixedHeight(44)
        self._name_input.textChanged.connect(self._validate_name)
        lay.addWidget(self._name_input)
        self._name_hint = QLabel("")
        self._name_hint.setObjectName("errorLabel")
        lay.addWidget(self._name_hint)
        lay.addStretch(1)
        return w

    def _page_shortcut(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(14)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        desc = QLabel(
            "Press the keyboard shortcut to assign to this gesture, "
            "or pick a system action below.<br>"
            "Leave both blank to assign one later from the main window."
        )
        desc.setWordWrap(True)
        desc.setObjectName("descLabel")
        lay.addWidget(desc)
        self._shortcut_edit = QKeySequenceEdit()
        self._shortcut_edit.setMaximumSequenceLength(4)
        self._shortcut_edit.setFixedHeight(44)
        self._shortcut_edit.keySequenceChanged.connect(self._on_wizard_shortcut_changed)
        lay.addWidget(self._shortcut_edit)

        lay.addWidget(QLabel("Or system action"))
        self._system_action_combo = QComboBox()
        self._system_action_combo.setFixedHeight(38)
        self._system_action_combo.addItem("— none —", "")
        for label, payload in SYSTEM_ACTIONS:
            self._system_action_combo.addItem(label, payload)
        self._system_action_combo.currentIndexChanged.connect(
            self._on_wizard_system_action_changed
        )
        lay.addWidget(self._system_action_combo)

        hint = QLabel("Leave empty to skip shortcut assignment now.")
        hint.setObjectName("statusHint")
        lay.addWidget(hint)

        self._hold_repeat_checkbox = QCheckBox("Repeat while held  (for volume, scrolling, etc.)")
        lay.addWidget(self._hold_repeat_checkbox)

        lay.addStretch(1)
        return w

    def _on_wizard_shortcut_changed(self, sequence) -> None:
        if not sequence.isEmpty():
            self._system_action_combo.setCurrentIndex(0)

    def _on_wizard_system_action_changed(self, index: int) -> None:
        if index > 0:
            self._shortcut_edit.clear()

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _go_to_page(self, index: int) -> None:
        self._stack.setCurrentIndex(index)

        for i, dot in enumerate(self._step_dots):
            if i == index:
                dot.setObjectName("stepDotActive")
            elif i < index:
                dot.setObjectName("stepDotDone")
            else:
                dot.setObjectName("stepDotInactive")
            dot.style().unpolish(dot)
            dot.style().polish(dot)

        self._back_btn.setEnabled(index > 0)

        if index == self._PAGE_SHORTCUT:
            self._next_btn.hide()
            self._finish_btn.show()
        else:
            self._finish_btn.hide()
            self._next_btn.show()
            self._next_btn.setEnabled(self._can_advance(index))

        camera_pages = (self._PAGE_START, self._PAGE_MID, self._PAGE_END, self._PAGE_REVIEW)
        self._view_btn.setVisible(index in camera_pages)
        if index in camera_pages:
            self._ensure_camera_open()
        else:
            self._timer.stop()

        if index == self._PAGE_MID and isinstance(self._start_vec, np.ndarray):
            self._lbl_start_ref_mid.setPixmap(
                _snapshot_pixmap(self._start_vec, self._start_color, self._start_pts, 148, 82)
            )

        if index == self._PAGE_END and isinstance(self._start_vec, np.ndarray):
            self._lbl_start_ref.setPixmap(
                _snapshot_pixmap(self._start_vec, self._start_color, self._start_pts, 148, 82)
            )

        if index == self._PAGE_REVIEW:
            self._reset_detection()
            if isinstance(self._start_vec, np.ndarray):
                self._review_start.setPixmap(
                    _snapshot_pixmap(self._start_vec, self._start_color, self._start_pts, 122, 82)
                )
            if isinstance(self._mid_vec, np.ndarray):
                self._review_mid.setPixmap(
                    _snapshot_pixmap(self._mid_vec, self._mid_color, self._mid_pts, 122, 82)
                )
            else:
                self._review_mid.setText("not used")
            if isinstance(self._end_vec, np.ndarray):
                self._review_end.setPixmap(
                    _snapshot_pixmap(self._end_vec, self._end_color, self._end_pts, 122, 82)
                )

        if index == self._PAGE_NAME:
            self._name_input.setFocus()

    def _can_advance(self, page: int) -> bool:
        if page == self._PAGE_START:
            return self._start_vec is not None
        if page == self._PAGE_MID:
            return True  # the middle pose is optional
        if page == self._PAGE_END:
            return self._end_vec is not None
        if page == self._PAGE_NAME:
            return bool(self._clean_name())
        return True

    def _go_next(self) -> None:
        cur = self._stack.currentIndex()
        if not self._can_advance(cur):
            return
        self._go_to_page(cur + 1)

    def _go_back(self) -> None:
        cur = self._stack.currentIndex()
        if cur > 0:
            self._go_to_page(cur - 1)

    def _finish(self) -> None:
        name = self._clean_name()
        if not name:
            self._go_to_page(self._PAGE_NAME)
            return
        if self._start_vec is None or self._end_vec is None:
            QMessageBox.warning(self, "Incomplete", "Both poses must be captured.")
            return
        sample_path = _encode_sample(
            self._start_vec, self._end_vec, self._tolerance,
            self._start_color, self._end_color,
            self._start_pts, self._end_pts,
            mid=self._mid_vec,
            mid_color=self._mid_color,
            mid_pts=self._mid_pts,
            hold_repeat=self._hold_repeat_checkbox.isChecked(),
        )
        # A chosen system action takes priority over the captured key sequence.
        shortcut = str(self._system_action_combo.currentData() or "")
        if not shortcut:
            shortcut = self._shortcut_edit.keySequence().toString(
                QKeySequence.SequenceFormat.PortableText
            ).strip()
        self.gesture_ready.emit(name, shortcut, sample_path, self._tolerance,
                                self._hold_repeat_checkbox.isChecked())
        self.accept()

    # ── Camera ─────────────────────────────────────────────────────────────────

    def _ensure_camera_open(self) -> None:
        try:
            self._stream.open()
        except CameraBusyError as e:
            QMessageBox.warning(self, "Camera Busy", str(e))
            return
        except RuntimeError as e:
            QMessageBox.critical(self, "Camera Error", str(e))
            return
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self) -> None:
        frame = self._stream.read()
        if frame is None:
            return

        now = time.monotonic()
        h, w = frame.shape[:2]
        tracking_frame = frame
        if w > 480:
            nh = max(1, int(h * 480 / w))
            tracking_frame = cv2.resize(frame, (480, nh), interpolation=cv2.INTER_LINEAR)

        result = self._tracker.process(tracking_frame)
        has_hand = bool(result.landmarks)

        if has_hand:
            hand = result.landmarks[0]
            self._current_vec = _build_pose_vector(hand)
            self._current_color = _build_color_signature(tracking_frame, hand)
        else:
            self._current_vec = None
            self._current_color = None

        if self._hand_only:
            display = _hand_only_frame(
                result.landmarks, frame.shape[1], frame.shape[0],
                getattr(self._tracker, "_LANDMARK_COLORS", ()),
            )
        else:
            display = frame.copy()
            self._tracker.draw_landmarks(display, result.landmarks)

        page = self._stack.currentIndex()

        if page == self._PAGE_START:
            self._feed_start.setPixmap(_frame_to_pixmap(display, 430, 286))
            ready = has_hand
            self._btn_cap_start.setEnabled(ready)
            self._start_status.setText(
                "✔ Hand detected — ready to capture" if ready else "● Show your hand in frame"
            )

        elif page == self._PAGE_MID:
            self._feed_mid.setPixmap(_frame_to_pixmap(display, 430, 286))
            self._btn_cap_mid.setEnabled(has_hand)
            if self._mid_vec is None:
                self._mid_status.setText(
                    "✔ Hand detected — ready to capture" if has_hand
                    else "● Show your hand in frame"
                )

        elif page == self._PAGE_END:
            self._feed_end.setPixmap(_frame_to_pixmap(display, 430, 286))
            ready = has_hand
            self._btn_cap_end.setEnabled(ready)
            self._end_status.setText(
                "✔ Hand detected — ready to capture" if ready else "● Show your hand in frame"
            )

        elif page == self._PAGE_REVIEW:
            self._feed_review.setPixmap(_frame_to_pixmap(display, 390, 258))
            self._run_live_detection(now)

    def _run_live_detection(self, now: float) -> None:
        if self._start_vec is None or self._end_vec is None or self._current_vec is None:
            return

        has_mid = isinstance(self._mid_vec, np.ndarray)

        # Mirror of MainWindow._advance_points_sequence so the test matches runtime.
        candidates = [
            ("start", _pose_distance(self._current_vec, self._start_vec)),
            ("end", _pose_distance(self._current_vec, self._end_vec)),
        ]
        if has_mid:
            candidates.append(("mid", _pose_distance(self._current_vec, self._mid_vec)))

        zone = ""
        best = float("inf")
        ambiguous = False
        for name, distance in candidates:
            if distance > self._tolerance:
                continue
            if distance < best - 1e-6:
                best, zone, ambiguous = distance, name, False
            elif abs(distance - best) <= 1e-6:
                ambiguous = True
        if ambiguous:
            zone = ""

        if self._phase > 0 and now > self._deadline:
            self._phase = 0
            self._streak = 0

        if self._phase == 0:
            self._streak = self._streak + 1 if zone == "start" else 0
            if self._streak >= _STABLE_FRAMES:
                self._phase = 1 if has_mid else 2
                self._streak = 0
                self._departed = False
                self._deadline = now + _DEFAULT_MAX_GAP
        else:
            if zone != "start":
                self._departed = True

            if self._phase == 1:
                self._streak = self._streak + 1 if zone == "mid" else 0
                if self._streak >= _MID_STABLE_FRAMES:
                    self._phase = 2
                    self._streak = 0
            else:
                counts = zone == "end" and self._departed
                self._streak = self._streak + 1 if counts else 0
                if self._streak >= _STABLE_FRAMES:
                    self._phase = 0
                    self._streak = 0
                    self._departed = False
                    self._last_hit_at = now
                    self._set_detection("✔  Gesture detected!", "detectionHit")
                    return

        if (now - self._last_hit_at) > 1.5:
            if self._phase == 1:
                self._set_detection("… start matched, waiting for middle…", "detectionArmed")
            elif self._phase == 2:
                self._set_detection("… waiting for end pose…", "detectionArmed")
            else:
                self._set_detection("Perform the gesture to test…", "detectionIdle")

    def _set_detection(self, text: str, obj_name: str) -> None:
        if self._detection_label.text() == text:
            return
        self._detection_label.setText(text)
        self._detection_label.setObjectName(obj_name)
        self._detection_label.style().unpolish(self._detection_label)
        self._detection_label.style().polish(self._detection_label)

    def _reset_detection(self) -> None:
        self._phase = 0
        self._streak = 0
        self._deadline = 0.0
        self._departed = False
        self._last_hit_at = 0.0

    # ── Capture ────────────────────────────────────────────────────────────────

    def _capture_start(self) -> None:
        if self._current_vec is None:
            return
        self._start_vec = self._current_vec.copy()
        self._start_color = self._current_color.copy() if self._current_color is not None else None
        self._start_pts = _preview_points(self._start_vec)
        self._lbl_start_snap.setPixmap(
            _snapshot_pixmap(self._start_vec, self._start_color, self._start_pts)
        )
        self._start_status.setText("✔ Start pose captured!")
        self._next_btn.setEnabled(True)

    def _capture_mid(self) -> None:
        if self._current_vec is None:
            return
        self._mid_vec = self._current_vec.copy()
        self._mid_color = self._current_color.copy() if self._current_color is not None else None
        self._mid_pts = _preview_points(self._mid_vec)
        self._lbl_mid_snap.setPixmap(
            _snapshot_pixmap(self._mid_vec, self._mid_color, self._mid_pts, 148, 82)
        )

        gap = (
            _pose_distance(self._start_vec, self._mid_vec)
            if self._start_vec is not None
            else 1.0
        )
        if gap < 0.15:
            self._mid_status.setText(
                f"⚠ Too close to the start pose (gap {gap:.2f}) — move further along"
            )
        else:
            self._mid_status.setText(f"✔ Middle pose captured  (gap {gap:.2f})")

    def _skip_mid(self) -> None:
        self._mid_vec = None
        self._mid_color = None
        self._mid_pts = None
        self._lbl_mid_snap.clear()
        self._lbl_mid_snap.setText("Not captured")
        self._mid_status.setText("Middle pose skipped")
        self._go_to_page(self._PAGE_END)

    def _capture_end(self) -> None:
        if self._current_vec is None:
            return
        self._end_vec = self._current_vec.copy()
        self._end_color = self._current_color.copy() if self._current_color is not None else None
        self._end_pts = _preview_points(self._end_vec)
        self._lbl_end_snap.setPixmap(
            _snapshot_pixmap(self._end_vec, self._end_color, self._end_pts, 148, 82)
        )

        separation = (
            _pose_distance(self._start_vec, self._end_vec)
            if self._start_vec is not None
            else 1.0
        )
        if separation < 0.15:
            self._end_status.setText(
                f"⚠ Poses very similar (gap {separation:.2f}) — make them more distinct"
            )
        else:
            self._end_status.setText(f"✔ End pose captured  (gap {separation:.2f})")
        self._next_btn.setEnabled(True)

    # ── Validation ─────────────────────────────────────────────────────────────

    def _clean_name(self) -> str:
        raw = self._name_input.text().strip().lower() if hasattr(self, "_name_input") else ""
        return re.sub(r"[^a-z0-9]+", "_", raw).strip("_") if raw else ""

    def _validate_name(self) -> None:
        name = self._clean_name()
        if name:
            self._name_hint.setText(f"Will be saved as:  {name}")
            self._name_hint.setObjectName("nameOk")
        else:
            self._name_hint.setText("Name cannot be empty.")
            self._name_hint.setObjectName("errorLabel")
        self._name_hint.style().unpolish(self._name_hint)
        self._name_hint.style().polish(self._name_hint)
        if self._stack.currentIndex() == self._PAGE_NAME:
            self._next_btn.setEnabled(bool(name))

    def _on_tolerance_changed(self, val: int) -> None:
        self._tolerance = float(np.clip(val / 100.0, 0.08, 0.60))
        profile = "strict" if self._tolerance <= 0.16 else ("balanced" if self._tolerance <= 0.30 else "loose")
        self._tolerance_lbl.setText(f"Tolerance: {self._tolerance:.2f}  ({profile})")
        self._reset_detection()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_camera_open()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if not self._stream_was_open:
            self._stream.close()
        super().closeEvent(event)

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog, QWidget {
                background: #141821;
                color: #e8edf7;
                font-size: 13px;
            }
            QLabel#stepDotActive {
                background: #2374e1;
                border: 1px solid #2e82f0;
                border-radius: 6px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 2px;
            }
            QLabel#stepDotDone {
                background: #183520;
                border: 1px solid #2a6040;
                border-radius: 6px;
                color: #60c090;
                font-size: 11px;
                padding: 4px 2px;
            }
            QLabel#stepDotInactive {
                background: #1a2130;
                border: 1px solid #2a3447;
                border-radius: 6px;
                color: #6a7a9a;
                font-size: 11px;
                padding: 4px 2px;
            }
            QLabel#descLabel {
                color: #c8d4ea;
                font-size: 13px;
                padding: 2px 0;
            }
            QLabel#statusHint {
                color: #7a9fc0;
                font-size: 12px;
            }
            QLabel#errorLabel {
                color: #e05060;
                font-size: 12px;
            }
            QLabel#nameOk {
                color: #60c090;
                font-size: 12px;
            }
            QLabel#detectionIdle {
                background: #1a2435;
                border: 1px solid #2a3a52;
                border-radius: 8px;
                color: #7a9fc0;
                padding: 4px 10px;
                font-size: 12px;
            }
            QLabel#detectionArmed {
                background: #1e2a18;
                border: 1px solid #3a5a28;
                border-radius: 8px;
                color: #b0e080;
                padding: 4px 10px;
                font-size: 12px;
            }
            QLabel#detectionHit {
                background: #183020;
                border: 1px solid #30b060;
                border-radius: 8px;
                color: #60e090;
                font-weight: 700;
                padding: 4px 10px;
                font-size: 13px;
            }
            QPushButton {
                background: #2b3548;
                border: 1px solid #3b4a63;
                border-radius: 8px;
                padding: 7px 14px;
            }
            QPushButton:hover { background: #34425a; }
            QPushButton:disabled {
                color: #8e9bb1; background: #222a39; border: 1px solid #2b3447;
            }
            QPushButton#primaryButton {
                background: #2374e1; border: 1px solid #2e82f0;
                color: #f4f9ff; font-weight: 600;
            }
            QPushButton#primaryButton:hover { background: #2a84ff; }
            QPushButton#primaryButton:disabled {
                background: #222a39; border: 1px solid #2b3447; color: #8e9bb1;
            }
            QPushButton#captureButton {
                background: #1e3a58; border: 1px solid #2a5a88;
                color: #9ec8ff; font-weight: 600; font-size: 13px;
            }
            QPushButton#captureButton:hover { background: #254870; }
            QPushButton#captureButton:disabled {
                background: #1a2130; border: 1px solid #2a3447; color: #4a5a6a;
            }
            QPushButton#secondaryButton { background: #2c3e55; }
            QPushButton#cancelButton { background: #2b3548; color: #c8bbb0; }
            QLineEdit, QKeySequenceEdit, QComboBox {
                background: #121a27; border: 1px solid #30405a;
                border-radius: 8px; padding: 6px 10px;
            }
            QSlider::groove:horizontal {
                background: #223047; height: 8px; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6fb3ff; width: 14px; margin: -4px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #2c7be5; border-radius: 4px; }
            """
        )
