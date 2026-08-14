from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QCursor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QKeySequenceEdit,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QGraphicsOpacityEffect,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from motion_cut.config import CameraConfig, RuntimeConfig
from motion_cut.actions.dispatcher import ActionDispatcher
from motion_cut.gestures import motion_matcher
from motion_cut.gestures.rule_engine import GesturePrediction
from motion_cut.storage.repository import Repository
from motion_cut.ui.wizard import SYSTEM_ACTIONS
from motion_cut.vision.capture import CameraBusyError, CameraStream
from motion_cut.vision.hand_tracker import HandTracker


class MainWindow(QMainWindow):
    _SEQUENCE_SAMPLE_PREFIX = "sequence"
    _SEQUENCE_POINTS_PREFIX = "sequence_points"
    _SEQUENCE_DEFAULT_THRESHOLD = 0.68
    _SEQUENCE_DEFAULT_TOLERANCE = 0.23
    _SEQUENCE_SWEET_SPOT_START_COLOR_TOLERANCE = 0.24
    _SEQUENCE_SWEET_SPOT_END_COLOR_TOLERANCE = 0.24
    _SEQUENCE_DEFAULT_MAX_GAP_SECONDS = 2.0
    _SEQUENCE_STABLE_FRAMES = 2
    _SEQUENCE_MID_STABLE_FRAMES = 1  # the middle is passed through, not held
    _SEQUENCE_HOLD_REPEAT_INTERVAL = 0.35  # seconds between repeats while end zone held
    _SYSTEM_ACTIONS = SYSTEM_ACTIONS
    _HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motion Cut")
        self.setWindowIcon(self._make_app_icon())
        self.resize(1200, 760)

        runtime_cfg = RuntimeConfig()
        camera_cfg = CameraConfig()
        self._min_detection_confidence = runtime_cfg.min_detection_confidence
        self._min_tracking_confidence = runtime_cfg.min_tracking_confidence

        self._stream = CameraStream(camera_cfg)
        self._tracker = HandTracker(
            min_detection_confidence=runtime_cfg.min_detection_confidence,
            min_tracking_confidence=runtime_cfg.min_tracking_confidence,
        )
        self._dispatcher = ActionDispatcher()
        self._repository = Repository()
        self._repository.ensure_schema()
        self._shortcut_mappings: dict[str, str] = {}
        self._trigger_cooldown_seconds = 0.6  # rapid repeat: 0.6 s between distinct gesture fires
        self._last_trigger_at: dict[str, float] = {}
        self._gesture_switch_guard_seconds = 0.30
        self._last_triggered_gesture = ""
        self._last_triggered_at = 0.0
        self._gesture_stability_frames = 4
        self._gesture_candidate_name = ""
        self._gesture_candidate_count = 0
        self._last_stable_gesture = "unknown"
        self._debug_overlay_enabled = False
        self._preview_fps = 0.0
        self._last_tick_at = 0.0
        self._tracking_resize_width = 480
        self._tracking_resize_height = 270
        self._had_landmarks_last_tick = False
        self._no_landmark_streak = 0
        self._tracker_recover_after_frames = 30
        self._last_tracker_recover_at = 0.0
        self._tracker_recover_cooldown_seconds = 2.0
        self._frame_miss_streak = 0
        self._frame_recover_after_frames = 40
        self._last_camera_recover_at = 0.0
        self._camera_recover_cooldown_seconds = 2.0
        self._guide_overlay_enabled = False
        self._camera_feed_enabled = True
        self._window_is_minimized = False
        self._last_hand_bbox: tuple[int, int, int, int] | None = None
        self._sequence_templates: dict[str, dict[str, object]] = {}
        self._sequence_armed_until: dict[str, float] = {}
        self._sequence_state: dict[str, dict] = {}
        self._sequence_pose_candidate_name = ""
        self._sequence_pose_candidate_count = 0
        self._sequence_last_stable_pose = ""
        self._captured_sequence_start_pose = ""
        self._captured_sequence_end_pose = ""
        self._last_sequence_pose_vector: np.ndarray | None = None
        self._last_sequence_color_signature: np.ndarray | None = None
        self._captured_sequence_start_vector: np.ndarray | None = None
        self._captured_sequence_mid_vector: np.ndarray | None = None
        self._captured_sequence_end_vector: np.ndarray | None = None
        self._captured_sequence_start_color_signature: np.ndarray | None = None
        self._captured_sequence_mid_color_signature: np.ndarray | None = None
        self._captured_sequence_end_color_signature: np.ndarray | None = None
        self._captured_sequence_start_preview_points: np.ndarray | None = None
        self._captured_sequence_mid_preview_points: np.ndarray | None = None
        self._captured_sequence_end_preview_points: np.ndarray | None = None
        self._sequence_selected_name = ""
        self._sequence_start_dirty = False
        self._sequence_end_dirty = False
        self._sequence_tolerance_value = self._SEQUENCE_DEFAULT_TOLERANCE
        self._feedback_animations: list = []
        self._capture_toast: QFrame | None = None
        self._capture_toast_close_timer = QTimer(self)
        self._capture_toast_close_timer.setSingleShot(True)
        self._capture_toast_close_timer.timeout.connect(self._close_capture_toast)

        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._refresh_gesture_options()
        self._reload_mappings()
        if not self._tracker.is_available:
            self.status_label.setText("Status: tracker unavailable")
            self._log(f"Tracker unavailable: {self._tracker.error_message}", "error")

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        self._apply_modern_theme()

        layout = QGridLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(14)

        control_box = QGroupBox()
        control_layout = QVBoxLayout(control_box)
        control_layout.setSpacing(10)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.start_button.setObjectName("primaryButton")
        self.stop_button.setObjectName("warningButton")
        self.stop_button.setEnabled(False)

        self.start_button.setToolTip("Start the camera and hand tracking.")
        self.stop_button.setToolTip("Stop camera and tracking.")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        self.status_label = QLabel("Status: idle")
        self.status_label.setObjectName("statusLabel")

        self.recognition_mode_combo = QComboBox()  # kept for tooltip/compat; hidden

        camera_row = QHBoxLayout()
        camera_row.setSpacing(8)
        camera_row.addWidget(self.start_button)
        camera_row.addWidget(self.stop_button)

        self.camera_feed_button = QPushButton("Camera Feed: On")
        self.camera_feed_button.setObjectName("secondaryButton")
        self.camera_feed_button.setCheckable(True)
        self.camera_feed_button.setChecked(True)
        self.camera_feed_button.setToolTip(
            "Turn camera image on/off. Tracking keeps running either way."
        )
        self.camera_feed_button.toggled.connect(self._on_camera_feed_toggled)

        self.record_motion_button = QPushButton("✚  Record New Motion")
        self.record_motion_button.setObjectName("recordButton")
        self.record_motion_button.setFixedHeight(40)
        self.record_motion_button.setToolTip(
            "Open the wizard to record a new start→end pose gesture."
        )
        self.record_motion_button.clicked.connect(self._open_record_wizard)

        control_layout.addWidget(self.record_motion_button)
        control_layout.addLayout(camera_row)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.camera_feed_button)

        self.manage_gestures_button = QPushButton("⚙️  Manage Gestures")
        self.manage_gestures_button.setObjectName("secondaryButton")
        self.manage_gestures_button.setFixedHeight(38)
        self.manage_gestures_button.clicked.connect(self._open_gesture_manager)
        control_layout.addWidget(self.manage_gestures_button)

        mapping_box = QGroupBox()
        mapping_layout = QVBoxLayout(mapping_box)
        mapping_layout.setSpacing(10)

        self.mapping_gesture_combo = QComboBox()
        self.mapping_shortcut_capture = QKeySequenceEdit()
        self.mapping_shortcut_capture.setMaximumSequenceLength(4)
        self.mapping_gesture_combo.currentTextChanged.connect(
            self._on_mapping_selection_changed
        )

        self.system_action_combo = QComboBox()
        self.system_action_combo.addItem("— none —", "")
        for label, payload in self._SYSTEM_ACTIONS:
            self.system_action_combo.addItem(label, payload)
        self.system_action_combo.setToolTip(
            "Pick a system/media action instead of a keyboard shortcut."
        )
        self.system_action_combo.currentIndexChanged.connect(self._on_system_action_changed)

        mapping_button_row = QHBoxLayout()
        self.save_mapping_button = QPushButton("Save Mapping")
        self.delete_mapping_button = QPushButton("Delete Gesture + Mapping")

        self.save_mapping_button.setObjectName("primaryButton")
        self.delete_mapping_button.setObjectName("warningButton")

        self.mapping_gesture_combo.setToolTip("Choose the gesture to map.")
        self.mapping_shortcut_capture.setToolTip("Press keys directly to capture shortcut.")

        self.save_mapping_button.clicked.connect(self.save_mapping)
        self.delete_mapping_button.clicked.connect(self.delete_selected_mapping)

        mapping_button_row.setSpacing(8)
        mapping_button_row.addWidget(self.save_mapping_button)
        mapping_button_row.addWidget(self.delete_mapping_button)

        self.mapping_list = QListWidget()
        self.mapping_list.setToolTip("Saved gesture -> shortcut mappings")

        sequence_box = QGroupBox()
        sequence_layout = QVBoxLayout(sequence_box)
        sequence_layout.setSpacing(6)

        self.sequence_name_input = QLineEdit()
        self.sequence_name_input.setPlaceholderText("test")

        self.sequence_tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.sequence_tolerance_slider.setRange(8, 60)
        self.sequence_tolerance_slider.setValue(int(self._SEQUENCE_DEFAULT_TOLERANCE * 100))
        self.sequence_tolerance_slider.valueChanged.connect(self._on_sequence_tolerance_changed)
        self.sequence_tolerance_label = QLabel("")
        self.sequence_tolerance_label.setObjectName("hintLabel")
        self._on_sequence_tolerance_changed(self.sequence_tolerance_slider.value())

        self.capture_sequence_start_button = QPushButton("Snapshot Start Pose")
        self.capture_sequence_mid_button = QPushButton("Snapshot Mid Pose  (optional)")
        self.sequence_hold_repeat_checkbox = QCheckBox("Repeat while held  (for volume, scrolling, etc.)")
        self.capture_sequence_end_button = QPushButton("Snapshot End Pose")
        self.save_sequence_button = QPushButton("Save Sequence Gesture")
        self.capture_sequence_start_button.setObjectName("secondaryButton")
        self.capture_sequence_mid_button.setObjectName("secondaryButton")
        self.capture_sequence_end_button.setObjectName("secondaryButton")
        self.save_sequence_button.setObjectName("primaryButton")

        self.sequence_start_label = QLabel("Start: -")
        self.sequence_mid_label = QLabel("Mid: -")
        self.sequence_end_label = QLabel("End: -")
        self.sequence_start_label.setObjectName("hintLabel")
        self.sequence_mid_label.setObjectName("hintLabel")
        self.sequence_end_label.setObjectName("hintLabel")
        _snap_style = "background: #101520; border: 1px solid #2a3447; border-radius: 8px; color: #8ea2c1;"
        self.sequence_start_preview_label = QLabel("No start snapshot")
        self.sequence_mid_preview_label = QLabel("No mid snapshot")
        self.sequence_end_preview_label = QLabel("No end snapshot")
        for lbl in (self.sequence_start_preview_label, self.sequence_mid_preview_label,
                    self.sequence_end_preview_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumHeight(88)
            lbl.setStyleSheet(_snap_style)

        self.capture_sequence_start_button.setToolTip(
            "Capture the current static hand pose as the sequence start."
        )
        self.capture_sequence_mid_button.setToolTip(
            "Capture the current pose as the middle checkpoint (optional)."
        )
        self.capture_sequence_end_button.setToolTip(
            "Capture the current static hand pose as the sequence end."
        )
        self.save_sequence_button.setToolTip(
            "Save start->end sequence as a mappable gesture."
        )
        self.sequence_tolerance_slider.setToolTip(
            "Lower = stricter matching. Higher = looser matching for both start and end poses."
        )

        self.capture_sequence_start_button.clicked.connect(self._capture_sequence_start_pose)
        self.capture_sequence_mid_button.clicked.connect(self._capture_sequence_mid_pose)
        self.capture_sequence_end_button.clicked.connect(self._capture_sequence_end_pose)
        self.save_sequence_button.clicked.connect(self.save_sequence_gesture)

        sequence_layout.addWidget(QLabel("Sequence name"))
        sequence_layout.addWidget(self.sequence_name_input)
        sequence_layout.addWidget(QLabel("Sequence tolerance"))
        sequence_layout.addWidget(self.sequence_tolerance_slider)
        sequence_layout.addWidget(self.sequence_tolerance_label)

        poses_row = QHBoxLayout()
        poses_row.setSpacing(8)
        for cap_btn, lbl, preview in (
            (self.capture_sequence_start_button, self.sequence_start_label, self.sequence_start_preview_label),
            (self.capture_sequence_mid_button,   self.sequence_mid_label,   self.sequence_mid_preview_label),
            (self.capture_sequence_end_button,   self.sequence_end_label,   self.sequence_end_preview_label),
        ):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(cap_btn)
            col.addWidget(lbl)
            col.addWidget(preview)
            poses_row.addLayout(col)
        sequence_layout.addLayout(poses_row)

        sequence_layout.addWidget(self.sequence_hold_repeat_checkbox)
        sequence_layout.addWidget(self.save_sequence_button)

        mapping_layout.addWidget(QLabel("Gesture"))
        mapping_layout.addWidget(self.mapping_gesture_combo)
        mapping_layout.addWidget(QLabel("Shortcut (press keys)"))
        mapping_layout.addWidget(self.mapping_shortcut_capture)
        mapping_layout.addWidget(QLabel("Or system action"))
        mapping_layout.addWidget(self.system_action_combo)
        mapping_layout.addLayout(mapping_button_row)
        mapping_layout.addWidget(sequence_box)

        # Gesture manager lives in a separate window so it doesn't crowd the main panel.
        self._gesture_manager = QDialog(self)
        self._gesture_manager.setWindowTitle("Gesture Manager")
        self._gesture_manager.setModal(False)
        self._gesture_manager.resize(780, 700)
        gm_root = QVBoxLayout(self._gesture_manager)
        gm_root.setContentsMargins(12, 12, 12, 12)
        gm_scroll = QScrollArea()
        gm_scroll.setWidgetResizable(True)
        gm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gm_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        gm_inner = QWidget()
        gm_inner_layout = QVBoxLayout(gm_inner)
        gm_inner_layout.setContentsMargins(0, 0, 0, 0)
        gm_inner_layout.setSpacing(12)
        gm_inner_layout.addWidget(mapping_box)
        gm_inner_layout.addStretch(1)
        gm_scroll.setWidget(gm_inner)
        gm_root.addWidget(gm_scroll)

        logs_box = QGroupBox()
        logs_layout = QVBoxLayout(logs_box)
        logs_layout.setSpacing(8)
        logs_button_row = QHBoxLayout()
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setObjectName("secondaryButton")
        self.clear_log_button.setToolTip("Clear all items from the session log.")
        self.clear_log_button.clicked.connect(self.clear_session_log)
        logs_button_row.addStretch(1)
        logs_button_row.addWidget(self.clear_log_button)
        self.log_list = QListWidget()
        self.log_list.setWordWrap(True)
        self.log_list.setToolTip("Runtime events and gesture detections")
        logs_layout.addLayout(logs_button_row)
        logs_layout.addWidget(self.log_list)

        preview_box = QGroupBox()
        preview_layout = QHBoxLayout(preview_box)
        preview_layout.setSpacing(12)
        self.preview_label = QLabel("Camera is not started")
        self.preview_label.setMinimumSize(320, 220)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        preview_layout.addWidget(self.preview_label, 1)

        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(12)
        left_panel_layout.addWidget(control_box)
        left_panel_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(250)
        left_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #2a3447; border-radius: 10px; background: #111824; }"
        )
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        logs_box.setMinimumWidth(250)
        logs_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        preview_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._main_layout = layout
        self._left_scroll = left_scroll
        self._preview_box = preview_box
        self._logs_box = logs_box
        self._responsive_layout_mode = ""

        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        width = max(self.width(), self.centralWidget().width())

        if width < 980:
            mode = "stacked"
        elif width < 1320:
            mode = "two_row"
        else:
            mode = "three_col"

        if mode == self._responsive_layout_mode:
            return

        self._responsive_layout_mode = mode
        layout = self._main_layout

        if mode == "stacked":
            self._left_scroll.setMinimumWidth(220)
            self._logs_box.setMinimumWidth(220)

            layout.addWidget(self._left_scroll, 0, 0, 1, 1)
            layout.addWidget(self._preview_box, 1, 0, 1, 1)
            layout.addWidget(self._logs_box, 2, 0, 1, 1)

            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 0)
            layout.setColumnStretch(2, 0)
            layout.setRowStretch(0, 0)
            layout.setRowStretch(1, 1)
            layout.setRowStretch(2, 1)
        elif mode == "two_row":
            self._left_scroll.setMinimumWidth(240)
            self._logs_box.setMinimumWidth(240)

            layout.addWidget(self._left_scroll, 0, 0, 1, 1)
            layout.addWidget(self._preview_box, 0, 1, 1, 1)
            layout.addWidget(self._logs_box, 1, 0, 1, 2)

            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 0)
            layout.setRowStretch(0, 1)
            layout.setRowStretch(1, 1)
            layout.setRowStretch(2, 0)
        else:
            self._left_scroll.setMinimumWidth(250)
            self._logs_box.setMinimumWidth(250)

            layout.addWidget(self._left_scroll, 0, 0, 3, 1)
            layout.addWidget(self._preview_box, 0, 1, 3, 1)
            layout.addWidget(self._logs_box, 0, 2, 3, 1)

            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 0)
            layout.setRowStretch(0, 1)
            layout.setRowStretch(1, 0)
            layout.setRowStretch(2, 0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._window_is_minimized = bool(
                self.windowState() & Qt.WindowState.WindowMinimized
            )

    def _is_window_obscured(self) -> bool:
        # Wayland minimize reporting is unreliable; exposure is the best signal.
        handle = self.windowHandle()
        if handle is not None and not handle.isExposed():
            return True
        return not self.isVisible() or self._window_is_minimized or self.isMinimized()

    @staticmethod
    def _make_app_icon() -> QIcon:
        """Load the project PNG logo; fall back to a generated icon if missing."""
        # Resolve assets/motion-cut.png relative to sys._MEIPASS (frozen) or project root
        candidates = []
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "assets", "motion-cut.png"))
        # dev: two levels up from this file → project root / assets
        candidates.append(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "motion-cut.png")
        )
        for path in candidates:
            norm = os.path.normpath(path)
            if os.path.isfile(norm):
                return QIcon(norm)
        # fallback: generated scissors icon
        import math
        size = 64
        px = QPixmap(size, size)
        px.fill(QColor(0, 0, 0, 0))
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(30, 30, 40)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, size, size, 12, 12)
        blade = QPen(QColor(220, 200, 80), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        cx, cy = size // 2, size // 2
        for sign in (1, -1):
            x0, y0 = cx - sign * 18, cy + 18
            x1, y1 = cx + sign * 14, cy - 16
            p.setPen(blade)
            p.drawLine(x0, y0, x1, y1)
            p.setPen(QPen(QColor(220, 200, 80), 2))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawEllipse(x0 - 6, y0 - 6, 12, 12)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - 3, cy - 3, 6, 6)
        p.end()
        return QIcon(px)

    @staticmethod
    def _make_gesture_icon(name: str, w: int, h: int) -> QPixmap:
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(0, 0, 0, 0))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 6

        def arrow(dx: float, dy: float, color: QColor) -> None:
            p.setPen(QPen(color, 2))
            sx, sy = int(cx - dx * r * 0.6), int(cy - dy * r * 0.6)
            ex, ey = int(cx + dx * r * 0.6), int(cy + dy * r * 0.6)
            p.drawLine(sx, sy, ex, ey)
            # arrowhead
            import math
            angle = math.atan2(ey - sy, ex - sx)
            for da in (0.5, -0.5):
                hx = int(ex - 10 * math.cos(angle + da))
                hy = int(ey - 10 * math.sin(angle + da))
                p.drawLine(ex, ey, hx, hy)

        if name == "move_up":
            arrow(0, -1, QColor(100, 200, 255))
        elif name == "move_down":
            arrow(0, 1, QColor(100, 200, 255))
        elif name == "move_left":
            arrow(-1, 0, QColor(100, 200, 255))
        elif name == "move_right":
            arrow(1, 0, QColor(100, 200, 255))
        elif name == "index_up":
            p.setPen(QPen(QColor(255, 200, 80), 2))
            p.drawEllipse(cx - 4, cy + 4, 8, 8)
            arrow(0, -1, QColor(255, 200, 80))
        elif name == "index_down":
            p.setPen(QPen(QColor(255, 200, 80), 2))
            p.drawEllipse(cx - 4, cy - 12, 8, 8)
            arrow(0, 1, QColor(255, 200, 80))
        elif name == "index_left":
            p.setPen(QPen(QColor(255, 200, 80), 2))
            p.drawEllipse(cx + 4, cy - 4, 8, 8)
            arrow(-1, 0, QColor(255, 200, 80))
        elif name == "index_right":
            p.setPen(QPen(QColor(255, 200, 80), 2))
            p.drawEllipse(cx - 12, cy - 4, 8, 8)
            arrow(1, 0, QColor(255, 200, 80))
        elif name == "pinch_in":
            p.setPen(QPen(QColor(255, 160, 60), 2))
            p.drawEllipse(cx - r // 2 - 5, cy - r // 2 - 5, 10, 10)
            p.drawEllipse(cx + r // 2 - 5, cy + r // 2 - 5, 10, 10)
            p.drawLine(cx - r // 2, cy - r // 2, cx - 3, cy - 3)
            p.drawLine(cx + r // 2, cy + r // 2, cx + 3, cy + 3)
        elif name == "pinch_out":
            p.setPen(QPen(QColor(100, 220, 160), 2))
            p.drawEllipse(cx - 5, cy - 5, 10, 10)
            p.drawLine(cx, cy - 5, cx, cy - r // 2 - 5)
            p.drawLine(cx, cy + 5, cx, cy + r // 2 + 5)
            p.drawLine(cx - 5, cy, cx - r // 2 - 5, cy)
            p.drawLine(cx + 5, cy, cx + r // 2 + 5, cy)
        elif name in ("finger_circle_cw", "finger_circle_ccw"):
            import math
            p.setPen(QPen(QColor(180, 100, 255), 2))
            points_count = 28
            angles = [i * 2 * math.pi * 0.82 / (points_count - 1) for i in range(points_count)]
            if name == "finger_circle_ccw":
                angles = angles[::-1]
            pts = [(int(cx + r * 0.7 * math.cos(a)), int(cy + r * 0.7 * math.sin(a))) for a in angles]
            for i in range(1, len(pts)):
                p.drawLine(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
            # arrowhead at end
            ex, ey = pts[-1]
            dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
            ang = math.atan2(dy, dx)
            for da in (0.5, -0.5):
                p.drawLine(ex, ey, int(ex - 8 * math.cos(ang + da)), int(ey - 8 * math.sin(ang + da)))
        else:
            # static gesture: draw a simple hand outline placeholder
            p.setPen(QPen(QColor(140, 170, 220), 2))
            p.drawRoundedRect(cx - r // 2, cy - r // 2, r, r, 6, 6)
            p.setPen(QPen(QColor(140, 170, 220), 1))
            label_map = {
                "closed_fist": "✊", "open_palm": "✋", "pointing_up": "☝",
                "thumb_up": "👍", "thumb_down": "👎", "victory": "✌", "iloveyou": "🤟",
            }
            p.setPen(QPen(QColor(200, 220, 255), 1))
            from PySide6.QtGui import QFont
            f = QFont()
            f.setPointSize(20)
            p.setFont(f)
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, label_map.get(name, "?"))

        p.end()
        return pixmap

    def _apply_modern_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #141821;
                color: #e8edf7;
                font-size: 13px;
            }

            QGroupBox {
                border: 1px solid #2a3447;
                border-radius: 10px;
                padding: 10px;
                background: #1a2130;
            }

            QLabel#statusLabel {
                background: #101520;
                border: 1px solid #283246;
                border-radius: 8px;
                padding: 7px 9px;
            }

            QLabel#hintLabel {
                color: #b7c2d8;
            }

            QLabel#previewLabel {
                background: #0e131c;
                color: #95a3bf;
                border: 1px solid #2a3447;
                border-radius: 10px;
            }

            QPushButton {
                background: #2b3548;
                border: 1px solid #3b4a63;
                border-radius: 8px;
                padding: 7px 11px;
            }

            QPushButton:hover {
                background: #34425a;
            }

            QPushButton:disabled {
                color: #8e9bb1;
                background: #222a39;
                border: 1px solid #2b3447;
            }

            QPushButton#primaryButton {
                background: #2374e1;
                border: 1px solid #2e82f0;
                color: #f4f9ff;
                font-weight: 600;
            }

            QPushButton#primaryButton:hover {
                background: #2a84ff;
            }

            QPushButton#primaryButton:disabled {
                background: #222a39;
                border: 1px solid #2b3447;
                color: #8e9bb1;
            }

            QPushButton#secondaryButton {
                background: #2c3e55;
            }

            QPushButton#secondaryButton:disabled {
                background: #222a39;
                border: 1px solid #2b3447;
                color: #8e9bb1;
            }

            QPushButton#warningButton {
                background: #7a5a1f;
                border: 1px solid #926a1f;
                color: #fff7e5;
            }

            QPushButton#warningButton:hover {
                background: #8b6723;
            }

            QPushButton#warningButton:disabled {
                background: #222a39;
                border: 1px solid #2b3447;
                color: #8e9bb1;
            }

            QPushButton#dangerButton {
                background: #7a2630;
                border: 1px solid #94313f;
                color: #fff0f2;
            }

            QPushButton#dangerButton:hover {
                background: #8d2d39;
            }

            QPushButton#dangerButton:disabled {
                background: #222a39;
                border: 1px solid #2b3447;
                color: #8e9bb1;
            }

            QPushButton#recordButton {
                background: #1e3a58;
                border: 1px solid #2a5a88;
                color: #9ec8ff;
                font-weight: 700;
                font-size: 13px;
            }

            QPushButton#recordButton:hover {
                background: #254870;
                border: 1px solid #3a7ac0;
                color: #cce4ff;
            }

            QLineEdit, QComboBox, QKeySequenceEdit {
                background: #121a27;
                border: 1px solid #30405a;
                border-radius: 8px;
                padding: 6px 8px;
            }

            QSlider::groove:horizontal {
                background: #223047;
                border: 1px solid #2f4361;
                height: 8px;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #6fb3ff;
                border: 1px solid #9acbff;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }

            QSlider::sub-page:horizontal {
                background: #2c7be5;
                border-radius: 4px;
            }

            QSlider::add-page:horizontal {
                background: #1b2638;
                border-radius: 4px;
            }

            QScrollBar:vertical {
                background: #151e2c;
                width: 12px;
                margin: 2px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background: #4f6d95;
                min-height: 28px;
                border-radius: 6px;
                border: 1px solid #6f8db5;
            }

            QScrollBar::handle:vertical:hover {
                background: #6b8db8;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: #223047;
                height: 12px;
                border-radius: 4px;
                border: 1px solid #2f4361;
            }

            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                width: 7px;
                height: 7px;
                background: #9ec2ff;
            }

            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }

            QListWidget {
                background: #111824;
                border: 1px solid #2a3447;
                border-radius: 8px;
                padding: 4px;
            }

            QListWidget::item {
                padding: 4px 6px;
                border-radius: 6px;
            }

            QListWidget::item:selected {
                background: #2a4a7e;
            }
            """
        )

    def _open_gesture_manager(self) -> None:
        self._gesture_manager.setStyleSheet(self.styleSheet())
        self._gesture_manager.show()
        self._gesture_manager.raise_()
        self._gesture_manager.activateWindow()

    def _open_record_wizard(self) -> None:
        from motion_cut.ui.wizard import RecordMotionWizard

        was_running = self._timer.isActive()
        if was_running:
            self._timer.stop()

        wizard = RecordMotionWizard(self._stream, self._tracker, parent=self)
        wizard.gesture_ready.connect(self._on_wizard_gesture_ready)
        wizard.exec()

        if was_running and self._stream._cap is not None:
            self._timer.start()

    def _on_wizard_gesture_ready(
        self, name: str, shortcut: str, sample_path: str, tolerance: float,
        hold_repeat: bool = False,
    ) -> None:
        existed_before = name in self._sequence_templates
        template = self._repository.upsert_gesture_template(
            name=name, sample_path=sample_path, threshold=tolerance
        )
        parsed = self._parse_sequence_points_sample(template.sample_path)
        if parsed is not None:
            self._sequence_templates[template.name] = parsed

        if shortcut:
            shortcut = self._normalize_shortcut(shortcut)
            self._repository.upsert_gesture_mapping(
                gesture_name=name, action_type="shortcut", action_payload=shortcut
            )
            self._shortcut_mappings[name] = shortcut

        if not existed_before:
            self._refresh_gesture_options()
        self._reload_mappings()
        self._log(
            f"New gesture saved: '{name}'" + (f"  →  {shortcut}" if shortcut else ""),
            "success",
        )

    def start_camera(self) -> None:
        try:
            self._stream.open()
        except CameraBusyError as exc:
            self.status_label.setText("Status: camera busy")
            self._log("Camera busy — close other camera apps and retry", "error")
            QMessageBox.warning(
                self,
                "Camera In Use",
                "Camera is being used by another application. Close it and click Start Camera again.",
            )
            return
        except RuntimeError as exc:
            self.status_label.setText("Status: camera unavailable")
            self._log(f"Camera error: {exc}", "error")
            QMessageBox.critical(self, "Camera Error", str(exc))
            return

        self._tracker.reset_runtime()
        self._timer.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: running")
        self._log(f"Camera started  (backend: {self._tracker.backend})", "system")

    def stop_camera(self) -> None:
        self._timer.stop()
        self._stream.close()
        self._reset_sequence_runtime()
        self._no_landmark_streak = 0
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: stopped")
        self.preview_label.setText("Camera is not started")
        self._log("Camera stopped", "system")

    def _recover_tracker_backend(self) -> None:
        now = time.monotonic()
        if (now - self._last_tracker_recover_at) < self._tracker_recover_cooldown_seconds:
            return

        self._last_tracker_recover_at = now
        old_tracker = self._tracker
        self._tracker = HandTracker(
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._tracker.reset_runtime()
        self._no_landmark_streak = 0
        self._had_landmarks_last_tick = False

        if old_tracker is not None:
            old_tracker.close()

    def _recover_camera_stream(self) -> None:
        now = time.monotonic()
        if (now - self._last_camera_recover_at) < self._camera_recover_cooldown_seconds:
            return

        self._last_camera_recover_at = now
        try:
            self._stream.close()
            self._stream.open()
            self._frame_miss_streak = 0
            self.status_label.setText("Status: running")
            self._log("Camera stream reconnected", "system")
        except CameraBusyError:
            self.status_label.setText("Status: camera busy")
            self._log("Camera busy — waiting for device to be released", "warn")
        except RuntimeError as exc:
            self.status_label.setText("Status: camera unavailable")
            self._log(f"Camera stream error: {exc}", "error")

    def _capture_sequence_start_pose(self) -> None:
        self._capture_sequence_pose(which="start")

    def _capture_sequence_mid_pose(self) -> None:
        self._capture_sequence_pose(which="mid")

    def _capture_sequence_end_pose(self) -> None:
        self._capture_sequence_pose(which="end")

    def _capture_sequence_pose(self, which: str) -> None:
        pose_vector = self._last_sequence_pose_vector
        if pose_vector is None:
            QMessageBox.information(
                self,
                "Pose Not Ready",
                "Show your hand in frame first, then snapshot again.",
            )
            return

        predicted_name = "custom_pose"

        if which == "start":
            self._captured_sequence_start_vector = pose_vector.copy()
            self._captured_sequence_start_color_signature = (
                None
                if self._last_sequence_color_signature is None
                else self._last_sequence_color_signature.copy()
            )
            self._captured_sequence_start_preview_points = self._compute_snapshot_preview_points(
                self._captured_sequence_start_vector
            )
            self._sequence_start_dirty = True
            self._captured_sequence_start_pose = predicted_name
            self.sequence_start_label.setText(f"Start: {predicted_name}")
            self.sequence_start_preview_label.setPixmap(
                self._make_sequence_snapshot_pixmap(
                    self._captured_sequence_start_vector,
                    self._captured_sequence_start_color_signature,
                    self._captured_sequence_start_preview_points,
                )
            )
            self._log("Start pose captured", "info")
        elif which == "mid":
            self._captured_sequence_mid_vector = pose_vector.copy()
            self._captured_sequence_mid_color_signature = (
                None if self._last_sequence_color_signature is None
                else self._last_sequence_color_signature.copy()
            )
            self._captured_sequence_mid_preview_points = self._compute_snapshot_preview_points(
                self._captured_sequence_mid_vector
            )
            self.sequence_mid_label.setText(f"Mid: {predicted_name}")
            self.sequence_mid_preview_label.setPixmap(
                self._make_sequence_snapshot_pixmap(
                    self._captured_sequence_mid_vector,
                    self._captured_sequence_mid_color_signature,
                    self._captured_sequence_mid_preview_points,
                )
            )
            self._log("Mid pose captured", "info")
        else:
            self._captured_sequence_end_vector = pose_vector.copy()
            self._captured_sequence_end_color_signature = (
                None
                if self._last_sequence_color_signature is None
                else self._last_sequence_color_signature.copy()
            )
            self._captured_sequence_end_preview_points = self._compute_snapshot_preview_points(
                self._captured_sequence_end_vector
            )
            self._sequence_end_dirty = True
            self._captured_sequence_end_pose = predicted_name
            self.sequence_end_label.setText(f"End: {predicted_name}")
            self.sequence_end_preview_label.setPixmap(
                self._make_sequence_snapshot_pixmap(
                    self._captured_sequence_end_vector,
                    self._captured_sequence_end_color_signature,
                    self._captured_sequence_end_preview_points,
                )
            )
            self._log("End pose captured", "info")

    def _make_sequence_snapshot_pixmap(
        self,
        pose_vector: np.ndarray,
        color_signature: np.ndarray | None = None,
        preview_points: np.ndarray | None = None,
    ) -> QPixmap:
        width = 220
        height = 92
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#101520"))

        if pose_vector.shape != (21, 3):
            return pixmap

        if isinstance(preview_points, np.ndarray) and preview_points.shape == (21, 2):
            mapped = np.empty((21, 2), dtype=np.float32)
            mapped[:, 0] = np.clip(preview_points[:, 0], 0.0, 1.0) * float(width)
            mapped[:, 1] = np.clip(preview_points[:, 1], 0.0, 1.0) * float(height)
        else:
            mapped = self._compute_snapshot_preview_pixels(pose_vector, width, height)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bone_pen = QPen(QColor("#4f6d95"), 2)
        painter.setPen(bone_pen)
        for a, b in self._HAND_CONNECTIONS:
            painter.drawLine(
                int(mapped[a, 0]), int(mapped[a, 1]),
                int(mapped[b, 0]), int(mapped[b, 1]),
            )

        signature_colors: dict[int, QColor] = {}
        if isinstance(color_signature, np.ndarray) and color_signature.shape == (6, 3):
            key_indices = (0, 4, 8, 12, 16, 20)
            hsv_u8 = np.clip(
                np.column_stack((
                    color_signature[:, 0] * 180.0,
                    color_signature[:, 1] * 255.0,
                    color_signature[:, 2] * 255.0,
                )),
                0,
                255,
            ).astype(np.uint8)
            hsv_pixels = hsv_u8.reshape((-1, 1, 3))
            bgr_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR).reshape((-1, 3))
            for i, idx in enumerate(key_indices):
                b, g, r = bgr_pixels[i]
                signature_colors[idx] = QColor(int(r), int(g), int(b))

        tip_indices = {4, 8, 12, 16, 20}
        for idx in range(mapped.shape[0]):
            if idx in signature_colors:
                color = signature_colors[idx]
                radius = 4 if idx == 0 else 3
            elif idx == 0:
                color = QColor("#ffffff")
                radius = 4
            elif idx in tip_indices:
                color = QColor("#9de7b8")
                radius = 3
            else:
                color = QColor("#9ec2ff")
                radius = 2

            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            painter.drawEllipse(
                int(mapped[idx, 0] - radius),
                int(mapped[idx, 1] - radius),
                radius * 2,
                radius * 2,
            )

        painter.end()
        return pixmap

    @staticmethod
    def _compute_snapshot_preview_pixels(
        pose_vector: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        points = pose_vector[:, :2].astype(np.float32)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) * 0.5
        span_x = float(max(maxs[0] - mins[0], 1e-5))
        span_y = float(max(maxs[1] - mins[1], 1e-5))
        content_w = width * 0.76
        content_h = height * 0.76
        scale_x = content_w / span_x
        scale_y = content_h / span_y
        scale = min(scale_x, scale_y)
        mapped = np.empty((points.shape[0], 2), dtype=np.float32)
        mapped[:, 0] = (points[:, 0] - center[0]) * scale + (width * 0.5)
        mapped[:, 1] = (points[:, 1] - center[1]) * scale + (height * 0.5)
        return mapped

    def _compute_snapshot_preview_points(self, pose_vector: np.ndarray) -> np.ndarray:
        width = 220
        height = 92
        pixels = self._compute_snapshot_preview_pixels(pose_vector, width, height)
        norm = np.empty((pixels.shape[0], 2), dtype=np.float32)
        norm[:, 0] = np.clip(pixels[:, 0] / float(width), 0.0, 1.0)
        norm[:, 1] = np.clip(pixels[:, 1] / float(height), 0.0, 1.0)
        return norm

    def save_sequence_gesture(self) -> None:
        raw_name = self.sequence_name_input.text().strip().lower()
        gesture_name = re.sub(r"[^a-z0-9]+", "_", raw_name).strip("_")
        if not gesture_name:
            QMessageBox.information(self, "Missing Name", "Enter a gesture name like test.")
            return

        existed_before = gesture_name in self._sequence_templates

        start_pose = self._captured_sequence_start_pose.strip() or "custom_pose"
        end_pose = self._captured_sequence_end_pose.strip() or "custom_pose"
        if self._captured_sequence_start_vector is None or self._captured_sequence_end_vector is None:
            QMessageBox.information(
                self,
                "Missing Snapshots",
                "Capture both start and end poses before saving.",
            )
            return

        tolerance = float(np.clip(self.sequence_tolerance_slider.value() / 100.0, 0.08, 0.60))
        sample_path = self._encode_points_sequence_sample(
            self._captured_sequence_start_vector,
            self._captured_sequence_end_vector,
            max_gap_seconds=self._SEQUENCE_DEFAULT_MAX_GAP_SECONDS,
            start_tolerance=tolerance,
            end_tolerance=tolerance,
            start_color_signature=self._captured_sequence_start_color_signature,
            end_color_signature=self._captured_sequence_end_color_signature,
            start_preview_points=self._captured_sequence_start_preview_points,
            end_preview_points=self._captured_sequence_end_preview_points,
            start_color_tolerance=self._SEQUENCE_SWEET_SPOT_START_COLOR_TOLERANCE,
            end_color_tolerance=self._SEQUENCE_SWEET_SPOT_END_COLOR_TOLERANCE,
            mid_vector=self._captured_sequence_mid_vector,
            mid_color_signature=self._captured_sequence_mid_color_signature,
            mid_preview_points=self._captured_sequence_mid_preview_points,
            hold_repeat=self.sequence_hold_repeat_checkbox.isChecked(),
        )
        template = self._repository.upsert_gesture_template(
            name=gesture_name,
            sample_path=sample_path,
            threshold=tolerance,
        )

        # Keep cache in sync with the exact snapshots just saved.
        self._sequence_templates[template.name] = {
            "mode": "points",
            "start_vector": self._captured_sequence_start_vector.copy(),
            "end_vector": self._captured_sequence_end_vector.copy(),
            "mid_vector": (
                self._captured_sequence_mid_vector.copy()
                if isinstance(self._captured_sequence_mid_vector, np.ndarray) else None
            ),
            "start_tolerance": tolerance,
            "end_tolerance": tolerance,
            "start_color_signature": (
                self._captured_sequence_start_color_signature.copy()
                if isinstance(self._captured_sequence_start_color_signature, np.ndarray)
                else None
            ),
            "end_color_signature": (
                self._captured_sequence_end_color_signature.copy()
                if isinstance(self._captured_sequence_end_color_signature, np.ndarray)
                else None
            ),
            "start_preview_points": (
                self._captured_sequence_start_preview_points.copy()
                if isinstance(self._captured_sequence_start_preview_points, np.ndarray)
                else None
            ),
            "end_preview_points": (
                self._captured_sequence_end_preview_points.copy()
                if isinstance(self._captured_sequence_end_preview_points, np.ndarray)
                else None
            ),
            "start_color_tolerance": self._SEQUENCE_SWEET_SPOT_START_COLOR_TOLERANCE,
            "end_color_tolerance": self._SEQUENCE_SWEET_SPOT_END_COLOR_TOLERANCE,
            "max_gap_seconds": self._SEQUENCE_DEFAULT_MAX_GAP_SECONDS,
            "hold_repeat": self.sequence_hold_repeat_checkbox.isChecked(),
        }

        # Refresh combo options only when this is a new gesture name.
        if not existed_before:
            self._refresh_gesture_options()

        self._sequence_selected_name = template.name
        self._sequence_start_dirty = False
        self._sequence_end_dirty = False
        self._log(f"Gesture saved: '{template.name}'  (tolerance {tolerance:.2f})", "success")
        self._animate_button_feedback(self.save_sequence_button)

    def _on_sequence_tolerance_changed(self, value: int) -> None:
        tolerance = float(np.clip(value / 100.0, 0.08, 0.60))
        self._sequence_tolerance_value = tolerance
        if tolerance <= 0.16:
            profile = "strict"
        elif tolerance <= 0.30:
            profile = "balanced"
        else:
            profile = "loose"
        self.sequence_tolerance_label.setText(
            f"Tolerance: {tolerance:.2f} ({profile})"
        )

    def save_mapping(self) -> None:
        gesture_name = self.mapping_gesture_combo.currentText().strip()
        if not gesture_name:
            QMessageBox.information(self, "Missing Gesture", "Select a gesture first.")
            return

        # A chosen system action takes priority over the captured key sequence.
        payload = str(self.system_action_combo.currentData() or "")
        if not payload:
            payload = self.mapping_shortcut_capture.keySequence().toString(
                QKeySequence.SequenceFormat.PortableText
            ).strip()

        if not payload:
            QMessageBox.information(
                self,
                "Missing Action",
                "Press a shortcut or pick a system action.",
            )
            return

        payload = self._normalize_shortcut(payload)

        mapping = self._repository.upsert_gesture_mapping(
            gesture_name=gesture_name,
            action_type="shortcut",
            action_payload=payload,
        )
        self._shortcut_mappings[gesture_name] = mapping.action_payload
        self._reload_mappings()
        self._log(f"Mapping saved: {gesture_name}  →  {payload}", "success")
        self._animate_button_feedback(self.save_mapping_button)

    def test_mapping(self) -> None:
        gesture_name = self.mapping_gesture_combo.currentText().strip()
        if not gesture_name:
            return

        shortcut = self._shortcut_mappings.get(gesture_name)
        if not shortcut:
            QMessageBox.information(
                self,
                "Mapping Not Found",
                "Save a mapping for this gesture first.",
            )
            return

        self._execute_shortcut(shortcut)
        self._log(f"Test fired: {gesture_name}  →  {shortcut}", "trigger")

    def delete_selected_mapping(self) -> None:
        selected = self.mapping_list.currentItem()
        gesture_name = ""
        if selected is not None:
            row_text = selected.text()
            if "->" in row_text:
                gesture_name = row_text.split("->", 1)[0].strip()

        if not gesture_name:
            gesture_name = self.mapping_gesture_combo.currentText().strip()

        if not gesture_name:
            QMessageBox.information(
                self,
                "No Gesture",
                "Select a mapped row or choose a gesture first.",
            )
            return

        confirmation = QMessageBox.question(
            self,
            "Delete Gesture And Mapping",
            (
                f"Delete gesture '{gesture_name}' and all its shortcut mappings?\n\n"
                "Built-in gestures will keep existing as recognizers; only their mappings are removed."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return

        deleted_mappings = self._repository.delete_gesture_mappings(gesture_name)
        deleted_templates = self._repository.delete_gesture_template(gesture_name)

        if deleted_mappings <= 0 and deleted_templates <= 0:
            QMessageBox.information(
                self,
                "Nothing Deleted",
                f"No saved template or mapping found for '{gesture_name}'.",
            )
            self._refresh_gesture_options()
            self._reload_mappings()
            return

        self._shortcut_mappings.pop(gesture_name, None)
        self._last_trigger_at.pop(gesture_name, None)
        self._refresh_gesture_options()
        self._reload_mappings()
        self._log(f"Gesture deleted: '{gesture_name}'", "warn")
        self._animate_button_feedback(self.delete_mapping_button)

    def _tick(self) -> None:
        try:
            source_frame = self._stream.read()
        except RuntimeError:
            self._frame_miss_streak += 1
            if self._frame_miss_streak >= self._frame_recover_after_frames:
                self._recover_camera_stream()
            return

        if source_frame is None:
            self._frame_miss_streak += 1
            if self._frame_miss_streak >= self._frame_recover_after_frames:
                self._recover_camera_stream()
            return

        self._frame_miss_streak = 0

        frame = source_frame.copy() if self._camera_feed_enabled else np.zeros_like(source_frame)

        now = time.monotonic()
        if self._last_tick_at > 0.0:
            elapsed = now - self._last_tick_at
            if elapsed > 1e-6:
                instant_fps = 1.0 / elapsed
                if self._preview_fps <= 0.0:
                    self._preview_fps = instant_fps
                else:
                    self._preview_fps = (self._preview_fps * 0.85) + (instant_fps * 0.15)
        self._last_tick_at = now

        tracking_frame = source_frame
        frame_height, frame_width = source_frame.shape[:2]
        if frame_width > self._tracking_resize_width:
            tracking_frame = cv2.resize(
                source_frame,
                (self._tracking_resize_width, self._tracking_resize_height),
                interpolation=cv2.INTER_LINEAR,
            )

        result = self._tracker.process(tracking_frame)

        if result.landmarks:
            self._no_landmark_streak = 0
            self._had_landmarks_last_tick = True
            hand_landmarks = result.landmarks[0]
            self._last_sequence_pose_vector = self._build_sequence_pose_vector(hand_landmarks)
            self._last_sequence_color_signature = self._build_sequence_color_signature(
                tracking_frame,
                hand_landmarks,
            )

            gesture, source = self._classify_live_gesture(tracking_frame, hand_landmarks)

            if source == "sequence" and gesture.name != "unknown":
                # sequence gestures bypass the stability gate
                self._last_stable_gesture = gesture.name
                self._gesture_candidate_name = gesture.name
                self._gesture_candidate_count = self._gesture_stability_frames
                self._maybe_trigger_mapping(gesture.name, gesture.confidence)
            else:
                stable_gesture = self._stable_gesture(gesture.name, gesture.confidence)
                self._maybe_trigger_mapping(stable_gesture, gesture.confidence)
        else:
            self._no_landmark_streak += 1
            self._last_sequence_pose_vector = None
            self._last_sequence_color_signature = None
            self._had_landmarks_last_tick = False
            self._reset_gesture_stability()
            if self._no_landmark_streak >= self._tracker_recover_after_frames:
                self._recover_tracker_backend()

        if self._guide_overlay_enabled and self._camera_feed_enabled:
            self._draw_gesture_guide(frame)

        if self._camera_feed_enabled:
            self._tracker.draw_landmarks(frame, result.landmarks)
        else:
            self._draw_centered_tracking(frame, result.landmarks)

        self._draw_frame(frame)

    @staticmethod
    def _extract_landmark_xy(points) -> np.ndarray:
        if not points:
            return np.empty((0, 2), dtype=np.float32)

        coords = np.empty((len(points), 2), dtype=np.float32)
        for i, point in enumerate(points):
            if isinstance(point, (tuple, list)):
                coords[i, 0] = float(point[0])
                coords[i, 1] = float(point[1])
            else:
                coords[i, 0] = float(point.x)
                coords[i, 1] = float(point.y)
        return coords

    def _draw_centered_tracking(self, frame, hands_landmarks: list) -> None:
        if not hands_landmarks:
            return

        hand_landmarks = hands_landmarks[0]
        points = getattr(hand_landmarks, "landmark", hand_landmarks)
        xy = self._extract_landmark_xy(points)
        if xy.shape[0] < 2:
            return

        h, w = frame.shape[:2]
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(np.max(maxs - mins))
        if span <= 1e-5:
            return

        zoom_ratio = 0.72
        scale = (min(w, h) * zoom_ratio) / span
        mapped = np.empty((xy.shape[0], 2), dtype=np.float32)
        mapped[:, 0] = (xy[:, 0] - center[0]) * scale + (w * 0.5)
        mapped[:, 1] = (xy[:, 1] - center[1]) * scale + (h * 0.5)

        palette = getattr(self._tracker, "_LANDMARK_COLORS", ())

        for start, end in self._HAND_CONNECTIONS:
            if start >= mapped.shape[0] or end >= mapped.shape[0]:
                continue
            cv2.line(
                frame,
                (int(mapped[start, 0]), int(mapped[start, 1])),
                (int(mapped[end, 0]), int(mapped[end, 1])),
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )

        for idx in range(mapped.shape[0]):
            if idx < len(palette):
                color = palette[idx]
            else:
                color = (255, 255, 255)

            cx = int(mapped[idx, 0])
            cy = int(mapped[idx, 1])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

    def _classify_live_gesture(self, tracking_frame, hand_landmarks) -> tuple[GesturePrediction, str]:
        static_gesture = GesturePrediction(name="unknown", confidence=0.0)
        seq_gesture = self._classify_sequence_gesture(static_gesture, hand_landmarks)
        if seq_gesture is not None:
            return seq_gesture, "sequence"

        return static_gesture, "none"

    def _classify_sequence_gesture(
        self,
        static_prediction: GesturePrediction,
        hand_landmarks,
    ) -> GesturePrediction | None:
        if not self._sequence_templates:
            return None

        now = self._last_tick_at if self._last_tick_at > 0.0 else time.monotonic()
        for gesture_name, deadline in list(self._sequence_armed_until.items()):
            if now > deadline:
                self._sequence_armed_until.pop(gesture_name, None)

        stable_pose = self._stable_sequence_pose(static_prediction)
        current_pose = self._build_sequence_pose_vector(hand_landmarks)

        fired_name = ""
        fired_confidence = 0.0
        for gesture_name, template in self._sequence_templates.items():
            mode = str(template.get("mode", "labels"))
            deadline = self._sequence_armed_until.get(gesture_name, 0.0)

            if mode == "points":
                confidence = self._advance_points_sequence(
                    gesture_name, template, current_pose, now
                )
                if confidence > 0.0:
                    fired_name = gesture_name
                    fired_confidence = max(fired_confidence, confidence)
                continue

            start_pose = str(template.get("start_pose", ""))
            end_pose = str(template.get("end_pose", ""))
            min_confidence = float(template.get("min_confidence", self._SEQUENCE_DEFAULT_THRESHOLD))
            max_gap_seconds = float(template.get("max_gap_seconds", self._SEQUENCE_DEFAULT_MAX_GAP_SECONDS))

            if stable_pose == start_pose:
                self._sequence_armed_until[gesture_name] = now + max_gap_seconds
                continue

            if stable_pose != end_pose:
                continue

            if deadline >= now and static_prediction.confidence >= min_confidence:
                self._sequence_armed_until.pop(gesture_name, None)
                fired_name = gesture_name
                fired_confidence = max(fired_confidence, static_prediction.confidence)

        if not fired_name:
            return None

        confidence = float(np.clip(fired_confidence, 0.65, 0.95))
        return GesturePrediction(name=fired_name, confidence=confidence)

    def _advance_points_sequence(
        self,
        gesture_name: str,
        template: dict,
        current_pose: np.ndarray,
        now: float,
    ) -> float:
        """Run the start -> (middle) -> end state machine. Returns confidence on fire."""
        start_vector = template.get("start_vector")
        end_vector = template.get("end_vector")
        if not isinstance(start_vector, np.ndarray) or not isinstance(end_vector, np.ndarray):
            return 0.0

        mid_vector = template.get("mid_vector")
        has_mid = isinstance(mid_vector, np.ndarray)

        start_tolerance = float(template.get("start_tolerance", self._SEQUENCE_DEFAULT_TOLERANCE))
        end_tolerance = float(template.get("end_tolerance", self._SEQUENCE_DEFAULT_TOLERANCE))
        mid_tolerance = float(
            template.get("mid_tolerance", (start_tolerance + end_tolerance) * 0.5)
        )
        max_gap_seconds = float(
            template.get("max_gap_seconds", self._SEQUENCE_DEFAULT_MAX_GAP_SECONDS)
        )

        # Nearest-template rule: the frame belongs to the single closest checkpoint
        # that is also within tolerance. An ambiguous tie belongs to no zone, which
        # keeps overlapping tolerance ranges from hijacking the state machine.
        end_distance = self._sequence_pose_distance(current_pose, end_vector)
        candidates = [
            ("start", self._sequence_pose_distance(current_pose, start_vector), start_tolerance),
            ("end", end_distance, end_tolerance),
        ]
        if has_mid:
            candidates.append(
                ("mid", self._sequence_pose_distance(current_pose, mid_vector), mid_tolerance)
            )

        zone = ""
        best_distance = float("inf")
        ambiguous = False
        for name, distance, tolerance in candidates:
            if distance > tolerance:
                continue
            if distance < best_distance - 1e-6:
                best_distance = distance
                zone = name
                ambiguous = False
            elif abs(distance - best_distance) <= 1e-6:
                ambiguous = True
        if ambiguous:
            zone = ""

        state = self._sequence_state.setdefault(
            gesture_name, {"phase": 0, "streak": 0, "deadline": 0.0, "departed": False, "hold": False}
        )

        if state["phase"] > 0 and now > state["deadline"]:
            state["phase"] = 0
            state["streak"] = 0

        if state["phase"] == 0:
            # Hold-to-repeat: while end zone held after a fire, repeat at fast interval.
            if state["hold"]:
                if zone == "end":
                    hold_elapsed = now - state.get("hold_last", now)
                    if hold_elapsed >= self._SEQUENCE_HOLD_REPEAT_INTERVAL:
                        state["hold_last"] = now
                        # Pre-adjust so _maybe_trigger_mapping's cooldown won't block this.
                        self._last_trigger_at[gesture_name] = now - self._trigger_cooldown_seconds
                        return 0.75
                    return 0.0
                else:
                    state["hold"] = False

            state["streak"] = state["streak"] + 1 if zone == "start" else 0
            if state["streak"] >= self._SEQUENCE_STABLE_FRAMES:
                # Skip straight to the end-watch phase when no middle was recorded.
                state["phase"] = 1 if has_mid else 2
                state["streak"] = 0
                state["departed"] = False
                state["deadline"] = now + max_gap_seconds
            return 0.0

        # The hand must physically leave the start zone before the end can count.
        if zone != "start":
            state["departed"] = True

        if state["phase"] == 1:
            state["streak"] = state["streak"] + 1 if zone == "mid" else 0
            if state["streak"] >= self._SEQUENCE_MID_STABLE_FRAMES:
                state["phase"] = 2
                state["streak"] = 0
            return 0.0

        # phase 2: watching for the end pose
        counts = zone == "end" and state["departed"]
        state["streak"] = state["streak"] + 1 if counts else 0
        if state["streak"] < self._SEQUENCE_STABLE_FRAMES:
            return 0.0

        state["phase"] = 0
        state["streak"] = 0
        state["departed"] = False
        state["hold"] = bool(template.get("hold_repeat", False))  # only enter hold mode if opted in
        state["hold_last"] = now
        match_confidence = 1.0 - min(end_distance / max(end_tolerance, 1e-6), 1.0)
        if has_mid:
            match_confidence += 0.08  # a matched middle is strong extra evidence
        return float(np.clip(match_confidence, 0.65, 0.95))

    def _stable_sequence_pose(self, prediction: GesturePrediction) -> str | None:
        if prediction.name == "unknown" or prediction.confidence < 0.55:
            self._sequence_pose_candidate_name = ""
            self._sequence_pose_candidate_count = 0
            return None

        if prediction.name == self._sequence_pose_candidate_name:
            self._sequence_pose_candidate_count += 1
        else:
            self._sequence_pose_candidate_name = prediction.name
            self._sequence_pose_candidate_count = 1

        if self._sequence_pose_candidate_count < self._SEQUENCE_STABLE_FRAMES:
            return None

        if prediction.name == self._sequence_last_stable_pose:
            return None

        self._sequence_last_stable_pose = prediction.name
        return prediction.name

    @staticmethod
    def _parse_sequence_template_sample(sample_path: str) -> tuple[str, str, float] | None:
        parts = sample_path.strip().split(":")
        if len(parts) < 3:
            return None
        if parts[0] != MainWindow._SEQUENCE_SAMPLE_PREFIX:
            return None

        start_pose = parts[1].strip()
        end_pose = parts[2].strip()
        if not start_pose or not end_pose:
            return None

        max_gap = MainWindow._SEQUENCE_DEFAULT_MAX_GAP_SECONDS
        if len(parts) >= 4:
            try:
                max_gap = float(parts[3])
            except ValueError:
                max_gap = MainWindow._SEQUENCE_DEFAULT_MAX_GAP_SECONDS
        max_gap = float(np.clip(max_gap, 0.4, 5.0))
        return start_pose, end_pose, max_gap

    @classmethod
    def _parse_sequence_points_sample(cls, sample_path: str) -> dict[str, object] | None:
        """Decode a stored points sample into a ready-to-use template dict."""
        if not sample_path.startswith(f"{cls._SEQUENCE_POINTS_PREFIX}:"):
            return None

        payload = sample_path[len(cls._SEQUENCE_POINTS_PREFIX) + 1 :]
        try:
            obj = json.loads(payload)
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        def vec(key: str) -> np.ndarray | None:
            arr = np.asarray(obj.get(key, []), dtype=np.float32)
            return arr.reshape(21, 3) if arr.size == 63 else None

        def color(key: str) -> np.ndarray | None:
            raw = obj.get(key)
            if not isinstance(raw, list):
                return None
            arr = np.asarray(raw, dtype=np.float32)
            return arr.reshape(6, 3) if arr.size == 18 else None

        def preview(key: str) -> np.ndarray | None:
            raw = obj.get(key)
            if not isinstance(raw, list):
                return None
            arr = np.asarray(raw, dtype=np.float32)
            return np.clip(arr.reshape(21, 2), 0.0, 1.0) if arr.size == 42 else None

        def num(key: str, fallback: float, low: float, high: float) -> float:
            raw = obj.get(key)
            if raw is None:
                raw = obj.get("tolerance", fallback)
            try:
                return float(np.clip(float(raw), low, high))
            except (TypeError, ValueError):
                return fallback

        start_vector = vec("start")
        end_vector = vec("end")
        if start_vector is None or end_vector is None:
            return None

        start_tolerance = num("start_tolerance", cls._SEQUENCE_DEFAULT_TOLERANCE, 0.08, 0.60)
        end_tolerance = num("end_tolerance", cls._SEQUENCE_DEFAULT_TOLERANCE, 0.08, 0.60)

        template: dict[str, object] = {
            "mode": "points",
            "start_vector": start_vector,
            "end_vector": end_vector,
            "mid_vector": vec("mid"),
            "start_tolerance": start_tolerance,
            "end_tolerance": end_tolerance,
            "mid_tolerance": num(
                "mid_tolerance", (start_tolerance + end_tolerance) * 0.5, 0.08, 0.60
            ),
            "start_color_signature": color("start_color"),
            "end_color_signature": color("end_color"),
            "mid_color_signature": color("mid_color"),
            "start_preview_points": preview("start_preview_points"),
            "end_preview_points": preview("end_preview_points"),
            "mid_preview_points": preview("mid_preview_points"),
            "start_color_tolerance": num(
                "start_color_tolerance",
                cls._SEQUENCE_SWEET_SPOT_START_COLOR_TOLERANCE, 0.08, 0.80,
            ),
            "end_color_tolerance": num(
                "end_color_tolerance",
                cls._SEQUENCE_SWEET_SPOT_END_COLOR_TOLERANCE, 0.08, 0.80,
            ),
            "max_gap_seconds": num(
                "max_gap", cls._SEQUENCE_DEFAULT_MAX_GAP_SECONDS, 0.4, 5.0
            ),
            "hold_repeat": bool(obj.get("hold_repeat", False)),
        }
        return template

    @classmethod
    def _encode_points_sequence_sample(
        cls,
        start_vector: np.ndarray,
        end_vector: np.ndarray,
        max_gap_seconds: float,
        start_tolerance: float,
        end_tolerance: float,
        start_color_signature: np.ndarray | None,
        end_color_signature: np.ndarray | None,
        start_preview_points: np.ndarray | None,
        end_preview_points: np.ndarray | None,
        start_color_tolerance: float,
        end_color_tolerance: float,
        mid_vector: np.ndarray | None = None,
        mid_color_signature: np.ndarray | None = None,
        mid_preview_points: np.ndarray | None = None,
        hold_repeat: bool = False,
    ) -> str:
        payload = {
            "start": np.round(start_vector.reshape(-1), 5).tolist(),
            "end": np.round(end_vector.reshape(-1), 5).tolist(),
            "max_gap": float(np.clip(max_gap_seconds, 0.4, 5.0)),
            "start_tolerance": float(np.clip(start_tolerance, 0.08, 0.60)),
            "end_tolerance": float(np.clip(end_tolerance, 0.08, 0.60)),
            "start_color_tolerance": float(np.clip(start_color_tolerance, 0.08, 0.80)),
            "end_color_tolerance": float(np.clip(end_color_tolerance, 0.08, 0.80)),
            "hold_repeat": bool(hold_repeat),
        }
        if isinstance(mid_vector, np.ndarray) and mid_vector.shape == (21, 3):
            payload["mid"] = np.round(mid_vector.reshape(-1), 5).tolist()
        if isinstance(mid_color_signature, np.ndarray) and mid_color_signature.shape == (6, 3):
            payload["mid_color"] = np.round(mid_color_signature.reshape(-1), 5).tolist()
        if isinstance(mid_preview_points, np.ndarray) and mid_preview_points.shape == (21, 2):
            payload["mid_preview_points"] = np.round(mid_preview_points.reshape(-1), 5).tolist()
        if isinstance(start_color_signature, np.ndarray) and start_color_signature.shape == (6, 3):
            payload["start_color"] = np.round(start_color_signature.reshape(-1), 5).tolist()
        if isinstance(end_color_signature, np.ndarray) and end_color_signature.shape == (6, 3):
            payload["end_color"] = np.round(end_color_signature.reshape(-1), 5).tolist()
        if isinstance(start_preview_points, np.ndarray) and start_preview_points.shape == (21, 2):
            payload["start_preview_points"] = np.round(start_preview_points.reshape(-1), 5).tolist()
        if isinstance(end_preview_points, np.ndarray) and end_preview_points.shape == (21, 2):
            payload["end_preview_points"] = np.round(end_preview_points.reshape(-1), 5).tolist()
        return f"{cls._SEQUENCE_POINTS_PREFIX}:{json.dumps(payload, separators=(',', ':'))}"

    @staticmethod
    def _build_sequence_pose_vector(hand_landmarks) -> np.ndarray:
        wrist_idx = 0
        middle_mcp_idx = 9
        vec = np.array(
            [[float(p.x), float(p.y), float(p.z)] for p in hand_landmarks.landmark],
            dtype=np.float32,
        )
        wrist = vec[wrist_idx]
        scale = float(np.linalg.norm(vec[middle_mcp_idx, :2] - wrist[:2]))
        if scale < 1e-4:
            spread = np.linalg.norm(vec[:, :2] - wrist[:2], axis=1)
            scale = float(spread.max(initial=1e-4))

        normalized = (vec - wrist) / max(scale, 1e-4)
        normalized[:, 2] *= motion_matcher.DEPTH_WEIGHT
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _sequence_pose_distance(current: np.ndarray, target: np.ndarray) -> float:
        if current.shape != target.shape:
            return float("inf")
        return float(np.linalg.norm(current - target, axis=1).mean())

    @staticmethod
    def _sequence_color_distance(current: np.ndarray, target: np.ndarray) -> float:
        if current.shape != target.shape:
            return float("inf")
        return float(np.linalg.norm(current - target, axis=1).mean())

    @staticmethod
    def _build_sequence_color_signature(frame_bgr, hand_landmarks) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        indices = (0, 4, 8, 12, 16, 20)
        signature = np.zeros((len(indices), 3), dtype=np.float32)

        for i, idx in enumerate(indices):
            p = hand_landmarks.landmark[idx]
            cx = int(np.clip(p.x * w, 0, max(w - 1, 0)))
            cy = int(np.clip(p.y * h, 0, max(h - 1, 0)))

            r = 3
            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)
            patch = hsv[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            median_hsv = np.median(patch.reshape(-1, 3), axis=0)
            signature[i, 0] = float(median_hsv[0] / 180.0)
            signature[i, 1] = float(median_hsv[1] / 255.0)
            signature[i, 2] = float(median_hsv[2] / 255.0)

        return signature

    def _load_sequence_templates(self) -> None:
        templates: dict[str, dict[str, object]] = {}
        for item in self._repository.list_gesture_templates():
            points_parsed = self._parse_sequence_points_sample(item.sample_path)
            if points_parsed is not None:
                templates[item.name] = points_parsed
                continue

            parsed = self._parse_sequence_template_sample(item.sample_path)
            if parsed is None:
                continue

            start_pose, end_pose, max_gap = parsed
            min_confidence = float(np.clip(item.threshold, 0.5, 0.95))
            templates[item.name] = {
                "mode": "labels",
                "start_pose": start_pose,
                "end_pose": end_pose,
                "min_confidence": min_confidence,
                "max_gap_seconds": max_gap,
            }

        self._sequence_templates = templates
        self._sequence_armed_until = {
            name: deadline
            for name, deadline in self._sequence_armed_until.items()
            if name in self._sequence_templates
        }
        self._sequence_state = {
            name: v for name, v in self._sequence_state.items() if name in self._sequence_templates
        }

    def _reset_sequence_runtime(self) -> None:
        self._sequence_armed_until.clear()
        self._sequence_state.clear()
        self._sequence_pose_candidate_name = ""
        self._sequence_pose_candidate_count = 0
        self._sequence_last_stable_pose = ""

    def _on_recognition_mode_changed(self, _index: int) -> None:
        pass  # mode selector removed; always hybrid

    def _capture_shortcut_dialog(self, default_shortcut: str = "") -> str | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Assign Shortcut")
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setSpacing(8)

        hint = QLabel(
            "Press the shortcut keys directly below, or type it manually."
        )
        hint.setWordWrap(True)

        capture = QKeySequenceEdit()
        capture.setMaximumSequenceLength(4)
        shortcut_input = QLineEdit()
        shortcut_input.setPlaceholderText("ctrl+shift+n")

        normalized_default = self._normalize_shortcut(default_shortcut.strip())
        if normalized_default:
            capture.setKeySequence(QKeySequence(normalized_default))
            shortcut_input.setText(normalized_default)

        def _sync_from_capture(sequence: QKeySequence) -> None:
            portable = sequence.toString(QKeySequence.SequenceFormat.PortableText)
            if portable:
                shortcut_input.setText(self._normalize_shortcut(portable))

        capture.keySequenceChanged.connect(_sync_from_capture)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        dialog_layout.addWidget(hint)
        dialog_layout.addWidget(QLabel("Press shortcut"))
        dialog_layout.addWidget(capture)
        dialog_layout.addWidget(QLabel("Or type shortcut"))
        dialog_layout.addWidget(shortcut_input)
        dialog_layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        payload = shortcut_input.text().strip()
        if not payload:
            portable = capture.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            payload = portable.strip()

        return self._normalize_shortcut(payload)

    def _on_guide_toggled(self, checked: bool) -> None:
        self._guide_overlay_enabled = checked
        self.guide_button.setText("Hide Gesture Guide" if checked else "Show Gesture Guide")

    def _on_camera_feed_toggled(self, checked: bool) -> None:
        self._camera_feed_enabled = checked
        self.camera_feed_button.setText(
            "Camera Feed: On" if checked else "Camera Feed: Off"
        )

    def _crop_to_hand(self, frame, hand_landmarks) -> np.ndarray:
        h, w = frame.shape[:2]
        points = hand_landmarks.landmark
        xs = [float(p[0]) if isinstance(p, (tuple, list)) else float(p.x) for p in points]
        ys = [float(p[1]) if isinstance(p, (tuple, list)) else float(p.y) for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        span_x = (x_max - x_min) or 0.15
        span_y = (y_max - y_min) or 0.15
        pad_x, pad_y = span_x * 0.45, span_y * 0.45
        x1 = max(0, int((x_min - pad_x) * w))
        x2 = min(w, int((x_max + pad_x) * w))
        y1 = max(0, int((y_min - pad_y) * h))
        y2 = min(h, int((y_max + pad_y) * h))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return frame
        return frame[y1:y2, x1:x2]

    def _draw_gesture_guide(self, frame) -> None:
        h, w = frame.shape[:2]
        cell = 90
        cols = 4
        guides = [
            ("move_up",          self._guide_arrow(0, -1)),
            ("move_down",        self._guide_arrow(0, +1)),
            ("move_left",        self._guide_arrow(-1, 0)),
            ("move_right",       self._guide_arrow(+1, 0)),
            ("index_up",         self._guide_finger_swipe(0, -1)),
            ("index_down",       self._guide_finger_swipe(0, +1)),
            ("index_left",       self._guide_finger_swipe(-1, 0)),
            ("index_right",      self._guide_finger_swipe(+1, 0)),
            ("pinch_in",         self._guide_pinch(closing=True)),
            ("pinch_out",        self._guide_pinch(closing=False)),
            ("finger_circle_cw", self._guide_circle(cw=True)),
            ("finger_circle_ccw",self._guide_circle(cw=False)),
        ]
        rows = (len(guides) + cols - 1) // cols
        panel_w = cols * cell
        panel_h = rows * cell
        ox = max(0, w - panel_w - 8)
        oy = max(0, h - panel_h - 8)

        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + panel_w, oy + panel_h), (20, 26, 38), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (ox, oy), (ox + panel_w, oy + panel_h), (60, 80, 120), 1)

        mapped = set(self._shortcut_mappings.keys())

        for i, (name, draw_fn) in enumerate(guides):
            col = i % cols
            row = i // cols
            cx = ox + col * cell + cell // 2
            cy = oy + row * cell + cell // 2
            has_mapping = name in mapped
            border_color = (60, 200, 100) if has_mapping else (60, 80, 120)
            cv2.rectangle(frame,
                (ox + col * cell + 2, oy + row * cell + 2),
                (ox + col * cell + cell - 2, oy + row * cell + cell - 2),
                border_color, 1)
            draw_fn(frame, cx, cy - 10, cell - 16)
            cv2.putText(frame, name.replace("_", " "),
                (ox + col * cell + 4, oy + row * cell + cell - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (180, 220, 255) if has_mapping else (120, 140, 170), 1, cv2.LINE_AA)

    @staticmethod
    def _guide_arrow(dx: int, dy: int):
        def draw(frame, cx, cy, size):
            s = size // 2 - 4
            ex, ey = cx + dx * s, cy + dy * s
            sx, sy = cx - dx * s, cy - dy * s
            cv2.arrowedLine(frame, (sx, sy), (ex, ey), (100, 200, 255), 2,
                            cv2.LINE_AA, tipLength=0.35)
        return draw

    @staticmethod
    def _guide_finger_swipe(dx: int, dy: int):
        def draw(frame, cx, cy, size):
            s = size // 2 - 6
            # draw small fingertip circle + arrow
            cv2.circle(frame, (cx - dx * s, cy - dy * s), 4, (255, 200, 80), -1, cv2.LINE_AA)
            cv2.arrowedLine(frame,
                (cx - dx * s, cy - dy * s),
                (cx + dx * s, cy + dy * s),
                (255, 200, 80), 2, cv2.LINE_AA, tipLength=0.38)
        return draw

    @staticmethod
    def _guide_pinch(closing: bool):
        def draw(frame, cx, cy, size):
            r = size // 2 - 6
            if closing:
                cv2.circle(frame, (cx - r // 2, cy - r // 2), 5, (255, 160, 60), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx + r // 2, cy + r // 2), 5, (255, 160, 60), -1, cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx - r // 2, cy - r // 2), (cx - 3, cy - 3),
                                (255, 160, 60), 1, cv2.LINE_AA, tipLength=0.4)
                cv2.arrowedLine(frame, (cx + r // 2, cy + r // 2), (cx + 3, cy + 3),
                                (255, 160, 60), 1, cv2.LINE_AA, tipLength=0.4)
            else:
                cv2.circle(frame, (cx - 4, cy - 4), 5, (100, 220, 160), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx + 4, cy + 4), 5, (100, 220, 160), -1, cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx - 3, cy - 3), (cx - r // 2, cy - r // 2),
                                (100, 220, 160), 1, cv2.LINE_AA, tipLength=0.4)
                cv2.arrowedLine(frame, (cx + 3, cy + 3), (cx + r // 2, cy + r // 2),
                                (100, 220, 160), 1, cv2.LINE_AA, tipLength=0.4)
        return draw

    @staticmethod
    def _guide_circle(cw: bool):
        def draw(frame, cx, cy, size):
            r = size // 2 - 8
            # draw arc using polyline approximation
            angles = np.linspace(0, 2 * np.pi * 0.82, 24)
            if not cw:
                angles = angles[::-1]
            pts = np.array(
                [[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a in angles],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts], False, (180, 100, 255), 2, cv2.LINE_AA)
            tip = pts[-1]
            prev = pts[-2]
            tang = tip - prev
            tang_end = tip + (tang * 0.5).astype(np.int32)
            cv2.arrowedLine(frame, tuple(prev.tolist()), tuple(tang_end.tolist()),
                            (180, 100, 255), 2, cv2.LINE_AA, tipLength=0.6)
        return draw

    def _draw_frame(self, bgr_frame) -> None:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    _LOG_MAX_ROWS = 500
    _LOG_STYLES: dict[str, tuple[str, str]] = {
        "info":    ("●", "#b0bcd4"),
        "success": ("✔", "#5ad08a"),
        "trigger": ("⚡", "#4fc8ef"),
        "warn":    ("⚠", "#e8b84b"),
        "error":   ("✗", "#e05060"),
        "system":  ("◈", "#9a9fe0"),
    }

    def _log(self, message: str, level: str = "info") -> None:
        icon, hex_color = self._LOG_STYLES.get(level, self._LOG_STYLES["info"])
        ts = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"{ts}  {icon}  {message}")
        item.setForeground(QColor(hex_color))
        self.log_list.addItem(item)

        while self.log_list.count() > self._LOG_MAX_ROWS:
            self.log_list.takeItem(0)

        # scrollToItem beats scrollToBottom here: wrapped rows have variable height.
        self.log_list.scrollToItem(
            item, QAbstractItemView.ScrollHint.PositionAtBottom
        )

    def _animate_button_feedback(self, button: QPushButton) -> None:
        effect = button.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(button)
            effect.setOpacity(1.0)
            button.setGraphicsEffect(effect)

        from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSequentialAnimationGroup

        fade_out = QPropertyAnimation(effect, b"opacity", self)
        fade_out.setDuration(90)
        fade_out.setStartValue(float(effect.opacity()))
        fade_out.setEndValue(0.45)
        fade_out.setEasingCurve(QEasingCurve.Type.OutCubic)

        fade_in = QPropertyAnimation(effect, b"opacity", self)
        fade_in.setDuration(130)
        fade_in.setStartValue(0.45)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.InOutCubic)

        sequence = QSequentialAnimationGroup(self)
        sequence.addAnimation(fade_out)
        sequence.addAnimation(fade_in)

        def _cleanup() -> None:
            effect.setOpacity(1.0)
            if sequence in self._feedback_animations:
                self._feedback_animations.remove(sequence)

        sequence.finished.connect(_cleanup)
        self._feedback_animations.append(sequence)
        sequence.start()

    def _show_capture_toast(self, message: str) -> None:
        gesture_text = message.strip()
        action_text = ""
        if "->" in message:
            left, right = message.split("->", 1)
            gesture_text = left.strip()
            action_text = right.strip()

        if self._capture_toast_close_timer.isActive():
            self._capture_toast_close_timer.stop()

        if self._capture_toast is not None:
            self._capture_toast.close()
            self._capture_toast.deleteLater()
            self._capture_toast = None

        toast = QFrame(
            None,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        toast.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        toast.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        toast.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        shell = QFrame(toast)
        shell.setStyleSheet(
            "QFrame {"
            " background: #09111b;"
            " border: 1px solid #1f3551;"
            " border-radius: 14px;"
            "}"
            "QLabel#title {"
            " color: #eef6ff;"
            " font-size: 11px;"
            " font-weight: 700;"
            "}"
            "QLabel#detail {"
            " color: #f5f9ff;"
            " font-size: 14px;"
            " font-weight: 700;"
            "}"
            "QLabel#action {"
            " color: #dff7ff;"
            " font-size: 12px;"
            " font-weight: 600;"
            "}"
        )

        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(14, 10, 14, 10)
        shell_layout.setSpacing(3)

        title_label = QLabel("Gesture Captured")
        title_label.setObjectName("title")
        detail_label = QLabel(gesture_text)
        detail_label.setObjectName("detail")
        detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shell_layout.addWidget(title_label)
        shell_layout.addWidget(detail_label)

        if action_text:
            action_label = QLabel(action_text)
            action_label.setObjectName("action")
            action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            shell_layout.addWidget(action_label)

        root_layout = QVBoxLayout(toast)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(shell)
        toast.adjustSize()

        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        if screen is not None:
            rect = screen.availableGeometry()
            x = rect.x() + (rect.width() - toast.width()) // 2
            y = rect.y() + (rect.height() - toast.height()) // 2
            toast.move(x, y)

        toast.show()
        self._capture_toast = toast
        self._capture_toast_close_timer.start(1800)

    def _close_capture_toast(self) -> None:
        if self._capture_toast is not None:
            self._capture_toast.close()
            self._capture_toast.deleteLater()
            self._capture_toast = None

    def clear_session_log(self) -> None:
        self.log_list.clear()

    def _refresh_gesture_options(self) -> None:
        current = self.mapping_gesture_combo.currentText().strip()
        template_items = self._repository.list_gesture_templates()
        custom = [item.name for item in template_items]

        names = sorted(set(custom), key=str.lower)
        self.mapping_gesture_combo.clear()
        self.mapping_gesture_combo.addItems(names)

        if current and current in names:
            self.mapping_gesture_combo.setCurrentText(current)

        self._load_sequence_templates()

    def _reload_mappings(self) -> None:
        self._shortcut_mappings.clear()
        self.mapping_list.clear()

        mappings = self._repository.list_gesture_mappings()
        for mapping in mappings:
            if mapping.action_type != "shortcut":
                continue
            self._shortcut_mappings[mapping.gesture_name] = mapping.action_payload
            self.mapping_list.addItem(f"{mapping.gesture_name} -> {mapping.action_payload}")

        self._on_mapping_selection_changed(self.mapping_gesture_combo.currentText())

    def _maybe_trigger_mapping(self, gesture_name: str, confidence: float) -> None:
        if gesture_name == "unknown" or confidence < 0.65:
            return

        is_saved_motion = gesture_name in self._sequence_templates
        shortcut = self._shortcut_mappings.get(gesture_name)
        now = time.monotonic()

        if not shortcut:
            # Log once per cooldown window so the user knows detection IS working.
            last = self._last_trigger_at.get(f"__no_sc_{gesture_name}", 0.0)
            if now - last >= self._trigger_cooldown_seconds:
                self._last_trigger_at[f"__no_sc_{gesture_name}"] = now
                self._log(
                    f"'{gesture_name}' detected ({confidence:.0%}) — no shortcut assigned",
                    "warn",
                )
            return

        # Prevent opposite actions from firing during brief transition jitter.
        if (
            self._last_triggered_gesture
            and gesture_name != self._last_triggered_gesture
            and (now - self._last_triggered_at) < self._gesture_switch_guard_seconds
        ):
            return

        last = self._last_trigger_at.get(gesture_name, 0.0)
        if now - last < self._trigger_cooldown_seconds:
            return

        keys = self._parse_shortcut(shortcut)
        if not keys:
            self._log(f"Bad shortcut for '{gesture_name}': {shortcut}", "error")
            return

        self._last_trigger_at[gesture_name] = now
        self._last_triggered_gesture = gesture_name
        self._last_triggered_at = now
        self._execute_shortcut(shortcut)
        conf_str = f" ({confidence:.0%})" if is_saved_motion else ""
        self._log(f"{gesture_name}{conf_str}  →  {'+'.join(keys)}", "trigger")
        if self._is_window_obscured():
            toast_conf = f" ({confidence:.0%})" if is_saved_motion else ""
            self._show_capture_toast(f"{gesture_name}{toast_conf} -> {'+'.join(keys)}")
        if self._dispatcher.last_error:
            self._log(f"Shortcut warning: {self._dispatcher.last_error}", "warn")

    def _stable_gesture(self, gesture_name: str, confidence: float) -> str:
        if gesture_name == "unknown" or confidence < 0.65:
            self._reset_gesture_stability()
            return "unknown"

        if gesture_name == self._gesture_candidate_name:
            self._gesture_candidate_count += 1
        else:
            self._gesture_candidate_name = gesture_name
            self._gesture_candidate_count = 1

        if self._gesture_candidate_count >= self._gesture_stability_frames:
            self._last_stable_gesture = gesture_name
            return gesture_name

        # During transition, do not reuse the previous stable gesture for triggering.
        return "unknown"

    def _reset_gesture_stability(self) -> None:
        self._gesture_candidate_name = ""
        self._gesture_candidate_count = 0
        self._last_stable_gesture = "unknown"

    def _execute_shortcut(self, payload: str) -> None:
        keys = self._parse_shortcut(payload)
        if not keys:
            return
        self._dispatcher.trigger_shortcut(keys)

    @staticmethod
    def _parse_shortcut(payload: str) -> list[str]:
        return [part.strip() for part in payload.split("+") if part.strip()]

    def _on_mapping_selection_changed(self, gesture_name: str) -> None:
        selected_name = gesture_name.strip()
        shortcut = self._shortcut_mappings.get(selected_name, "")

        system_index = self.system_action_combo.findData(shortcut) if shortcut else 0
        self.system_action_combo.blockSignals(True)
        self.system_action_combo.setCurrentIndex(max(system_index, 0))
        self.system_action_combo.blockSignals(False)

        # Only show it in the key capture widget if it is not a system action.
        self.mapping_shortcut_capture.setKeySequence(
            QKeySequence() if system_index > 0 else QKeySequence(shortcut)
        )
        self._load_sequence_snapshots_for_selection(selected_name)

    def _on_system_action_changed(self, index: int) -> None:
        if index > 0:
            self.mapping_shortcut_capture.clear()

    def _clear_sequence_snapshot_views(self) -> None:
        self._captured_sequence_start_vector = None
        self._captured_sequence_end_vector = None
        self._captured_sequence_start_color_signature = None
        self._captured_sequence_end_color_signature = None
        self._captured_sequence_start_preview_points = None
        self._captured_sequence_mid_preview_points = None
        self._captured_sequence_end_preview_points = None
        self._captured_sequence_start_pose = ""
        self._captured_sequence_end_pose = ""
        self._sequence_start_dirty = False
        self._sequence_end_dirty = False
        self.sequence_start_label.setText("Start: -")
        self.sequence_mid_label.setText("Mid: -")
        self.sequence_end_label.setText("End: -")
        self.sequence_start_preview_label.clear()
        self.sequence_mid_preview_label.clear()
        self.sequence_end_preview_label.clear()
        self.sequence_start_preview_label.setText("No start snapshot")
        self.sequence_mid_preview_label.setText("No mid snapshot")
        self.sequence_end_preview_label.setText("No end snapshot")

    def _load_sequence_snapshots_for_selection(self, gesture_name: str, force: bool = False) -> None:
        if not gesture_name:
            self._clear_sequence_snapshot_views()
            self._sequence_selected_name = ""
            return

        if (
            not force
            and gesture_name == self._sequence_selected_name
            and (self._sequence_start_dirty or self._sequence_end_dirty)
        ):
            # Keep in-progress edits for the currently selected sequence.
            return

        template = self._sequence_templates.get(gesture_name)
        if template is None or str(template.get("mode", "")) != "points":
            self._clear_sequence_snapshot_views()
            self._sequence_selected_name = gesture_name
            return

        start_vector = template.get("start_vector")
        end_vector = template.get("end_vector")
        if not isinstance(start_vector, np.ndarray) or not isinstance(end_vector, np.ndarray):
            self._clear_sequence_snapshot_views()
            return

        self._captured_sequence_start_vector = start_vector.copy()
        self._captured_sequence_end_vector = end_vector.copy()
        start_color = template.get("start_color_signature")
        end_color = template.get("end_color_signature")
        mid_vector = template.get("mid_vector")
        mid_color = template.get("mid_color_signature")
        mid_preview_points = template.get("mid_preview_points")
        start_preview_points = template.get("start_preview_points")
        end_preview_points = template.get("end_preview_points")
        self._captured_sequence_mid_vector = (
            mid_vector.copy() if isinstance(mid_vector, np.ndarray) else None
        )
        self._captured_sequence_start_color_signature = (
            start_color.copy() if isinstance(start_color, np.ndarray) else None
        )
        self._captured_sequence_mid_color_signature = (
            mid_color.copy() if isinstance(mid_color, np.ndarray) else None
        )
        self._captured_sequence_end_color_signature = (
            end_color.copy() if isinstance(end_color, np.ndarray) else None
        )
        self._captured_sequence_start_preview_points = (
            start_preview_points.copy() if isinstance(start_preview_points, np.ndarray) else None
        )
        self._captured_sequence_mid_preview_points = (
            mid_preview_points.copy() if isinstance(mid_preview_points, np.ndarray) else None
        )
        self._captured_sequence_end_preview_points = (
            end_preview_points.copy() if isinstance(end_preview_points, np.ndarray) else None
        )
        self._captured_sequence_start_pose = "saved_snapshot"
        self._captured_sequence_end_pose = "saved_snapshot"
        self._sequence_selected_name = gesture_name
        self._sequence_start_dirty = False
        self._sequence_end_dirty = False
        start_tolerance = float(
            template.get("start_tolerance", self._SEQUENCE_DEFAULT_TOLERANCE)
        )
        end_tolerance = float(
            template.get("end_tolerance", self._SEQUENCE_DEFAULT_TOLERANCE)
        )
        value = int(np.clip((start_tolerance + end_tolerance) * 50, 8, 60))
        blocked = self.sequence_tolerance_slider.blockSignals(True)
        self.sequence_tolerance_slider.setValue(value)
        self.sequence_tolerance_slider.blockSignals(blocked)
        self._on_sequence_tolerance_changed(value)
        self.sequence_name_input.setText(gesture_name)
        self.sequence_hold_repeat_checkbox.setChecked(bool(template.get("hold_repeat", False)))
        self.sequence_start_label.setText("Start: saved snapshot")
        self.sequence_end_label.setText("End: saved snapshot")

        if isinstance(mid_vector, np.ndarray):
            self.sequence_mid_label.setText("Mid: saved snapshot")
            self.sequence_mid_preview_label.setPixmap(
                self._make_sequence_snapshot_pixmap(
                    mid_vector,
                    mid_color if isinstance(mid_color, np.ndarray) else None,
                    mid_preview_points if isinstance(mid_preview_points, np.ndarray) else None,
                )
            )
        else:
            self.sequence_mid_label.setText("Mid: —")
            self.sequence_mid_preview_label.clear()
            self.sequence_mid_preview_label.setText("No mid snapshot")

        self.sequence_start_preview_label.setPixmap(
            self._make_sequence_snapshot_pixmap(
                self._captured_sequence_start_vector,
                self._captured_sequence_start_color_signature,
                self._captured_sequence_start_preview_points,
            )
        )
        self.sequence_end_preview_label.setPixmap(
            self._make_sequence_snapshot_pixmap(
                self._captured_sequence_end_vector,
                self._captured_sequence_end_color_signature,
                self._captured_sequence_end_preview_points,
            )
        )

    @staticmethod
    def _normalize_shortcut(value: str) -> str:
        aliases = {
            "control": "ctrl",
            "cmd": "cmd",
            "command": "cmd",
            "option": "alt",
            "return": "enter",
            "escape": "esc",
        }
        parts = [part.strip().lower() for part in value.split("+") if part.strip()]
        normalized = [aliases.get(part, part) for part in parts]
        return "+".join(normalized)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        if self._capture_toast_close_timer.isActive():
            self._capture_toast_close_timer.stop()
        if self._capture_toast is not None:
            self._capture_toast.close()
            self._capture_toast.deleteLater()
            self._capture_toast = None
        self._stream.close()
        self._tracker.close()
        super().closeEvent(event)
