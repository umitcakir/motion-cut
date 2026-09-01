from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional runtime dependency behavior
    mp = None

from motion_cut.vision.hand_tracker_tasks import TasksHandTracker


def _resolve_asset(relative_path: str) -> str:
    """Resolve a path relative to sys._MEIPASS when frozen, otherwise use as-is."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidate = os.path.join(bundle, relative_path)
        if os.path.isfile(candidate):
            return candidate
    return relative_path


def _prepare_frame_for_hand_detection(bgr_frame: np.ndarray) -> np.ndarray:
    """Improve dim frames for inference without changing the camera preview."""
    luminance = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    if float(luminance.mean()) >= 85.0:
        return bgr_frame

    gamma_table = np.array(
        [((value / 255.0) ** 0.65) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    brightened = cv2.LUT(bgr_frame, gamma_table)
    lab_frame = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_frame)
    enhanced_lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )


@dataclass(slots=True)
class TrackingResult:
    landmarks: list


class HandTracker:
    _HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_path: str = "data/models/hand_landmarker.task",
    ) -> None:
        self._min_detection_confidence = float(min_detection_confidence)
        self._min_tracking_confidence = float(min_tracking_confidence)
        self._model_path = _resolve_asset(model_path)
        self._mp_hands = None
        self._mp_draw = None
        self._hands = None
        self._is_available = False
        self._error_message = "MediaPipe hand tracking backend is unavailable"
        self._backend = "none"
        self._last_landmarks: list = []
        self._missed_frames = 0
        self._max_missed_frames = 6
        self._empty_streak = 0
        # Recover quickly after hand leaves frame so re-entry is picked up sooner.
        self._recover_no_hand_frames = 12
        self._recover_cooldown_seconds = 1.5
        self._last_recover_at = 0.0
        self._recovery_attempts = 0

        if mp is not None and hasattr(mp, "solutions"):
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._is_available = True
            self._error_message = ""
            self._backend = "solutions"
            return

        tasks_tracker = TasksHandTracker(
            model_path=self._model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        if tasks_tracker.is_available:
            self._hands = tasks_tracker
            self._is_available = True
            self._error_message = ""
            self._backend = "tasks"
            return

        self._error_message = tasks_tracker.error_message

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def backend(self) -> str:
        return self._backend

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

    def reset_runtime(self) -> None:
        self._last_landmarks = []
        self._missed_frames = 0
        self._empty_streak = 0
        self._last_recover_at = 0.0

    def _recover_tasks_backend(self) -> None:
        if self._backend != "tasks":
            return

        now = time.monotonic()
        if self._empty_streak < self._recover_no_hand_frames:
            return
        if self._last_recover_at > 0.0 and (now - self._last_recover_at) < self._recover_cooldown_seconds:
            return
        self._last_recover_at = now

        previous = self._hands
        tracker = TasksHandTracker(
            model_path=self._model_path,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        if not tracker.is_available:
            self._error_message = tracker.error_message
            return

        if previous is not None:
            previous.close()

        self._hands = tracker
        self._is_available = True
        self._error_message = ""
        self._empty_streak = 0
        self._recovery_attempts += 1

    def process(self, bgr_frame) -> TrackingResult:
        if self._hands is None:
            return TrackingResult(landmarks=[])

        detection_frame = _prepare_frame_for_hand_detection(bgr_frame)
        if self._backend == "tasks":
            result = self._hands.process(detection_frame)
            converted = [self._TaskHandPoints(points) for points in result.landmarks]
            if converted:
                self._last_landmarks = converted
                self._missed_frames = 0
                self._empty_streak = 0
                return TrackingResult(landmarks=converted)

            self._empty_streak += 1
            if self._last_landmarks and self._missed_frames < self._max_missed_frames:
                self._missed_frames += 1
                return TrackingResult(landmarks=self._last_landmarks)

            self._last_landmarks = []
            self._missed_frames = 0
            self._recover_tasks_backend()
            return TrackingResult(landmarks=[])

        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_frame)
        hand_landmarks = getattr(result, "multi_hand_landmarks", None)
        if hand_landmarks is None:
            hand_landmarks = getattr(result, "landmarks", [])

        if hand_landmarks:
            self._last_landmarks = hand_landmarks
            self._missed_frames = 0
            return TrackingResult(landmarks=hand_landmarks)

        if self._last_landmarks and self._missed_frames < self._max_missed_frames:
            self._missed_frames += 1
            return TrackingResult(landmarks=self._last_landmarks)

        self._last_landmarks = []
        self._missed_frames = 0
        return TrackingResult(landmarks=[])

    # One colour per landmark 0-20: wrist → thumb → index → middle → ring → pinky
    _LANDMARK_COLORS: tuple[tuple[int, int, int], ...] = (
        (255, 255, 255),  # 0  wrist
        (255, 180,  60),  # 1  thumb cmc
        (255, 200,  80),  # 2  thumb mcp
        (255, 220, 100),  # 3  thumb ip
        (255, 240, 120),  # 4  thumb tip
        ( 80, 200, 255),  # 5  index mcp
        (100, 220, 255),  # 6  index pip
        (120, 235, 255),  # 7  index dip
        (140, 255, 255),  # 8  index tip
        (100, 255, 140),  # 9  middle mcp
        (120, 255, 160),  # 10 middle pip
        (140, 255, 180),  # 11 middle dip
        (160, 255, 200),  # 12 middle tip
        (200, 130, 255),  # 13 ring mcp
        (215, 150, 255),  # 14 ring pip
        (225, 170, 255),  # 15 ring dip
        (235, 190, 255),  # 16 ring tip
        (255,  80, 160),  # 17 pinky mcp
        (255, 100, 180),  # 18 pinky pip
        (255, 120, 200),  # 19 pinky dip
        (255, 140, 220),  # 20 pinky tip
    )

    def draw_landmarks(self, frame, hands_landmarks: list) -> None:
        if not hands_landmarks:
            return

        height, width = frame.shape[:2]
        for hand_landmarks in hands_landmarks:
            points = getattr(hand_landmarks, "landmark", hand_landmarks)
            if not points:
                continue

            pixel_points: list[tuple[int, int]] = []
            for point in points:
                x = float(point[0]) if isinstance(point, (tuple, list)) else float(point.x)
                y = float(point[1]) if isinstance(point, (tuple, list)) else float(point.y)
                cx = int(np.clip(x, 0.0, 1.0) * (width - 1))
                cy = int(np.clip(y, 0.0, 1.0) * (height - 1))
                pixel_points.append((cx, cy))

            for start, end in self._HAND_CONNECTIONS:
                if start >= len(pixel_points) or end >= len(pixel_points):
                    continue
                cv2.line(
                    frame,
                    pixel_points[start],
                    pixel_points[end],
                    (80, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            for idx, (cx, cy) in enumerate(pixel_points):
                color = self._LANDMARK_COLORS[idx] if idx < len(self._LANDMARK_COLORS) else (255, 255, 255)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)

    @dataclass(slots=True)
    class _TaskPoint:
        x: float
        y: float
        z: float

    @dataclass(slots=True)
    class _TaskHandPoints:
        landmark: list["HandTracker._TaskPoint"]

        def __init__(self, points: list[tuple[float, float, float]]) -> None:
            self.landmark = [
                HandTracker._TaskPoint(x=x, y=y, z=z) for x, y, z in points
            ]
