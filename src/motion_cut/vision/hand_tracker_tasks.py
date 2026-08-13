from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover - depends on installed mediapipe package
    mp = None
    mp_python = None
    mp_vision = None


@dataclass(slots=True)
class TaskTrackingResult:
    landmarks: list[list[tuple[float, float, float]]]


class TasksHandTracker:
    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._model_path = Path(model_path)
        self._is_available = False
        self._error_message = "Tasks hand tracker backend is unavailable"
        self._landmarker = None
        self._last_timestamp_ms = 0

        if mp is None or mp_python is None or mp_vision is None:
            self._error_message = "mediapipe tasks backend is not installed"
            return

        if not self._model_path.exists():
            self._error_message = f"Hand landmark model not found at {self._model_path}"
            return

        base_options = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        try:
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._is_available = True
            self._error_message = ""
        except Exception as exc:  # pragma: no cover - runtime backend specific
            self._error_message = f"Failed to initialize tasks landmarker: {exc}"

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def error_message(self) -> str:
        return self._error_message

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()

    def process(self, bgr_frame) -> TaskTrackingResult:
        if self._landmarker is None:
            return TaskTrackingResult(landmarks=[])

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now_ms = int(time.time() * 1000)
        timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        landmarks = result.hand_landmarks or []
        converted: list[list[tuple[float, float, float]]] = []
        for hand in landmarks:
            points = [(p.x, p.y, p.z) for p in hand]
            converted.append(points)

        return TaskTrackingResult(landmarks=converted)

    def draw_landmarks(self, frame, hands_landmarks: list) -> None:
        if not hands_landmarks:
            return

        height, width = frame.shape[:2]
        for hand in hands_landmarks:
            points = getattr(hand, "landmark", hand)
            if not points:
                continue

            for point in points:
                x = float(point[0]) if isinstance(point, (tuple, list)) else float(point.x)
                y = float(point[1]) if isinstance(point, (tuple, list)) else float(point.y)
                cx = int(np.clip(x, 0.0, 1.0) * width)
                cy = int(np.clip(y, 0.0, 1.0) * height)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
