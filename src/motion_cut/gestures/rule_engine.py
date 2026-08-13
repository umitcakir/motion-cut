from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class GesturePrediction:
    name: str
    confidence: float


class RuleGestureEngine:
    """Movement-first gesture recognizer for intentional motion gestures."""

    def __init__(self, history_size: int = 18) -> None:
        self._finger_distance_history: deque[float] = deque(maxlen=history_size)
        self._base_distance: float | None = None

    def classify(self, hand_landmarks) -> GesturePrediction:
        lm = hand_landmarks.landmark

        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y

        two_finger_pose = index_up and middle_up and not ring_up and not pinky_up
        if two_finger_pose:
            return self._detect_scissors_motion(lm)

        self._reset_motion_state()
        return GesturePrediction(name="unknown", confidence=0.3)

    def _detect_scissors_motion(self, lm) -> GesturePrediction:
        index_tip = lm[8]
        middle_tip = lm[12]
        distance = abs(index_tip.x - middle_tip.x)

        self._finger_distance_history.append(distance)
        if self._base_distance is None:
            self._base_distance = distance

        base_delta = distance - self._base_distance
        if abs(base_delta) > 0.08:
            self._base_distance = (self._base_distance * 0.7) + (distance * 0.3)

        if len(self._finger_distance_history) < 10:
            return GesturePrediction(name="unknown", confidence=0.3)

        values = list(self._finger_distance_history)
        deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        positive_flips = 0
        negative_flips = 0
        prev_sign = 0
        for delta in deltas:
            if abs(delta) < 0.0025:
                continue

            sign = 1 if delta > 0 else -1
            if prev_sign == 0:
                prev_sign = sign
                continue

            if sign != prev_sign:
                if sign > 0:
                    positive_flips += 1
                else:
                    negative_flips += 1
                prev_sign = sign

        total_flips = positive_flips + negative_flips
        distance_range = max(values) - min(values)

        if total_flips >= 4 and distance_range > 0.05:
            return GesturePrediction(name="scissors_motion", confidence=0.72)

        return GesturePrediction(name="unknown", confidence=0.35)

    def _reset_motion_state(self) -> None:
        self._finger_distance_history.clear()
        self._base_distance = None
