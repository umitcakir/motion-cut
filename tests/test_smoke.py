from __future__ import annotations

import unittest

import numpy as np

from motion_cut.gestures.rule_engine import GesturePrediction
from motion_cut.ui.window import MainWindow


def _make_pose(variant: str) -> np.ndarray:
    pose = np.zeros((21, 3), dtype=np.float32)
    for idx in range(21):
        pose[idx, 0] = idx / 20.0
        pose[idx, 1] = -((idx % 5) * 0.08)

    pose[0, :2] = (0.0, 0.0)
    pose[9, :2] = (1.0, 0.0)

    shift = {"start": 0.0, "mid": -0.5, "end": -1.0}.get(variant, 0.0)
    for idx in (8, 12, 16, 20):
        pose[idx, 1] += shift

    return pose


def _make_mid_pose() -> np.ndarray:
    return MainWindow._build_sequence_pose_vector(_make_hand_landmarks(_make_pose("mid")))


def _make_hand_landmarks(pose: np.ndarray):
    landmark = [
        type("Point", (), {"x": float(x), "y": float(y), "z": float(z)})()
        for x, y, z in pose.tolist()
    ]
    return type("HandLandmarks", (), {"landmark": landmark})()


class SequenceGestureMatcherTests(unittest.TestCase):
    def _make_window(self) -> MainWindow:
        window = MainWindow.__new__(MainWindow)
        window._last_tick_at = 1.0
        window._sequence_templates = {
            "custom_swipe": {
                "mode": "points",
                "start_vector": _make_pose("start"),
                "end_vector": _make_pose("end"),
                "start_tolerance": 0.12,
                "end_tolerance": 0.12,
                "start_color_signature": np.ones((6, 3), dtype=np.float32),
                "end_color_signature": np.full((6, 3), 0.9, dtype=np.float32),
                "start_color_tolerance": 0.08,
                "end_color_tolerance": 0.08,
                "max_gap_seconds": 2.0,
            }
        }
        window._sequence_armed_until = {}
        window._sequence_state = {}
        window._sequence_pose_candidate_name = ""
        window._sequence_pose_candidate_count = 0
        window._sequence_last_stable_pose = ""
        window._last_sequence_color_signature = np.zeros((6, 3), dtype=np.float32)
        return window

    def test_smoke(self) -> None:
        self.assertTrue(True)

    def test_saved_sequence_points_match_even_with_color_mismatch(self) -> None:
        window = self._make_window()
        unknown = GesturePrediction(name="unknown", confidence=0.0)

        start_hand = _make_hand_landmarks(_make_pose("start"))
        end_hand = _make_hand_landmarks(_make_pose("end"))

        def tick(hand):
            window._last_tick_at += 0.02
            return window._classify_sequence_gesture(unknown, hand)

        # hold start pose to arm
        self.assertIsNone(tick(start_hand))
        self.assertIsNone(tick(start_hand))

        # transit window must elapse before end can register
        for _ in range(8):
            self.assertIsNone(tick(start_hand))

        self.assertIsNone(tick(end_hand))
        result = tick(end_hand)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "custom_swipe")
        self.assertGreaterEqual(result.confidence, 0.65)

    def test_holding_single_pose_does_not_fire(self) -> None:
        window = self._make_window()
        unknown = GesturePrediction(name="unknown", confidence=0.0)
        start_hand = _make_hand_landmarks(_make_pose("start"))

        for _ in range(40):
            window._last_tick_at += 0.02
            self.assertIsNone(window._classify_sequence_gesture(unknown, start_hand))

    def test_three_point_requires_middle(self) -> None:
        window = self._make_window()
        template = window._sequence_templates["custom_swipe"]
        template["mid_vector"] = _make_mid_pose()
        template["mid_tolerance"] = 0.12

        unknown = GesturePrediction(name="unknown", confidence=0.0)
        start_hand = _make_hand_landmarks(_make_pose("start"))
        mid_hand = _make_hand_landmarks(_make_pose("mid"))
        end_hand = _make_hand_landmarks(_make_pose("end"))

        def tick(hand):
            window._last_tick_at += 0.02
            return window._classify_sequence_gesture(unknown, hand)

        # start -> end without the middle must not fire
        for _ in range(4):
            self.assertIsNone(tick(start_hand))
        for _ in range(4):
            self.assertIsNone(tick(end_hand))

        # start -> middle -> end fires
        for _ in range(4):
            tick(start_hand)
        tick(mid_hand)
        self.assertIsNone(tick(end_hand))
        self.assertIsNotNone(tick(end_hand))


if __name__ == "__main__":
    unittest.main()
