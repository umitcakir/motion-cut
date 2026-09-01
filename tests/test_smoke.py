from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from unittest import mock

import numpy as np

from motion_cut.actions.dispatcher import ActionDispatcher
from motion_cut.config import AppSettings, AppSettingsStore
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
    def test_play_pause_targets_the_player_that_is_playing(self) -> None:
        dispatcher = ActionDispatcher.__new__(ActionDispatcher)
        dispatcher._playerctl = "/usr/bin/playerctl"
        dispatcher._session_type = "wayland"
        dispatcher._ydotool = None
        dispatcher._last_error = ""

        def run(command, **_kwargs):
            if command[-1] == "-l":
                return Mock(returncode=0, stdout="youtube\nspotify\n", stderr="")
            if command[-1] == "status":
                player = command[command.index("-p") + 1]
                status = "Paused\n" if player == "youtube" else "Playing\n"
                return Mock(returncode=0, stdout=status, stderr="")
            return Mock(returncode=0, stdout="", stderr="")

        with mock.patch(
            "motion_cut.actions.dispatcher.subprocess.run", side_effect=run
        ) as run_mock:
            dispatcher.trigger_shortcut(["playpause"])

        self.assertEqual(
            run_mock.call_args_list[-1].args[0],
            ["/usr/bin/playerctl", "-p", "spotify", "play-pause"],
        )

    def test_screenshot_action_uses_the_macos_shortcut(self) -> None:
        dispatcher = ActionDispatcher.__new__(ActionDispatcher)
        dispatcher.trigger_shortcut = Mock()

        with mock.patch("motion_cut.actions.dispatcher.sys.platform", "darwin"):
            dispatcher._trigger_named_action("screenshot")

        dispatcher.trigger_shortcut.assert_called_once_with(["cmd", "shift", "3"])

    def test_lock_screen_uses_the_windows_command(self) -> None:
        dispatcher = ActionDispatcher.__new__(ActionDispatcher)
        dispatcher._last_error = ""
        result = Mock(returncode=0, stderr="")

        with (
            mock.patch("motion_cut.actions.dispatcher.sys.platform", "win32"),
            mock.patch("motion_cut.actions.dispatcher.subprocess.run", return_value=result) as run,
        ):
            dispatcher._trigger_named_action("lockscreen")

        run.assert_called_once_with(
            ["rundll32.exe", "user32.dll,LockWorkStation"],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_app_settings_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            store = AppSettingsStore(Path(directory) / "settings.json")
            expected = AppSettings(
                window_width=1024,
                window_height=640,
                camera_feed_enabled=False,
            )

            store.save(expected)

            self.assertEqual(store.load(), expected)

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

    def test_selected_gesture_label_identifies_the_active_gesture(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.selected_gesture_label = Mock()
        window.gesture_editor_container = Mock()

        window._update_selected_gesture_label("next")
        window._set_gesture_editor_visible(True)
        window.selected_gesture_label.setText.assert_called_once_with("Editing: next")
        window.gesture_editor_container.setVisible.assert_called_once_with(True)

    def test_gesture_editor_hides_without_a_selection(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.gesture_editor_container = Mock()

        window._set_gesture_editor_visible(False)

        window.gesture_editor_container.setVisible.assert_called_once_with(False)

    def test_poll_interval_tracks_the_camera_frame_rate(self) -> None:
        self.assertEqual(MainWindow._poll_interval_ms(30.0), 26)
        self.assertEqual(MainWindow._poll_interval_ms(60.0), 16)
        self.assertEqual(MainWindow._poll_interval_ms(15.0), 53)

    def test_poll_interval_clamps_implausible_frame_rates(self) -> None:
        self.assertEqual(MainWindow._poll_interval_ms(0.0), 80)
        self.assertEqual(MainWindow._poll_interval_ms(10_000.0), 16)

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
