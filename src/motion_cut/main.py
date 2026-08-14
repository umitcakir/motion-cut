from __future__ import annotations

import argparse
import os
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2

from motion_cut.config import CameraConfig, RuntimeConfig
from motion_cut.gestures.rule_engine import RuleGestureEngine
from motion_cut.ui.app import launch_app
from motion_cut.vision.capture import CameraStream
from motion_cut.vision.hand_tracker import HandTracker


def _resolve_seed_db() -> str | None:
    """Return path to the bundled seed DB, works both under PyInstaller and in dev."""
    # PyInstaller onedir: files extracted to sys._MEIPASS
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        # dev: look relative to this file's package root
        base = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    candidate = os.path.join(base, "data", "seed_gestures.db")
    return candidate if os.path.isfile(candidate) else None


def _setup_user_db() -> None:
    """Ensure the user has a writable DB, seeding from the bundle on first run."""
    user_data = os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "motion-cut",
    )
    os.makedirs(user_data, exist_ok=True)
    user_db = os.path.join(user_data, "motion_cut.db")

    if not os.path.isfile(user_db):
        seed = _resolve_seed_db()
        if seed:
            shutil.copy2(seed, user_db)

    os.environ["MOTION_CUT_DB"] = user_db


def run_cli() -> None:
    camera_cfg = CameraConfig()
    runtime_cfg = RuntimeConfig()

    tracker = HandTracker(
        min_detection_confidence=runtime_cfg.min_detection_confidence,
        min_tracking_confidence=runtime_cfg.min_tracking_confidence,
    )
    gesture_engine = RuleGestureEngine()

    with CameraStream(camera_cfg) as stream:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            result = tracker.process(frame)
            if result.landmarks:
                gesture = gesture_engine.classify(result.landmarks[0])
                cv2.putText(
                    frame,
                    f"Gesture: {gesture.name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            tracker.draw_landmarks(frame, result.landmarks)
            cv2.imshow("Motion Cut Prototype", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


def main() -> None:
    _setup_user_db()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run OpenCV-only CLI preview instead of desktop UI",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
        return

    raise SystemExit(launch_app())


if __name__ == "__main__":
    main()
