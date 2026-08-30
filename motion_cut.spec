# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import mediapipe as _mp
_MP_DIR = Path(_mp.__file__).parent

mediapipe_datas   = collect_data_files("mediapipe")
mediapipe_bins    = collect_dynamic_libs("mediapipe")
cv2_datas         = collect_data_files("cv2")

a = Analysis(
    ["src/motion_cut/main.py"],
    pathex=["src"],
    binaries=mediapipe_bins,
    datas=[
        ("data/models", "data/models"),
        ("data/seed_gestures.db", "data"),
        ("assets/motion-cut.png", "assets"),
        # Keep mediapipe.tasks.c as a real on-disk package so importlib.resources works
        (str(_MP_DIR / "tasks" / "c" / "__init__.py"), "mediapipe/tasks/c"),
        *mediapipe_datas,
        *cv2_datas,
    ],
    hiddenimports=[
        "motion_cut",
        "motion_cut.ui.app",
        "motion_cut.ui.window",
        "motion_cut.ui.wizard",
        "motion_cut.gestures.motion_matcher",
        "motion_cut.gestures.rule_engine",
        "motion_cut.vision.capture",
        "motion_cut.vision.hand_tracker",
        "motion_cut.vision.hand_tracker_tasks",
        "motion_cut.storage.models",
        "motion_cut.storage.repository",
        "motion_cut.actions.dispatcher",
        "motion_cut.config",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "pynput.keyboard",
        "pynput.mouse",
        *(["AVFoundation"] if sys.platform == "darwin" else []),
        # mediapipe.tasks imports are inside try/except so PyInstaller misses them
        "mediapipe.tasks",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.core",
        "mediapipe.tasks.python.core.base_options",
        "mediapipe.tasks.python.core.base_options_c",
        "mediapipe.tasks.python.core.mediapipe_c_bindings",
        "mediapipe.tasks.python.core.mediapipe_c_utils",
        "mediapipe.tasks.python.vision",
        "mediapipe.tasks.python.vision.hand_landmarker",
        "mediapipe.tasks.python.vision.core",
        "mediapipe.tasks.python.vision.core.vision_task_running_mode",
        "mediapipe.tasks.python.components.containers.landmark",
        "mediapipe.tasks.python.components.containers.landmark_c",
        "mediapipe.tasks.python.components.containers.landmark_detection_result",
        "mediapipe.tasks.c",
    ],
    hookspath=[],
    runtime_hooks=["hooks/rth_mediapipe_tasks.py"],
    excludes=["tkinter", "matplotlib", "scipy"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="motion-cut",
    debug=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window on Windows/Mac
    icon=None,
    # single-file executable — everything merged into one binary
    bootloader_ignore_signals=False,
    runtime_tmpdir=None,
)

# macOS app bundle wraps the single-file exe
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="Motion Cut.app",
        bundle_identifier="com.motioncut.app",
        info_plist={
            "NSCameraUsageDescription": "Motion Cut needs the camera to detect hand gestures.",
            "NSAppleEventsUsageDescription": "Motion Cut sends keystrokes to control other apps.",
        },
    )
