# Motion Cut

Motion Cut is a desktop hand-gesture shortcut tool built with Python, OpenCV, MediaPipe, and PySide6.

The current app lets you define your own gestures by saving a start pose and an end pose from the webcam feed, then map that gesture to a keyboard shortcut.

## Release Status

This project is currently best described as an alpha source release.

- The main workflow works: save custom gestures, map them to shortcuts, and trigger them live.
- The shipped recognizer is user-defined gesture matching only. Predefined gestures are not part of the active runtime flow.
- Distribution is source-first right now. There are no packaged installers or signed desktop bundles yet.
- The current camera path is Linux-first because capture uses a V4L2 backend.

If you publish this on GitHub today, present it as a source-based preview for technical users rather than a polished end-user installer release.

## What It Does Today

- Opens a desktop UI with live camera preview.
- Tracks a single hand with MediaPipe landmarks.
- Lets you capture a custom gesture as a start snapshot plus end snapshot.
- Stores gesture templates and shortcut mappings locally in SQLite.
- Triggers keyboard shortcuts from matched custom gestures.
- Shows runtime feedback in an in-app session log.

## Current Limitations

- Linux-first runtime at the moment. Windows and macOS are project goals, but this repo is not documented as production-ready there yet.
- No bundled installers yet. Users currently run from source.
- Shortcut triggering is the only shipped action path.
- Recognition quality depends on stable framing, lighting, and distinct start/end poses.
- Gesture data is stored locally in `motion_cut.db` in the project folder.

## Quick Start

### Recommended launcher

The launcher creates a virtual environment if needed, installs dependencies, downloads the required MediaPipe task files, and starts the app.

1. Make the launcher executable once:

```sh
chmod +x start_motion_cut.sh
```

2. Start the app:

```sh
./start_motion_cut.sh
```

### Manual run

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
PYTHONPATH=src ./.venv/bin/python -m motion_cut.main
```

Optional CLI preview:

```sh
PYTHONPATH=src ./.venv/bin/python -m motion_cut.main --cli
```

## Requirements

- Python 3.10+
- Webcam
- Linux desktop session for the currently tested path

Wayland note:

- On Wayland, the launcher tries to install and start `ydotool` automatically when possible.
- If `ydotool` is unavailable, shortcut injection falls back to `pynput`, which may be less reliable depending on the desktop environment.

## How To Use

1. Launch the app.
2. Start the camera if it is not already running.
3. Enter a gesture name.
4. Hold the hand pose you want as the start position and click `Snapshot Start Pose`.
5. Hold the hand pose you want as the end position and click `Snapshot End Pose`.
6. Click `Save Sequence Gesture`.
7. Select that saved gesture in the mapping panel.
8. Capture or type a shortcut such as `ctrl+shift+n`.
9. Click `Save Mapping`.
10. Optionally click `Test Mapping` to confirm the shortcut path.
11. Perform the saved gesture while the camera is running.

## Project Layout

- `src/motion_cut/main.py`: entrypoint for UI mode and CLI preview mode
- `src/motion_cut/ui/`: desktop UI and live gesture flow
- `src/motion_cut/vision/`: camera capture and hand tracking
- `src/motion_cut/gestures/`: gesture matching and helper logic
- `src/motion_cut/actions/`: keyboard shortcut dispatch
- `src/motion_cut/storage/`: SQLite-backed template and mapping persistence
- `tests/`: regression tests for matcher behavior

## Development

Run the current test module with:

```sh
PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_smoke
```

## Suggested Public Positioning

For the first public GitHub release, describe Motion Cut as:

- a Linux-first alpha
- source-install only
- focused on custom webcam gestures mapped to shortcuts
- intended for early feedback on recognition quality and UX

## Near-Term Release Tasks

- Add screenshots or a short demo video to the repository
- Add a license file before publishing publicly
- Decide whether `motion_cut.db` should stay out of version control for release builds
- Test on a second Linux environment and document any desktop-specific issues
- Package a first end-user distribution path if you want non-technical users to adopt it
