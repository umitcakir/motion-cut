#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
  echo "[motion-cut] Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[motion-cut] Python executable not found in venv."
  exit 1
fi

echo "[motion-cut] Installing/updating dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"

MODEL_DIR="$ROOT_DIR/data/models"
MODEL_FILE="$MODEL_DIR/hand_landmarker.task"
GESTURE_MODEL_FILE="$MODEL_DIR/gesture_recognizer.task"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_FILE" ]; then
  echo "[motion-cut] Downloading hand landmark model..."
  curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o "$MODEL_FILE"
fi
if [ ! -f "$GESTURE_MODEL_FILE" ]; then
  echo "[motion-cut] Downloading gesture recognizer model..."
  curl -L "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" -o "$GESTURE_MODEL_FILE"
fi

if [ "$(uname -s)" = "Linux" ]; then
  if ! command -v playerctl >/dev/null 2>&1; then
    echo "[motion-cut] playerctl not found. Attempting install..."
    if command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm playerctl
    elif command -v apt >/dev/null 2>&1; then
      sudo apt update
      sudo apt install -y playerctl
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y playerctl
    else
      echo "[motion-cut] No supported package manager found for automatic playerctl install."
    fi
  fi
fi

if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
  if ! command -v ydotool >/dev/null 2>&1; then
    echo "[motion-cut] ydotool not found. Attempting install..."
    if command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm ydotool
    elif command -v apt >/dev/null 2>&1; then
      sudo apt update
      sudo apt install -y ydotool
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y ydotool
    else
      echo "[motion-cut] No supported package manager found for automatic ydotool install."
    fi
  fi

  if command -v ydotool >/dev/null 2>&1; then
    echo "[motion-cut] Ensuring ydotool daemon is running..."
    if ! pgrep -x ydotoold >/dev/null 2>&1; then
      (ydotoold >/dev/null 2>&1 &) || true
    fi
  else
    echo "[motion-cut] ydotool is unavailable. Gesture shortcuts may not affect other apps on Wayland."
  fi
fi

echo "[motion-cut] Starting app..."
cd "$ROOT_DIR"
TF_CPP_MIN_LOG_LEVEL=3 \
GLOG_minloglevel=3 \
ABSL_MIN_LOG_LEVEL=3 \
PYTHONPATH=src "$PYTHON_BIN" -m motion_cut.main "$@"
