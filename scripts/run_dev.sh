#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_DIR="$ROOT_DIR/data/models"
GESTURE_MODEL_FILE="$MODEL_DIR/gesture_recognizer.task"

mkdir -p "$MODEL_DIR"
if [ ! -f "$GESTURE_MODEL_FILE" ]; then
	echo "[motion-cut] Downloading gesture recognizer model..."
	curl -L "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" -o "$GESTURE_MODEL_FILE"
fi

TF_CPP_MIN_LOG_LEVEL=3 \
GLOG_minloglevel=3 \
ABSL_MIN_LOG_LEVEL=3 \
PYTHONPATH=src python -m motion_cut.main
