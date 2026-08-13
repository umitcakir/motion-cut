#!/usr/bin/env bash
# Build the Linux or macOS executable with PyInstaller.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

# ── 1. ensure model files are present ────────────────────────────────────────
MODEL_DIR="data/models"
mkdir -p "$MODEL_DIR"

HAND_MODEL="$MODEL_DIR/hand_landmarker.task"
if [ ! -f "$HAND_MODEL" ]; then
    echo "Downloading hand_landmarker.task …"
    curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
         -o "$HAND_MODEL"
fi

# ── 2. install build dependencies ────────────────────────────────────────────
pip install --quiet pyinstaller

# ── 3. build ─────────────────────────────────────────────────────────────────
pyinstaller motion_cut.spec --clean --noconfirm

echo ""
echo "Build complete → dist/motion-cut"
