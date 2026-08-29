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
if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Python 3 interpreter not found."
    exit 1
fi

"$PYTHON_BIN" -m pip install --quiet -r requirements.txt pyinstaller

# ── 3. build ─────────────────────────────────────────────────────────────────
"$PYTHON_BIN" -m PyInstaller motion_cut.spec --clean --noconfirm

echo ""
echo "Build complete → dist/motion-cut"
