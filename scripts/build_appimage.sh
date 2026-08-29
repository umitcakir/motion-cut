#!/usr/bin/env bash
# Build a Linux AppImage for Motion Cut.
# Requirements: Python venv activated, appimagetool available (downloaded if missing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

APP_NAME="motion-cut"
ARCH="$(uname -m)"

# ── 1. Ensure model files are present ────────────────────────────────────────
MODEL_DIR="data/models"
mkdir -p "$MODEL_DIR"
HAND_MODEL="$MODEL_DIR/hand_landmarker.task"
if [ ! -f "$HAND_MODEL" ]; then
    echo "Downloading hand_landmarker.task …"
    curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
         -o "$HAND_MODEL"
fi

# ── 2. Install build dependencies ────────────────────────────────────────────
if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Python 3 interpreter not found."
    exit 1
fi

APP_VERSION="$("$PYTHON_BIN" -c "import tomllib; d=tomllib.load(open('pyproject.toml', 'rb')); print(d['project']['version'])")"
APPIMAGE_OUT="dist/${APP_NAME}-${APP_VERSION}-${ARCH}.AppImage"

"$PYTHON_BIN" -m pip install --quiet -r requirements.txt pyinstaller

# ── 3. Build onedir bundle with PyInstaller ───────────────────────────────────
# Remove any prior build artifact that would block COLLECT creating a fresh directory
rm -rf "$ROOT/dist/motion-cut"
"$PYTHON_BIN" -m PyInstaller motion_cut_appimage.spec --clean --noconfirm
echo "PyInstaller build complete → dist/${APP_NAME}/"

# ── 4. Generate a placeholder icon if none exists ────────────────────────────
ICON_SRC="assets/motion-cut.png"
if [ ! -f "$ICON_SRC" ]; then
    echo "No icon found — generating placeholder with Python …"
    python - <<'PYEOF'
try:
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (256, 256), (30, 30, 46, 255))
    d = ImageDraw.Draw(img)
    # simple hand silhouette stand-in: white rounded rectangle
    d.rounded_rectangle([40, 40, 216, 216], radius=48, fill=(200, 200, 220, 255))
    img.save("assets/motion-cut.png")
    print("Icon created with Pillow.")
except ImportError:
    # fallback: 1×1 transparent PNG (valid but invisible)
    import struct, zlib, base64
    PNG1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    with open("assets/motion-cut.png", "wb") as f:
        f.write(PNG1x1)
    print("Pillow not found — wrote minimal placeholder icon.")
PYEOF
fi

# ── 5. Fetch appimagetool if not on PATH ──────────────────────────────────────
APPIMAGETOOL="$(command -v appimagetool 2>/dev/null || true)"
TOOLS_DIR="$ROOT/.build-tools"
mkdir -p "$TOOLS_DIR"

if [ -z "$APPIMAGETOOL" ]; then
    TOOL_PATH="$TOOLS_DIR/appimagetool-${ARCH}.AppImage"
    if [ ! -f "$TOOL_PATH" ]; then
        echo "Downloading appimagetool …"
        curl -fsSL "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage" \
             -o "$TOOL_PATH"
        chmod +x "$TOOL_PATH"
    fi
    APPIMAGETOOL="$TOOL_PATH"
fi

# Pre-download the AppImage type-2 runtime so appimagetool doesn't need to fetch it
# (appimagetool doesn't follow HTTP redirects; curl does)
RUNTIME_FILE="$TOOLS_DIR/runtime-${ARCH}"
if [ ! -f "$RUNTIME_FILE" ]; then
    echo "Downloading AppImage runtime …"
    curl -fsSL "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-${ARCH}" \
         -o "$RUNTIME_FILE"
fi

# ── 6. Assemble AppDir ────────────────────────────────────────────────────────
APPDIR="$ROOT/dist/${APP_NAME}.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/opt/${APP_NAME}"

# Copy the entire PyInstaller onedir output
cp -r "$ROOT/dist/${APP_NAME}/." "$APPDIR/opt/${APP_NAME}/"

# Desktop file (appimagetool requires it at root)
cp "$ROOT/assets/motion-cut.desktop" "$APPDIR/${APP_NAME}.desktop"

# Icon (appimagetool looks for <AppName>.png at root)
cp "$ROOT/assets/motion-cut.png" "$APPDIR/${APP_NAME}.png"

# AppRun launcher
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
set -e

ydotoold_pid=""
if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] && command -v ydotool >/dev/null 2>&1; then
    if ! pgrep -x ydotoold >/dev/null 2>&1 && command -v ydotoold >/dev/null 2>&1; then
        ydotoold >/dev/null 2>&1 &
        ydotoold_pid=$!

        for _ in {1..20}; do
            if [ -S "${YDOTOOL_SOCKET:-/tmp/.ydotool_socket}" ]; then
                break
            fi
            if ! kill -0 "$ydotoold_pid" 2>/dev/null; then
                echo "[motion-cut] ydotoold failed to start; shortcuts may not work on Wayland." >&2
                break
            fi
            sleep 0.05
        done
    fi
fi

cleanup() {
    if [ -n "$ydotoold_pid" ] && kill -0 "$ydotoold_pid" 2>/dev/null; then
        kill "$ydotoold_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

exec "$APPDIR/opt/motion-cut/motion-cut" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# ── 7. Package AppImage ───────────────────────────────────────────────────────
mkdir -p "$ROOT/dist"
ARCH="$ARCH" "$APPIMAGETOOL" --runtime-file "$RUNTIME_FILE" "$APPDIR" "$APPIMAGE_OUT"

echo ""
echo "AppImage ready → $APPIMAGE_OUT"
