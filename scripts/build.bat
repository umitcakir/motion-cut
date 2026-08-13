@echo off
REM Build the Windows executable with PyInstaller.

cd /d "%~dp0.."

REM ── 1. ensure model files are present ────────────────────────────────────
if not exist "data\models" mkdir "data\models"

if not exist "data\models\hand_landmarker.task" (
    echo Downloading hand_landmarker.task ...
    curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" ^
         -o "data\models\hand_landmarker.task"
)

REM ── 2. install build dependencies ────────────────────────────────────────
pip install --quiet pyinstaller

REM ── 3. build ─────────────────────────────────────────────────────────────
pyinstaller motion_cut.spec --clean --noconfirm

echo.
echo Build complete -- dist\motion-cut.exe
