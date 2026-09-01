from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading

import cv2

from motion_cut.config import CameraConfig


class CameraBusyError(RuntimeError):
    pass


def _ensure_macos_camera_permission(
    platform_name: str | None = None,
    avfoundation=None,
) -> tuple[bool, str]:
    platform_name = platform_name or sys.platform
    if platform_name != "darwin":
        return True, ""

    avf = avfoundation
    if avf is None:
        try:
            avf = importlib.import_module("AVFoundation")
        except Exception:
            return True, ""

    capture_device = getattr(avf, "AVCaptureDevice", None)
    media_type = getattr(avf, "AVMediaTypeVideo", None)
    if capture_device is None or media_type is None:
        return (
            False,
            "Camera permission API is unavailable. "
            "Allow camera access in System Settings > Privacy & Security > Camera.",
        )

    status = capture_device.authorizationStatusForMediaType_(media_type)
    authorized = getattr(avf, "AVAuthorizationStatusAuthorized", 3)
    not_determined = getattr(avf, "AVAuthorizationStatusNotDetermined", 0)

    if status == authorized:
        return True, ""

    if status == not_determined:
        gate = threading.Event()
        granted_flag = {"ok": False}

        def on_result(granted) -> None:
            granted_flag["ok"] = bool(granted)
            gate.set()

        try:
            capture_device.requestAccessForMediaType_completionHandler_(media_type, on_result)
            gate.wait(timeout=8.0)
        except Exception:
            return (
                False,
                "Could not request camera access on macOS. "
                "Allow camera access in System Settings > Privacy & Security > Camera.",
            )

        if granted_flag["ok"]:
            return True, ""

    return (
        False,
        "Camera access is blocked. Allow it in System Settings > Privacy & Security > "
        "Camera, then restart the app.",
    )


def _capture_apis_for_platform(platform_name: str | None = None) -> list[int]:
    platform_name = platform_name or sys.platform
    if platform_name == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    if platform_name.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if platform_name.startswith("win"):
        apis: list[int] = []
        for api in (getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)):
            if api is not None and api not in apis:
                apis.append(api)
        apis.append(cv2.CAP_ANY)
        return apis
    return [cv2.CAP_ANY]


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self.last_error_message = ""
        self.last_active_device_index: int | None = None

    def open(self) -> None:
        if self._cap is not None:
            return

        allowed, permission_message = _ensure_macos_camera_permission()
        if not allowed:
            self.last_error_message = permission_message
            raise RuntimeError(permission_message)

        candidates = self._device_candidates()
        capture_apis = _capture_apis_for_platform()
        opened_without_frames = False

        for index in candidates:
            cap = None
            for capture_api in capture_apis:
                candidate = cv2.VideoCapture(index, capture_api)
                if candidate.isOpened():
                    cap = candidate
                    break
                candidate.release()
            if cap is None:
                continue

            opened_without_frames = True
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise read() latency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue

            self._cap = cap
            self._latest_frame = cv2.flip(frame, 1)
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._read_latest_frame,
                name="motion-cut-camera",
                daemon=True,
            )
            self._reader_thread.start()
            self.last_error_message = ""
            self.last_active_device_index = index
            return

        if opened_without_frames:
            self.last_error_message = (
                "Camera device appears to be in use by another application. "
                "Close other camera apps and try again."
            )
            raise CameraBusyError(self.last_error_message)

        self.last_error_message = "Unable to open camera device"
        raise RuntimeError(self.last_error_message)

    @property
    def frame_rate(self) -> float:
        """Capture rate reported by the device, falling back to 30 fps."""
        if self._cap is None:
            return 30.0
        reported = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= reported <= 240.0:
            return 30.0
        return reported

    def _device_candidates(self) -> list[int]:
        candidates = [self._config.device_index, 0, 1]
        if not sys.platform.startswith("linux"):
            return list(dict.fromkeys(candidates))

        available: list[int] = []
        for path in Path("/dev").glob("video*"):
            suffix = path.name.replace("video", "", 1)
            if suffix.isdigit():
                available.append(int(suffix))

        merged = candidates + available
        ordered: list[int] = []
        seen: set[int] = set()
        for index in merged:
            if index in seen:
                continue
            seen.add(index)
            ordered.append(index)
        return ordered

    def close(self) -> None:
        self._reader_stop.set()
        reader = self._reader_thread
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_frame = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self):
        if self._cap is None:
            raise RuntimeError("CameraStream is not initialized")

        with self._frame_lock:
            return self._latest_frame

    def _read_latest_frame(self) -> None:
        cap = self._cap
        if cap is None:
            return

        while not self._reader_stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            with self._frame_lock:
                self._latest_frame = frame
