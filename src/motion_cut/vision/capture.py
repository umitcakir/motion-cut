from __future__ import annotations

from pathlib import Path

import cv2

from motion_cut.config import CameraConfig


class CameraBusyError(RuntimeError):
    pass


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self.last_error_message = ""
        self.last_active_device_index: int | None = None

    def open(self) -> None:
        if self._cap is not None:
            return

        candidates = self._device_candidates()
        opened_without_frames = False

        for index in candidates:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
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

    def _device_candidates(self) -> list[int]:
        candidates = [self._config.device_index, 0, 1]
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
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self):
        if self._cap is None:
            raise RuntimeError("CameraStream is not initialized")

        ok, frame = self._cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        return frame
