from dataclasses import dataclass


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 360


@dataclass(slots=True)
class RuntimeConfig:
    show_preview: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
