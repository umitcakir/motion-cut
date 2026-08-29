import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


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


@dataclass(slots=True)
class AppSettings:
    window_width: int = 1200
    window_height: int = 760
    gesture_manager_width: int = 780
    gesture_manager_height: int = 700
    camera_feed_enabled: bool = True


class AppSettingsStore:
    """Persist user-adjustable interface settings between launches."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_settings_path()

    def load(self) -> AppSettings:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AppSettings()

        if not isinstance(raw, dict):
            return AppSettings()

        defaults = AppSettings()
        return AppSettings(
            window_width=self._positive_int(raw.get("window_width"), defaults.window_width),
            window_height=self._positive_int(raw.get("window_height"), defaults.window_height),
            gesture_manager_width=self._positive_int(
                raw.get("gesture_manager_width"), defaults.gesture_manager_width
            ),
            gesture_manager_height=self._positive_int(
                raw.get("gesture_manager_height"), defaults.gesture_manager_height
            ),
            camera_feed_enabled=(
                raw["camera_feed_enabled"]
                if isinstance(raw.get("camera_feed_enabled"), bool)
                else defaults.camera_feed_enabled
            ),
        )

    def save(self, settings: AppSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _positive_int(value: object, default: int) -> int:
        return value if isinstance(value, int) and value > 0 else default


def _default_settings_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "motion-cut" / "settings.json"
    if sys.platform.startswith("win"):
        config_home = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return config_home / "motion-cut" / "settings.json"

    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "motion-cut" / "settings.json"
