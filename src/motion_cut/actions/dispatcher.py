from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Iterable

from pynput.keyboard import Controller, Key


class ActionDispatcher:
    _NAMED_ACTIONS = {"lockscreen", "screenshot"}

    def __init__(self) -> None:
        self._keyboard = Controller()
        self._session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        self._ydotool = shutil.which("ydotool")
        self._last_error = ""

    def trigger_shortcut(self, keys: Iterable[str]) -> None:
        normalized = [k.strip().lower() for k in keys if k and k.strip()]
        if not normalized:
            return

        self._last_error = ""
        if len(normalized) == 1 and normalized[0] in self._NAMED_ACTIONS:
            self._trigger_named_action(normalized[0])
            return

        if self._session_type == "wayland" and self._ydotool:
            self._trigger_with_ydotool(normalized)
            return

        if self._session_type == "wayland" and not self._ydotool:
            self._last_error = (
                "Wayland session detected but ydotool is unavailable; using pynput fallback"
            )

        modifiers = [k for k in normalized if k in {"ctrl", "shift", "alt", "cmd"}]
        regular = [k for k in normalized if k not in {"ctrl", "shift", "alt", "cmd"}]

        resolved_modifiers = [self._to_key(k) for k in modifiers]
        resolved_regular = [self._to_key(k) for k in regular]

        # A multi-char leftover string means no pynput Key exists for it.
        unsupported = [
            k for k, r in zip(regular, resolved_regular)
            if isinstance(r, str) and len(r) > 1
        ]
        if unsupported:
            self._last_error = (
                f"Key not supported without ydotool: {', '.join(unsupported)}"
            )
            return

        for key in resolved_modifiers:
            self._keyboard.press(key)
        for key in resolved_regular:
            self._keyboard.press(key)
            self._keyboard.release(key)
            time.sleep(0.01)
        for key in reversed(resolved_modifiers):
            self._keyboard.release(key)

    def _trigger_named_action(self, action: str) -> None:
        if action == "screenshot":
            if sys.platform == "darwin":
                self.trigger_shortcut(["cmd", "shift", "3"])
            elif sys.platform.startswith("win"):
                self.trigger_shortcut(["cmd", "shift", "s"])
            else:
                self.trigger_shortcut(["printscreen"])
            return

        if sys.platform == "darwin":
            command = [
                "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession",
                "-suspend",
            ]
        elif sys.platform.startswith("win"):
            command = ["rundll32.exe", "user32.dll,LockWorkStation"]
        else:
            command = ["loginctl", "lock-session"]

        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            self._last_error = f"Could not lock screen: {exc}"
            return
        if result.returncode != 0:
            self._last_error = result.stderr.strip() or "Could not lock screen"

    @property
    def last_error(self) -> str:
        return self._last_error

    def _trigger_with_ydotool(self, normalized: list[str]) -> None:
        key_codes = []
        for key in normalized:
            code = self._to_ydotool_keycode(key)
            if code is None:
                self._last_error = f"Unsupported key for ydotool backend: {key}"
                return
            key_codes.append(f"{code}:1")

        if not key_codes:
            self._last_error = "No supported keys for ydotool backend"
            return

        release_codes = [code.replace(":1", ":0") for code in reversed(key_codes)]
        command = [self._ydotool, "key", *key_codes, *release_codes]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            self._last_error = result.stderr.strip() or "ydotool key injection failed"

    @staticmethod
    def _to_key(value: str):
        normalized = value.strip().lower()
        special = {
            "ctrl": Key.ctrl,
            "shift": Key.shift,
            "alt": Key.alt,
            "cmd": Key.cmd,
            "enter": Key.enter,
            "return": Key.enter,
            "esc": Key.esc,
            "escape": Key.esc,
            "tab": Key.tab,
            "backspace": Key.backspace,
            "delete": Key.delete,
            "del": Key.delete,
            "ins": Key.insert,
            "insert": Key.insert,
            "home": Key.home,
            "end": Key.end,
            "pageup": Key.page_up,
            "page_up": Key.page_up,
            "prior": Key.page_up,
            "pagedown": Key.page_down,
            "page_down": Key.page_down,
            "next": Key.page_down,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "space": Key.space,
            "f1": Key.f1,  "f2": Key.f2,  "f3": Key.f3,  "f4": Key.f4,
            "f5": Key.f5,  "f6": Key.f6,  "f7": Key.f7,  "f8": Key.f8,
            "f9": Key.f9,  "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
            "volumeup": Key.media_volume_up,
            "volumedown": Key.media_volume_down,
            "mute": Key.media_volume_mute,
            "playpause": Key.media_play_pause,
            "nexttrack": Key.media_next,
            "prevtrack": Key.media_previous,
            "stop": getattr(Key, "media_stop", "media_stop"),
            "printscreen": Key.print_screen,
        }
        return special.get(normalized, normalized)

    @staticmethod
    def _to_ydotool_keycode(value: str) -> int | None:
        key_map = {
            "ctrl": 29,
            "control": 29,
            "shift": 42,
            "alt": 56,
            "cmd": 125,
            "meta": 125,
            "enter": 28,
            "return": 28,
            "esc": 1,
            "escape": 1,
            "tab": 15,
            "backspace": 14,
            "delete": 111,
            "del": 111,
            "ins": 110,
            "insert": 110,
            "home": 102,
            "end": 107,
            "pageup": 104,
            "page_up": 104,
            "prior": 104,
            "pagedown": 109,
            "page_down": 109,
            "next": 109,
            "space": 57,
            "up": 103,
            "down": 108,
            "left": 105,
            "right": 106,
            "f1": 59,  "f2": 60,  "f3": 61,  "f4": 62,
            "f5": 63,  "f6": 64,  "f7": 65,  "f8": 66,
            "f9": 67,  "f10": 68, "f11": 87, "f12": 88,
            "volumeup": 115,
            "volumedown": 114,
            "mute": 113,
            "playpause": 164,
            "nexttrack": 163,
            "prevtrack": 165,
            "stop": 166,
            "printscreen": 99,
            "brightnessup": 225,
            "brightnessdown": 224,
        }

        normalized = value.strip().lower()
        if normalized in key_map:
            return key_map[normalized]

        if len(normalized) == 1 and normalized.isalpha():
            alphabet_codes = {
                "a": 30, "b": 48, "c": 46, "d": 32, "e": 18,
                "f": 33, "g": 34, "h": 35, "i": 23, "j": 36,
                "k": 37, "l": 38, "m": 50, "n": 49, "o": 24,
                "p": 25, "q": 16, "r": 19, "s": 31, "t": 20,
                "u": 22, "v": 47, "w": 17, "x": 45, "y": 21,
                "z": 44,
            }
            return alphabet_codes.get(normalized)

        if len(normalized) == 1 and normalized.isdigit():
            digit_codes = {
                "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
                "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
            }
            return digit_codes.get(normalized)

        return None
