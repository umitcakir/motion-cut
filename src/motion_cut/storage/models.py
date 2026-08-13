from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GestureTemplate:
    id: int | None
    name: str
    sample_path: str
    threshold: float = 0.75


@dataclass(slots=True)
class GestureMapping:
    id: int | None
    gesture_name: str
    action_type: str
    action_payload: str
