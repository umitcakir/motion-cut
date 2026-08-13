from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST = 0
MIDDLE_MCP = 9
PINKY_MCP = 17
PALM_POINTS = (0, 5, 9, 13, 17)

RESAMPLE_FRAMES = 24
DEPTH_WEIGHT = 0.4

POSE_TOLERANCE = 1.10
DYNAMICS_TOLERANCE = 0.55
TRAJECTORY_TOLERANCE = 0.90

POSE_WEIGHT = 0.40
DYNAMICS_WEIGHT = 0.35
TRAJECTORY_WEIGHT = 0.25

STATIONARY_TRAVEL = 0.80
STRAIGHTNESS_MIN = 0.55
DIRECTION_MIN_COSINE = 0.25
ROTATION_STRAIGHTNESS_MAX = 0.35
ROTATION_AREA_MIN = 0.003
ROTATION_AREA_TOLERANCE = 0.035


@dataclass(slots=True)
class SegmenterConfig:
    """Thresholds expressed in hand-size units, so camera distance does not matter."""

    start_delta: float = 0.055
    stop_delta: float = 0.030
    idle_frames: int = 3
    min_frames: int = 8
    max_frames: int = 90
    preroll_frames: int = 6
    activity_ratio: float = 0.22
    padding_frames: int = 3


@dataclass(slots=True)
class GestureFeatures:
    pose: np.ndarray
    dynamics: np.ndarray
    trajectory: np.ndarray
    net_displacement: np.ndarray
    travel: float
    straightness: float
    trajectory_area: float


@dataclass(slots=True)
class MatchBreakdown:
    name: str
    score: float
    pose_score: float
    dynamics_score: float
    trajectory_score: float
    direction_cosine: float
    mirrored: bool


def _normalized_signed_area(path: np.ndarray, travel: float) -> float:
    if path.shape[0] < 3:
        return 0.0

    x = path[:, 0]
    y = path[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    scale = max(travel * travel, 1e-6)
    return area / scale


def as_frames(sequence) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 2 and array.shape[1] % 3 == 0:
        return array.reshape(array.shape[0], -1, 3)
    return array


def hand_scale(frames: np.ndarray) -> float:
    span = np.linalg.norm(frames[:, MIDDLE_MCP, :2] - frames[:, WRIST, :2], axis=1)
    scale = float(np.median(span))
    if scale < 1e-4:
        spread = np.linalg.norm(frames[:, :, :2] - frames[:, WRIST : WRIST + 1, :2], axis=2)
        scale = float(spread.max(initial=0.0))
    return max(scale, 1e-4)


def resample_track(track: np.ndarray, target_frames: int = RESAMPLE_FRAMES) -> np.ndarray:
    count = track.shape[0]
    if count == target_frames:
        return track.astype(np.float32, copy=False)
    if count == 1:
        return np.repeat(track.astype(np.float32, copy=False), target_frames, axis=0)

    source_steps = np.linspace(0.0, 1.0, count, dtype=np.float32)
    target_steps = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    flat = track.reshape(count, -1)
    resampled = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        resampled[:, column] = np.interp(target_steps, source_steps, flat[:, column])
    return resampled.reshape((target_frames,) + track.shape[1:])


def _motion_progress(palm_track: np.ndarray) -> np.ndarray:
    """Normalized cumulative travel from 0..1; invariant to speed changes over time."""

    if palm_track.shape[0] < 2:
        return np.linspace(0.0, 1.0, palm_track.shape[0], dtype=np.float32)

    steps = np.linalg.norm(np.diff(palm_track, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps, dtype=np.float32)))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.linspace(0.0, 1.0, palm_track.shape[0], dtype=np.float32)
    return (cumulative / total).astype(np.float32, copy=False)


def resample_track_by_progress(
    track: np.ndarray,
    progress: np.ndarray,
    target_frames: int = RESAMPLE_FRAMES,
) -> np.ndarray:
    """Resample track by movement progress, not frame index, to remove speed sensitivity."""

    count = track.shape[0]
    if count == target_frames:
        return track.astype(np.float32, copy=False)
    if count == 1:
        return np.repeat(track.astype(np.float32, copy=False), target_frames, axis=0)

    source = np.asarray(progress, dtype=np.float32)
    if source.ndim != 1 or source.size != count:
        source = np.linspace(0.0, 1.0, count, dtype=np.float32)

    source = np.maximum.accumulate(source)
    unique_source, unique_indices = np.unique(source, return_index=True)
    flat = track.reshape(count, -1)
    flat = flat[unique_indices]

    if unique_source.size < 2:
        return np.repeat(flat[:1], target_frames, axis=0).reshape((target_frames,) + track.shape[1:])

    target = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    resampled = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        resampled[:, column] = np.interp(target, unique_source, flat[:, column])
    return resampled.reshape((target_frames,) + track.shape[1:])


def frame_motion_delta(previous, current) -> float:
    previous_frame = as_frames(previous)[0]
    current_frame = as_frames(current)[0]
    scale = hand_scale(np.stack((previous_frame, current_frame)))
    movement = np.linalg.norm(current_frame[:, :2] - previous_frame[:, :2], axis=1).mean()
    return float(movement / scale)


def sequence_motion_deltas(sequence) -> np.ndarray:
    frames = as_frames(sequence)
    if frames.shape[0] < 2:
        return np.asarray([], dtype=np.float32)
    scale = hand_scale(frames)
    movement = np.linalg.norm(np.diff(frames[:, :, :2], axis=0), axis=2).mean(axis=1)
    return (movement / scale).astype(np.float32)


def trim_to_motion(sequence, config: SegmenterConfig | None = None) -> np.ndarray:
    config = config or SegmenterConfig()
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] < 3:
        return array

    deltas = sequence_motion_deltas(array)
    peak = float(deltas.max(initial=0.0))
    if peak <= 1e-6:
        return array

    threshold = max(config.stop_delta, peak * config.activity_ratio)
    active = np.flatnonzero(deltas >= threshold)
    if active.size == 0:
        return array

    start = max(0, int(active[0]) - config.padding_frames)
    end = min(array.shape[0], int(active[-1]) + 2 + config.padding_frames)
    trimmed = array[start:end]
    return trimmed if trimmed.size != 0 else array


def build_features(sequence) -> GestureFeatures | None:
    """Turn a raw landmark segment into position, scale and hand invariant features."""

    frames = as_frames(sequence)
    if frames.shape[0] < 2 or frames.shape[1] <= PINKY_MCP:
        return None

    scale = hand_scale(frames)

    palm = frames[:, PALM_POINTS, :2].mean(axis=1)
    progress = _motion_progress(palm)

    shapes = (frames - frames[:, WRIST : WRIST + 1, :]) / scale
    shapes[:, :, 2] *= DEPTH_WEIGHT
    shape_track = resample_track_by_progress(shapes, progress)

    pose = shape_track.mean(axis=0)
    dynamics = shape_track - pose[None, :, :]

    trajectory = resample_track_by_progress((palm - palm[0]) / scale, progress)
    travel = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    net_displacement = trajectory[-1]
    magnitude = float(np.linalg.norm(net_displacement))
    straightness = magnitude / travel if travel > 1e-6 else 0.0
    trajectory_area = _normalized_signed_area(trajectory, travel)

    return GestureFeatures(
        pose=pose.astype(np.float32, copy=False),
        dynamics=dynamics.astype(np.float32, copy=False),
        trajectory=trajectory.astype(np.float32, copy=False),
        net_displacement=net_displacement.astype(np.float32, copy=False),
        travel=travel,
        straightness=straightness,
        trajectory_area=trajectory_area,
    )


def _mirror_pose(pose: np.ndarray) -> np.ndarray:
    mirrored = pose.copy()
    mirrored[:, 0] *= -1.0
    return mirrored


def _mirror_dynamics(dynamics: np.ndarray) -> np.ndarray:
    mirrored = dynamics.copy()
    mirrored[:, :, 0] *= -1.0
    return mirrored


def _shape_scores(
    pose: np.ndarray,
    dynamics: np.ndarray,
    template: GestureFeatures,
) -> tuple[float, float]:
    pose_distance = float(np.linalg.norm(pose - template.pose, axis=1).mean())
    pose_score = max(0.0, 1.0 - (pose_distance / POSE_TOLERANCE))

    dynamics_distance = float(np.linalg.norm(dynamics - template.dynamics, axis=2).mean())
    dynamics_score = max(0.0, 1.0 - (dynamics_distance / DYNAMICS_TOLERANCE))
    return pose_score, dynamics_score


def _trajectory_score(
    current: GestureFeatures,
    template: GestureFeatures,
    mirrored: bool,
) -> tuple[float, float, bool]:
    # Keep rotational direction in screen space so clockwise/counterclockwise remains distinct.
    current_area = current.trajectory_area
    template_area = template.trajectory_area
    current_is_rotational = (
        current.straightness <= ROTATION_STRAIGHTNESS_MAX
        and abs(current_area) >= ROTATION_AREA_MIN
    )
    template_is_rotational = (
        template.straightness <= ROTATION_STRAIGHTNESS_MAX
        and abs(template_area) >= ROTATION_AREA_MIN
    )

    if current_is_rotational != template_is_rotational:
        return 0.25, 1.0, False

    current_moving = current.travel >= STATIONARY_TRAVEL
    template_moving = template.travel >= STATIONARY_TRAVEL

    # Rotational gestures can have short travel; don't bypass direction checks for them.
    if current_is_rotational and template_is_rotational:
        current_moving = True
        template_moving = True

    if not current_moving and not template_moving:
        return 1.0, 1.0, False

    if current_moving != template_moving:
        return 0.35, 1.0, False

    current_path = current.trajectory / max(current.travel, 1e-6)
    template_path = template.trajectory / max(template.travel, 1e-6)
    distance = float(np.linalg.norm(current_path - template_path, axis=1).mean())
    score = max(0.0, 1.0 - (distance / TRAJECTORY_TOLERANCE))

    cosine = 1.0
    blocking = False
    both_straight = (
        current.straightness >= STRAIGHTNESS_MIN
        and template.straightness >= STRAIGHTNESS_MIN
    )
    if both_straight:
        current_magnitude = float(np.linalg.norm(current.net_displacement))
        template_magnitude = float(np.linalg.norm(template.net_displacement))
        if current_magnitude > 1e-6 and template_magnitude > 1e-6:
            cosine = float(
                np.dot(current.net_displacement, template.net_displacement)
                / (current_magnitude * template_magnitude)
            )
            blocking = cosine < DIRECTION_MIN_COSINE

    rotation_score = 1.0
    if current_is_rotational and template_is_rotational:
        if np.sign(current_area) != np.sign(template_area):
            rotation_score = 0.0
            blocking = True
        else:
            delta = abs(current_area - template_area)
            rotation_score = max(0.0, 1.0 - (delta / ROTATION_AREA_TOLERANCE))

    score *= (0.35 + (0.65 * rotation_score))
    return score, cosine, blocking


def compare_features(current: GestureFeatures, template: GestureFeatures) -> MatchBreakdown:
    """Score a candidate against a template, allowing either hand to satisfy the shape."""

    direct = _shape_scores(current.pose, current.dynamics, template)
    mirrored = _shape_scores(
        _mirror_pose(current.pose), _mirror_dynamics(current.dynamics), template
    )

    use_mirror = sum(mirrored) > sum(direct)
    pose_score, dynamics_score = mirrored if use_mirror else direct

    trajectory_score, cosine, blocking = _trajectory_score(
        current,
        template,
        use_mirror,
    )

    score = (
        (pose_score * POSE_WEIGHT)
        + (dynamics_score * DYNAMICS_WEIGHT)
        + (trajectory_score * TRAJECTORY_WEIGHT)
    )
    if blocking:
        score = 0.0

    return MatchBreakdown(
        name="",
        score=score,
        pose_score=pose_score,
        dynamics_score=dynamics_score,
        trajectory_score=trajectory_score,
        direction_cosine=cosine,
        mirrored=use_mirror,
    )


def best_match(
    current: GestureFeatures,
    templates: dict[str, GestureFeatures],
) -> MatchBreakdown | None:
    best: MatchBreakdown | None = None
    for name, template in templates.items():
        breakdown = compare_features(current, template)
        breakdown.name = name
        if best is None or breakdown.score > best.score:
            best = breakdown
    return best


class MotionSegmenter:
    """Detects a deliberate motion: waits for movement to start, ends when it settles."""

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self._config = config or SegmenterConfig()
        self._recent: list[list[float]] = []
        self._active_frames: list[list[float]] = []
        self._last_vector: np.ndarray | None = None
        self._active = False
        self._idle_frames = 0
        self._last_delta = 0.0

    @property
    def config(self) -> SegmenterConfig:
        return self._config

    @property
    def active(self) -> bool:
        return self._active

    @property
    def frame_count(self) -> int:
        return len(self._active_frames)

    @property
    def idle_frames(self) -> int:
        return self._idle_frames

    @property
    def last_delta(self) -> float:
        return self._last_delta

    def reset(self) -> None:
        self._recent.clear()
        self._active_frames.clear()
        self._last_vector = None
        self._active = False
        self._idle_frames = 0
        self._last_delta = 0.0

    def push(self, vector: list[float]) -> np.ndarray | None:
        current = np.asarray(vector, dtype=np.float32)
        delta = 0.0
        if self._last_vector is not None:
            delta = frame_motion_delta(self._last_vector, current)
        self._last_vector = current
        self._last_delta = delta

        self._recent.append(list(vector))
        if len(self._recent) > self._config.preroll_frames:
            self._recent.pop(0)

        if not self._active:
            if delta >= self._config.start_delta:
                self._active = True
                self._idle_frames = 0
                self._active_frames = list(self._recent)
            return None

        self._active_frames.append(list(vector))
        if delta <= self._config.stop_delta:
            self._idle_frames += 1
        else:
            self._idle_frames = 0

        if len(self._active_frames) >= self._config.max_frames:
            return self._emit(self._active_frames)

        if (
            self._idle_frames >= self._config.idle_frames
            and len(self._active_frames) >= self._config.min_frames
        ):
            frames = self._active_frames
            if len(frames) > self._idle_frames:
                frames = frames[: -self._idle_frames]
            return self._emit(frames)

        return None

    def flush(self) -> np.ndarray | None:
        if not self._active or len(self._active_frames) < self._config.min_frames:
            self.reset()
            return None
        return self._emit(self._active_frames)

    def _emit(self, frames: list[list[float]]) -> np.ndarray | None:
        segment = trim_to_motion(np.asarray(frames, dtype=np.float32), self._config)
        self.reset()
        if segment.shape[0] < self._config.min_frames:
            return None
        return segment
