"""Quality-aware pinch oracle over a Physical State Trace."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class PinchOracleConfig:
    distance_threshold: float
    consecutive_frames: int
    quality_threshold: float
    short_event_max_frames: int = 10

    def __post_init__(self) -> None:
        if not np.isfinite(self.distance_threshold) or self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be finite and positive")
        if (isinstance(self.consecutive_frames, bool)
                or not isinstance(self.consecutive_frames, Integral)
                or self.consecutive_frames < 1):
            raise ValueError("consecutive_frames must be a positive integer")
        if (not np.isfinite(self.quality_threshold)
                or not 0 <= self.quality_threshold <= 1):
            raise ValueError("quality_threshold must be finite and between 0 and 1")
        if (isinstance(self.short_event_max_frames, bool)
                or not isinstance(self.short_event_max_frames, Integral)
                or self.short_event_max_frames < 1):
            raise ValueError("short_event_max_frames must be a positive integer")


def normalized_pinch_distance(joints: np.ndarray) -> np.ndarray:
    """Thumb-index distance normalized by wrist-to-middle-MCP hand scale."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape[-2:] != (21, 3):
        raise ValueError("Expected joints with shape [..., 21, 3]")
    fingertip = np.linalg.norm(joints[..., 4, :] - joints[..., 8, :], axis=-1)
    hand_scale = np.linalg.norm(joints[..., 0, :] - joints[..., 9, :], axis=-1)
    valid_scale = np.isfinite(hand_scale) & (hand_scale > 1e-6)
    distance = np.full(np.shape(fingertip), np.nan, dtype=np.float64)
    np.divide(fingertip, hand_scale, out=distance, where=valid_scale)
    return distance


_UNKNOWN = np.int8(-1)
_OPEN = np.int8(0)
_PINCHED = np.int8(1)


def _validated_trace(
    joints: np.ndarray,
    quality: np.ndarray,
    config: PinchOracleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError("Expected joints with shape [frames, 21, 3]")
    if quality.ndim != 1 or len(quality) != len(joints):
        raise ValueError("quality must be a one-dimensional value per frame")

    distance = normalized_pinch_distance(joints)
    quality_valid = np.isfinite(quality) & (quality >= config.quality_threshold)
    evidence_valid = quality_valid & np.isfinite(distance)
    states = np.full(len(distance), _UNKNOWN, dtype=np.int8)
    states[evidence_valid & (distance >= config.distance_threshold)] = _OPEN
    states[evidence_valid & (distance < config.distance_threshold)] = _PINCHED
    return distance, quality, states


def _validate_frame_window(
    expected_window: tuple[int, int], frame_count: int,
) -> tuple[int, int]:
    try:
        if len(expected_window) != 2:
            raise ValueError
        start, stop = expected_window
    except (TypeError, ValueError):
        raise ValueError("expected_window must be an inclusive (start, stop) pair") from None
    if any(isinstance(index, bool) or not isinstance(index, Integral)
           for index in (start, stop)):
        raise ValueError("expected_window must contain integer frame indices")
    start, stop = int(start), int(stop)
    if frame_count == 0 or start < 0 or stop < start or stop >= frame_count:
        raise ValueError(
            f"expected_window must satisfy 0 <= start <= stop < {frame_count}")
    return start, stop


def detect_pinch_events(
    joints: np.ndarray,
    quality: np.ndarray,
    config: PinchOracleConfig,
) -> list[tuple[int, int]]:
    _, _, states = _validated_trace(joints, quality, config)
    events: list[tuple[int, int]] = []

    def close_run(stop: int, start: int | None) -> None:
        if start is None or stop - start < config.consecutive_frames:
            return
        # A run after missing or unreliable evidence may have started earlier;
        # its onset is not known well enough to count as a confirmed event.
        if start > 0 and states[start - 1] != _OPEN:
            return
        events.append((start, stop - 1))

    start: int | None = None
    for index, state in enumerate(states):
        if state == _PINCHED:
            if start is None:
                start = index
        else:
            close_run(index, start)
            start = None
    close_run(len(states), start)
    return events


def judge_window(
    joints: np.ndarray,
    quality: np.ndarray,
    expected_window: tuple[int, int],
    config: PinchOracleConfig,
) -> str:
    _, _, states = _validated_trace(joints, quality, config)
    start, stop = _validate_frame_window(expected_window, len(states))
    events = detect_pinch_events(joints, quality, config)
    if any(start <= onset <= stop for onset, _ in events):
        return "pass"

    # For every possible onset in the requested window, inspect its complete
    # m-frame confirmation interval. A trusted open frame rules that candidate
    # out; unknown frames leave it possible. This includes the m-1 frames after
    # the window end, which are needed to confirm an onset at `stop`.
    for onset in range(start, stop + 1):
        if onset > 0 and states[onset - 1] == _PINCHED:
            continue  # this frame continues an event whose onset was earlier
        confirmation = states[onset:onset + config.consecutive_frames]
        if not np.any(confirmation == _OPEN):
            return "unknown"
    return "fail"


def judge_time_window(
    joints: np.ndarray,
    quality: np.ndarray,
    timestamps: np.ndarray,
    expected_time_window: tuple[float, float],
    config: PinchOracleConfig,
) -> str:
    """Judge an inclusive timestamp window using the trace's frame timestamps.

    Timestamps and window bounds must use the same unit and time origin. The
    timestamp array must be finite and strictly increasing. If the trace does
    not bracket the whole requested interval, a non-passing frame judgment is
    downgraded to ``unknown`` because part of the requested time lacks evidence.
    """
    _, _, states = _validated_trace(joints, quality, config)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) != len(states) or len(timestamps) == 0:
        raise ValueError("timestamps must be a non-empty value per trace frame")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    try:
        if len(expected_time_window) != 2:
            raise ValueError
        start_time, stop_time = expected_time_window
    except (TypeError, ValueError):
        raise ValueError("expected_time_window must be an inclusive (start, stop) pair") from None
    if (isinstance(start_time, bool) or isinstance(stop_time, bool)
            or not isinstance(start_time, Real) or not isinstance(stop_time, Real)
            or not np.isfinite(start_time) or not np.isfinite(stop_time)
            or start_time > stop_time):
        raise ValueError("expected_time_window must contain ordered finite times")

    start = int(np.searchsorted(timestamps, start_time, side="left"))
    stop = int(np.searchsorted(timestamps, stop_time, side="right") - 1)
    if start > stop:
        return "unknown"
    result = judge_window(joints, quality, (start, stop), config)
    if result != "pass" and (timestamps[0] > start_time or timestamps[-1] < stop_time):
        return "unknown"
    return result

