from __future__ import annotations

import numpy as np
import pytest

from dual_loop.oracle import (
    PinchOracleConfig,
    detect_pinch_events,
    judge_time_window,
    judge_window,
)


def test_quality_aware_pinch_oracle() -> None:
    joints = np.zeros((20, 21, 3), dtype=np.float32)
    joints[:, 9, 0] = 1.0
    joints[:, 4, 0] = 0.7
    joints[:, 8, 0] = 0.0
    joints[5:10, 4, 0] = 0.05
    quality = np.ones(20, dtype=np.float32)
    config = PinchOracleConfig(0.1, 3, 0.5)
    assert detect_pinch_events(joints, quality, config) == [(5, 9)]
    assert judge_window(joints, quality, (4, 11), config) == "pass"
    quality[4:12] = 0.0
    assert judge_window(joints, quality, (4, 11), config) == "unknown"


def make_oracle_joints(distances: np.ndarray) -> np.ndarray:
    joints = np.zeros((len(distances), 21, 3), dtype=np.float64)
    joints[:, 9, 0] = 1.0
    joints[:, 8, 0] = distances
    return joints


def make_oracle_config() -> PinchOracleConfig:
    return PinchOracleConfig(
        distance_threshold=0.2,
        consecutive_frames=3,
        quality_threshold=0.5,
    )


def test_onset_before_window_does_not_pass_by_overlap() -> None:
    joints = make_oracle_joints(np.array([0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5]))
    quality = np.ones(len(joints))
    config = make_oracle_config()

    assert detect_pinch_events(joints, quality, config) == [(1, 3)]
    assert judge_window(joints, quality, (3, 5), config) == "fail"


def test_nan_or_low_quality_gap_that_can_hide_event_returns_unknown() -> None:
    distances = np.full(12, 0.5)
    config = make_oracle_config()

    joints = make_oracle_joints(distances)
    joints[4:7] = np.nan
    quality = np.ones(len(joints))
    assert judge_window(joints, quality, (1, 10), config) == "unknown"

    joints = make_oracle_joints(distances)
    quality = np.ones(len(joints))
    quality[4:7] = np.nan
    assert judge_window(joints, quality, (1, 10), config) == "unknown"

    joints = make_oracle_joints(distances)
    quality = np.ones(len(joints))
    quality[4:7] = 0.1
    assert judge_window(joints, quality, (1, 10), config) == "unknown"


def test_onset_at_window_end_is_confirmed_by_following_frames() -> None:
    distances = np.full(8, 0.5)
    distances[4:7] = 0.1
    joints = make_oracle_joints(distances)
    quality = np.ones(len(joints))
    config = make_oracle_config()
    assert judge_window(joints, quality, (2, 4), config) == "pass"
    assert judge_window(joints[:6], quality[:6], (2, 4), config) == "unknown"


def test_frame_window_requires_valid_inclusive_indices() -> None:
    joints = make_oracle_joints(np.full(6, 0.5))
    quality = np.ones(len(joints))
    config = make_oracle_config()
    for window in ((-1, 2), (3, 2), (0, 6), (1.0, 3)):
        with pytest.raises(ValueError):
            judge_window(joints, quality, window, config)


def test_timestamp_window_maps_to_sample_onsets() -> None:
    distances = np.full(8, 0.5)
    distances[4:7] = 0.1
    joints = make_oracle_joints(distances)
    quality = np.ones(len(joints))
    timestamps = np.arange(len(joints), dtype=np.float64) / 10
    assert judge_time_window(
        joints, quality, timestamps, (0.2, 0.4), make_oracle_config()) == "pass"
