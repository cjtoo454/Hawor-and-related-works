import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('short_tail_frontend',
    Path(__file__).parents[1] / 'scripts/prepare_hawor_frontend.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_short_tail_uses_past_context_without_changing_regular_chunks():
    assert module.slam_tracking_start(1000, 1013) == 0
    assert module.slam_tracking_start(2000, 2013) == 1000
    assert module.slam_tracking_start(0, 1000) == 0
    assert module.slam_tracking_start(1000, 2000) == 1000
    with pytest.raises(ValueError):
        module.slam_tracking_start(1000, 1000)


def test_context_crop_preserves_requested_frames_and_local_keyframe_indices():
    trajectory = np.arange(1013 * 7).reshape(1013, 7)
    timestamps = np.array([0, 999, 1000, 1012])
    disparities = np.arange(4).reshape(4, 1, 1)
    output, ticks, disps = module.crop_slam_context(trajectory, timestamps, disparities, 1000, 13)
    np.testing.assert_array_equal(output, trajectory[1000:])
    np.testing.assert_array_equal(ticks, [0, 12])
    np.testing.assert_array_equal(disps, disparities[2:])
    with pytest.raises(ValueError):
        module.crop_slam_context(trajectory, timestamps, disparities, 1000, 14)


def test_short_tail_rejects_unprepared_context_masks():
    with pytest.raises(ValueError, match='Missing completed HaWoR masks'):
        module.validate_context_mask_coverage({'masks/1000_1013.npy': 'tail'}, 0, 1013)
    module.validate_context_mask_coverage(
        {'masks/0_500.npy': 'a', 'masks/500_1000.npy': 'b', 'masks/1000_1013.npy': 'tail'},
        0, 1013)
