from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np

from dual_loop.oracle import PinchOracleConfig


def test_window_confirmation_cannot_bridge_timestamp_gap():
    path = Path(__file__).resolve().parents[1] / 'scripts/eval_pinch_events.py'
    spec = importlib.util.spec_from_file_location('_pinch_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    joints = np.zeros((5, 21, 3))
    joints[:, 9, 0] = 1
    joints[:, 4, 0] = [.5, .1, .1, .1, .5]
    valid = np.ones(5, dtype=bool)
    config = PinchOracleConfig(.2, 3, .5)
    assert module.judge_expected_window(joints, valid, np.arange(5) / 30, 0., .04, config) == 'pass'
    assert module.judge_expected_window(joints, valid, np.array([0., .03, 1., 1.03, 1.06]), 0., .04, config) == 'unknown'
    complete = np.zeros((6, 21, 3))
    complete[:, 9, 0] = 1
    complete[:, 4, 0] = [.5, .1, .1, .1, .5, .5]
    assert module.judge_expected_window(complete, np.ones(6, dtype=bool),
        np.array([0., .03, .06, .09, .12, 1.]), 0., .12, config) == 'pass'
