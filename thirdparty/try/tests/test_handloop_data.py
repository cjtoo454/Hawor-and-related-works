from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from dual_loop.handloop_state import pack_state, state_to_legacy
from dual_loop.state import pack_state as pack_legacy, unpack_state as unpack_legacy


def prepare_module():
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('prepare_handloop_data_test', project / 'scripts/prepare_handloop_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WristGeometry:
    def joints(self, state):
        translation, _, _, betas = unpack_legacy(state)
        offset = torch.tensor([.03, -.02, .01]).to(state) + betas[:, :1] * .01
        return (translation + offset)[:, None].expand(-1, 21, -1)


def test_motion_converter_decodes_actual_wrist_and_masks_invalid(tmp_path):
    module = prepare_module()
    states = pack_legacy(torch.zeros(4, 3), torch.eye(3).expand(4, 1, 3, 3),
                         torch.eye(3).expand(4, 15, 3, 3), torch.ones(4, 10)).numpy()
    states[1] = np.nan
    source, destination = tmp_path / 'source.npz', tmp_path / 'destination.npz'
    np.savez(source, states=states, valid=np.array([True, False, True, True]),
             target_timestamps_ns=np.array([0, 33333333, 66666666, 1000000000]))
    module.convert_motion(source, destination, WristGeometry())
    with np.load(destination) as converted:
        np.testing.assert_allclose(converted['state'][0, :3], [.04, -.01, .02], atol=1e-7)
        assert not converted['available'][1].any()
        assert converted['segment_id'].tolist() == [0, 0, 0, 1]
        target = torch.from_numpy(converted['state'][[0, 2]])
    rebuilt = state_to_legacy(target, torch.ones(2, 10), WristGeometry())
    torch.testing.assert_close(rebuilt[:, :3], torch.zeros(2, 3), atol=1e-7, rtol=0)


def test_crop_fill_uses_only_past_detections_and_resets():
    module = prepare_module()
    boxes = np.array([[np.nan]*4, [10,20,30,40], [np.nan]*4, [np.nan]*4], np.float32)
    result = module.fill_boxes(boxes, np.array([100, 80]), np.array([0,0,0,1]))
    np.testing.assert_equal(result[0], [0,0,100,80])
    np.testing.assert_equal(result[2], [10,20,30,40])
    np.testing.assert_equal(result[3], [0,0,100,80])
