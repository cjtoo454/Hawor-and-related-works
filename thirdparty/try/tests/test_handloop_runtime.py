from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dual_loop.handloop_data import read_manifest
from dual_loop.handloop_frontend import HaWoRFrontend, project_mano16
from dual_loop.handloop_runtime import FeedbackHistory, HandLoopSystem, visual_clip_loss
from dual_loop.handloop_state import pack_state
from dual_loop.state import unpack_state as unpack_legacy


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.mano = nn.Module()
        self.mano.register_buffer('joint_map', torch.tensor([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]))

    def joints(self, legacy):
        p, _, _, beta = unpack_legacy(legacy)
        offset = torch.tensor([.01, .02, 0.], device=p.device)
        return (p + offset + beta[:, :1] * .001)[:, None].expand(-1, 21, -1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(3, 1280)

    def forward(self, image):
        return self.readout(image.mean((-2, -1)))[:, :, None, None].expand(-1, -1, 16, 12)


class FakeHaWoR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.pose = nn.Linear(1280, 13)

    def forward(self, batch):
        image = batch['img'].flatten(0, 1)
        features = self.backbone(image)
        p = self.pose(features.mean((-2, -1)))
        rotations = torch.eye(3).to(image).reshape(1, 1, 3, 3).expand(len(image), 16, 3, 3)
        return {'pred_mano_params': {'global_orient': rotations[:, :1], 'hand_pose': rotations[:, 1:], 'betas': p[:, 3:]},
                'out': {'trans_full': p[:, :3, None].transpose(-1, -2) + image.new_tensor([0., 0., 1.]),
                        'pred_cam': p[:, :3]}}


def make_batch():
    b, t = 1, 16
    identity = torch.eye(3).reshape(1, 1, 3, 3).expand(b, t, 3, 3).clone()
    target = pack_state(torch.tensor([0., 0., 1.]).expand(b, t, 3),
                        torch.eye(3).expand(b, t, 16, 3, 3))
    return {'hawor': {'img': torch.rand(b, t, 3, 8, 8), 'center': torch.zeros(b, t, 2),
                       'scale': torch.ones(b, t), 'img_focal': torch.ones(b, t), 'img_center': torch.zeros(b, t, 2)},
            'pixel_to_patch': identity, 'rotation_w2c': identity, 'translation_w2c': torch.zeros(b, t, 3),
            'timestamp_sec': torch.arange(t).double()[None] / 30, 'segment_id': torch.zeros(b, t, dtype=torch.long),
            'handedness': torch.ones(b, t, dtype=torch.long), 'raw_available': torch.ones(b, t, 17, dtype=torch.bool),
            'image_available': torch.ones(b, t, dtype=torch.bool), 'target_state': target,
            'target_available': torch.ones(b, t, 17, dtype=torch.bool),
            'target_joints': target[..., :3, None].transpose(-1, -2).expand(b, t, 21, 3).clone(),
            'joint_valid': torch.ones(b, t, 21, dtype=torch.bool), 'supervise': torch.ones(b, t, dtype=torch.bool)}


def test_actual_iam_hook_and_live_backward():
    from dual_loop.handloop_model import HandLoopConfig, HandLoopObserver
    torch.set_num_threads(1)
    config = HandLoopConfig.from_dict({'architecture': 'handloop_v2', 'rounds': 2,
        'twm': {'width': 16, 'layers': 1, 'heads': 4, 'state_size': 2, 'expand': 1,
                'experts': 2, 'history_length': 3, 'modes': 2, 'future_steps': 3},
        'visual': {'width': 16, 'heads': 4, 'feature_dim': 1280, 'blocks': 1}})
    hawor = FakeHaWoR()
    system = HandLoopSystem(hawor, HandLoopObserver(config), Geometry())
    batch = make_batch()
    # Invalid targets must never reach geometry or probability calculations.
    batch['target_available'][:, 0] = False
    batch['joint_valid'][:, 0] = False
    batch['target_state'][:, 0] = float('nan')
    batch['target_joints'][:, 0] = float('nan')
    losses = visual_clip_loss(system, batch)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    assert hawor.backbone.readout.weight.grad is not None
    assert torch.isfinite(hawor.backbone.readout.weight.grad).all()
    assert hawor.pose.weight.grad.abs().sum() > 0
    assert system.observer.twm is not None
    assert not hawor.backbone._forward_hooks


def test_history_commit_detaches_and_resets_individual_rows():
    history = FeedbackHistory(3)
    raw = torch.zeros(2, 99, requires_grad=True)
    time = torch.zeros(2, dtype=torch.double)
    segment = torch.zeros(2, dtype=torch.long)
    hand = torch.ones(2, dtype=torch.long)
    history.before(time, segment, hand)
    history.commit(raw, torch.ones(2, 17, dtype=torch.bool), time, torch.ones(2, 17))
    history.before(torch.tensor([1 / 30, .3]), segment, hand)
    result = history.tensors(raw, time)
    assert result['history_mask'].tolist() == [[True], [False]]
    assert not result['history'].requires_grad


def test_manifest_rejects_participant_overlap(tmp_path):
    path = tmp_path / 'manifest.json'
    entry = {'path': 'seq.npz', 'source': 'HOT3D', 'participant': 'P0001'}
    path.write_text(json.dumps({'schema': 'handloop_v2', 'train': [entry], 'calibration': [entry]}))
    with pytest.raises(ValueError, match='leakage'):
        read_manifest(path, 'train')


def test_projector_masks_invalid_depth_and_respects_crop():
    state = pack_state(torch.tensor([[0., 0., 1.], [0., 0., -1.]]), torch.eye(3).expand(2, 16, 3, 3))
    grid, valid = project_mano16(state, torch.zeros(2, 10), Geometry(),
                                 torch.eye(3).expand(2, 3, 3), torch.zeros(2, 3),
                                 torch.ones(2), torch.zeros(2, 2), torch.eye(3).expand(2, 3, 3))
    assert grid.shape == (2, 16, 2)
    assert valid[0].all() and not valid[1].any()
