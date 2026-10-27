"""Integration checks for a differentiable, time-preserving observer rollout."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import pytest
import json

from dual_loop.geometry import ManoGeometry
from dual_loop.model import DualLoopConfig, DualLoopObserver
from dual_loop.state import axis_angle_to_matrix, pack_state, matrix_to_rotation_6d, rotation_6d_to_matrix

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('try_training', ROOT/'thirdparty/try/scripts/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_rotation_encoding_nonidentity():
    rotation = axis_angle_to_matrix(torch.tensor([[.4, -.2, 1.1], [0., 0., 0.]]))
    rebuilt = rotation_6d_to_matrix(matrix_to_rotation_6d(rotation))
    assert torch.allclose(rotation, rebuilt, atol=1e-6)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand(2,3,3), atol=1e-6)


def test_window_index_rejects_gaps_and_unobserved_initialization():
    data = {'visual_valid': np.array([False,True,True,True,True]),
            'gt_valid': np.ones(5,bool), 'segment_id':np.array([0,0,0,1,1]),
            'timestamp_sec':np.arange(5)/30}
    assert training.window_index([data], 2) == [(0,1),(0,3)]


def test_full_training_rejects_unverified_observations(tmp_path):
    split=tmp_path/'split.json'
    split.write_text(json.dumps({'train':['P0002_example'], 'calibration':['P0015_example']}))
    for sequence in ['P0002_example','P0015_example']:
        np.savez(tmp_path/f'{sequence}.npz', segment_id=np.zeros(2), timestamp_sec=np.arange(2)/30,
                 coordinate_frame=np.asarray('per-interval local raw HaWoR SLAM world'),
                 oracle_assisted=np.asarray(False),oracle_assisted_status=np.asarray('unknown'),
                 input_provenance_json=np.asarray(json.dumps({'manifest_status':'missing_unknown'})))
    with pytest.raises(ValueError,match='origin is unverified'):
        training.load_split(SimpleNamespace(split_file=split,data_root=tmp_path))


def test_full_observer_backward_with_missing_supervision(tmp_path):
    torch.set_num_threads(1)
    geometry = ManoGeometry(ROOT)
    config = DualLoopConfig(feature_dim=16, evidence_dim=16, hidden_dim=32, image_size=32)
    model = DualLoopObserver(config)
    root = torch.eye(3).reshape(1,1,3,3)
    hand = root.repeat(1,15,1,1)
    state = pack_state(torch.tensor([[0.,0.,.8]]),root,hand,torch.zeros(1,10)).numpy()
    path=tmp_path/'frame.jpg'
    cv2.imwrite(str(path),np.full((32,32,3),128,np.uint8))
    sequence={'visual_state':np.repeat(state,4,0),'gt_state':np.repeat(state,4,0),
              'visual_valid':np.array([True,False,True,True]),'gt_valid':np.array([True,False,True,True]),
              'rotation_w2c':np.repeat(np.eye(3,dtype=np.float32)[None],4,0),
              'translation_w2c':np.zeros((4,3),np.float32),'focal_length':np.full(4,25,np.float32),
              'image_size':np.array([32,32]),'image_paths':[str(path)]*4}
    # If missing supervision were used as an input it would contaminate the rollout.
    sequence['gt_state'][1] = np.nan
    args=SimpleNamespace(window=4,quality_error_m=.03)
    loss=training.run_window(model,geometry,sequence,0,args,torch.device('cpu'))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.world_model.delta_head.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
