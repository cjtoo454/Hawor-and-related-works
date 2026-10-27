import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).parents[1] / 'scripts/prepare_training_data.py'
spec = importlib.util.spec_from_file_location('prepare_data_validity', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_explicit_mask_preserves_identity_and_missing_label():
    annotation = {'rot_r': torch.zeros(2, 3), 'valid_r': torch.tensor([True, False])}
    np.testing.assert_array_equal(module.ground_truth_valid(annotation), [True, False])


def test_legacy_mask_and_bad_explicit_shape():
    annotation = {'rot_r': torch.tensor([[0., 0., 0.], [1., 0., 0.]])}
    np.testing.assert_array_equal(module.ground_truth_valid(annotation), [False, True])
    annotation['valid_r'] = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match='frame count'):
        module.ground_truth_valid(annotation)
