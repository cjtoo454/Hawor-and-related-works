"""Focused CPU checks for world-model DDP batch and gradient behavior."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel


TRY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRY_ROOT / "scripts"))
sys.path.insert(0, str(TRY_ROOT))

from train_world_model import (  # noqa: E402
    RESUME_CRITICAL_KEYS,
    _sample_window_indices,
    _validate_resume_metadata,
)


def _ddp_linear_step(
    rank: int,
    init_method: str,
    initial_state: dict[str, torch.Tensor],
    features: torch.Tensor,
    targets: torch.Tensor,
    output_prefix: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=2)
    model = nn.Linear(features.shape[-1], targets.shape[-1])
    model.load_state_dict(initial_state)
    model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
    indices = _sample_window_indices(
        np.random.default_rng(19), batch_size=8, rank=rank,
        world_size=2, window_count=len(features))
    local_indices = torch.as_tensor(indices, dtype=torch.long)
    prediction = model(features[local_indices])
    loss = torch.nn.functional.mse_loss(prediction, targets[local_indices])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.save(model.module.state_dict(), f"{output_prefix}.{rank}.pt")
    dist.destroy_process_group()


def test_rank_batch_slices_reproduce_one_global_random_batch() -> None:
    expected = np.random.default_rng(7).integers(0, 23, size=8)
    actual = np.concatenate([
        _sample_window_indices(
            np.random.default_rng(7), batch_size=8, rank=rank,
            world_size=2, window_count=23)
        for rank in range(2)
    ])

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual[:4], actual[4:])
    with pytest.raises(ValueError, match="divisible"):
        _sample_window_indices(
            np.random.default_rng(7), batch_size=7, rank=0,
            world_size=2, window_count=23)


def test_cpu_gloo_ddp_update_matches_single_process_global_batch(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(31)
    features = torch.randn(8, 5)
    targets = torch.randn(8, 3)
    reference = nn.Linear(5, 3)
    initial_state = copy.deepcopy(reference.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.025)
    indices = np.random.default_rng(19).integers(0, len(features), size=8)
    selected = torch.as_tensor(indices, dtype=torch.long)
    loss = torch.nn.functional.mse_loss(reference(features[selected]), targets[selected])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    output_prefix = str(tmp_path / "rank")
    mp.spawn(
        _ddp_linear_step,
        args=(f"file://{tmp_path / 'gloo-rendezvous'}", initial_state,
              features, targets, output_prefix),
        nprocs=2,
        join=True,
    )

    for rank in range(2):
        actual_state = torch.load(f"{output_prefix}.{rank}.pt", weights_only=True)
        for name, expected_value in reference.state_dict().items():
            torch.testing.assert_close(actual_state[name], expected_value, rtol=1e-6, atol=1e-7)


def test_topology_can_change_while_other_resume_provenance_stays_strict() -> None:
    saved = {key: f"same-{key}" for key in RESUME_CRITICAL_KEYS}
    saved["device"] = "cuda:0"
    current = dict(saved)
    current["world_size"] = 4
    current["device_topology"] = {"world_size": 4, "requested_device": "cuda:0"}

    _validate_resume_metadata(saved, current, "test topology migration")

    current["prepared_data_sha256"] = "changed"
    with pytest.raises(ValueError, match="prepared_data_sha256"):
        _validate_resume_metadata(saved, current, "test topology migration")
