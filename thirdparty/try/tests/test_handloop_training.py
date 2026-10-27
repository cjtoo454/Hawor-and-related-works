from __future__ import annotations

import json
import random
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate


TRY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRY_ROOT))

from dual_loop.handloop_probability import geometric_loss, log_prob, mixture_log_prob  # noqa: E402
from dual_loop.handloop_state import (  # noqa: E402
    apply_delta,
    make_anchor,
    pack_state,
    to_local,
    to_world,
)
from dual_loop.handloop_training import (  # noqa: E402
    CORRUPTION_PROBABILITIES,
    HandLoopBatchStream,
    HandLoopMotionDataset,
    HandLoopTrainer,
    _fill_current_from_history,
    learning_rate_factor,
    load_checkpoint,
    make_optimizer,
    motion_losses,
    motion_training_step,
    normalize_motion_batch,
    parameter_groups_for_system,
    restore_rng_state,
    save_checkpoint,
    validate_manifest,
    validate_cross_manifest_splits,
)


def _states(count: int) -> np.ndarray:
    positions = torch.zeros(count, 3)
    positions[:, 0] = torch.arange(count, dtype=torch.float32) * 0.01
    positions[:, 1] = torch.sin(torch.arange(count, dtype=torch.float32) * 0.03) * 0.02
    rotations = torch.eye(3).expand(count, 16, 3, 3).clone()
    return pack_state(positions, rotations).numpy().astype(np.float32)


def _write_sequence(path: Path, *, participant: str, count: int = 100,
                    gap_at: int | None = None, missing_future: bool = False) -> dict[str, str]:
    states = _states(count)
    timestamps = np.arange(count, dtype=np.float64) / 30.0
    if gap_at is not None:
        timestamps[gap_at:] += 0.3
    available = np.ones((count, 17), dtype=bool)
    if missing_future:
        available[34, 4] = False
        states[34, 3 + 3 * 6:3 + 4 * 6] = np.nan
    handedness = np.full(count, "right")
    segment_id = np.zeros(count, dtype=np.int64)
    if gap_at is not None:
        segment_id[gap_at:] = 1
    np.savez_compressed(
        path,
        state=states,
        timestamp_sec=timestamps,
        available=available,
        handedness=handedness,
        segment_id=segment_id,
    )
    return {"path": str(path), "participant": participant, "source": "synthetic", "kind": "motion"}


def _manifest(tmp_path: Path, *, missing_future: bool = False) -> dict:
    return {
        "schema": "handloop_v2",
        "train": [_write_sequence(tmp_path / "train.npz", participant="P1", missing_future=missing_future)],
        "calibration": [_write_sequence(tmp_path / "calibration.npz", participant="P2")],
        "test": [_write_sequence(tmp_path / "test.npz", participant="P3")],
    }


def _distribution_batch(batch: int = 1, future_steps: int = 3):
    current_target = torch.zeros(batch, 99)
    current_target[:, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(16)
    future_target = current_target[:, None].repeat(1, future_steps, 1)
    future_target[:, :, 0] = torch.arange(1, future_steps + 1, dtype=torch.float32) * 0.02

    current = {
        "state": current_target.clone().requires_grad_(),
        "sigma": torch.full((batch, 3), 0.1, requires_grad=True),
        "kappa": torch.full((batch, 16), 2.0, requires_grad=True),
    }
    mode0 = future_target.clone()
    mode0[:, 0, 0] += 0.2
    mode1 = future_target.clone()
    mode1[:, 1:, 0] += 0.2
    mode2 = future_target.clone()
    mode2[:, :, 0] -= 0.4
    mode3 = future_target.clone()
    mode3[:, :, 0] += 0.5
    modes = [mode0, mode1, mode2, mode3]
    mode_centers = torch.stack(modes, dim=1).requires_grad_()
    future = {
        "state": mode_centers,
        "sigma": torch.full((batch, 4, future_steps, 3), 0.1, requires_grad=True),
        "kappa": torch.full((batch, 4, future_steps, 16), 2.0, requires_grad=True),
        "logits": torch.zeros(batch, 4, requires_grad=True),
    }
    output = {
        "current": current,
        "future": future,
        "balance_loss": torch.zeros((), requires_grad=True),
    }
    targets = {
        "current": current_target,
        "current_valid": torch.ones(batch, dtype=torch.bool),
        "future": future_target,
        "future_valid": torch.ones(batch, future_steps, dtype=torch.bool),
        "future_times": torch.arange(1, future_steps + 1, dtype=torch.float32)[None].repeat(batch, 1) / 30,
    }
    return output, targets


def test_motion_loss_normalizes_after_whole_trajectory_mixture_and_masks_nan():
    output, targets = _distribution_batch()
    targets["future_valid"][:, -1] = False
    targets["future"][:, -1] = float("nan")

    losses = motion_losses(output, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    safe_target = targets["future"].clone()
    safe_target[:, -1] = output["future"]["state"][:, 0, -1].detach()
    valid = targets["future_valid"]
    expected_mix = mixture_log_prob(output["future"], safe_target, valid)
    expected = -expected_mix / (51 * valid.sum(-1))
    torch.testing.assert_close(losses["future_nll"], expected.mean())

    per_step_mix = torch.logsumexp(
        output["future"]["logits"].log_softmax(-1)[:, :, None]
        + log_prob(output["future"], safe_target[:, None]),
        dim=1,
    )
    incorrect_stepwise = -(
        per_step_mix * valid.to(per_step_mix.dtype)
    ).sum(-1).div(valid.sum(-1)).mean() / 51
    assert not torch.allclose(losses["future_nll"], incorrect_stepwise)
    assert torch.isfinite(geometric_loss(output["future"]["state"], safe_target[:, None])).all()


def test_velocity_posterior_is_detached_from_mode_logits():
    output, targets = _distribution_batch()
    losses = motion_losses(output, targets)
    losses["velocity"].backward()
    assert output["future"]["state"].grad is not None
    assert output["future"]["logits"].grad is None


def test_motion_windows_respect_real_time_gaps_and_corruption_contract(tmp_path):
    manifest = _manifest(tmp_path, missing_future=True)
    manifest["train"] = [_write_sequence(
        tmp_path / "train_gap.npz", participant="P1", count=120, gap_at=70,
        missing_future=True)]
    dataset = HandLoopMotionDataset(manifest, "train", seed=31)
    assert len(dataset) > 0
    assert all(
        not (window.current_index < 70 and any(index >= 70 for index in window.future_indices))
        for window in dataset.windows
    )
    assert all(
        all(dataset.sequences[window.sequence_index].segment_id[index] ==
            dataset.sequences[window.sequence_index].segment_id[window.current_index]
            for index in window.future_indices)
        for window in dataset.windows
    )

    categories = set()
    for epoch in range(4):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            sample = dataset[index]
            categories.add(int(sample["corruption_category"]))
            assert torch.equal(sample["raw"], sample["candidate"])
            assert torch.equal(sample["raw_available"], sample["candidate_available"])
            assert sample["history"].shape == (30, 99)
            assert sample["history_available"].shape == (30, 17)
            assert sample["future"].shape == (60, 99)
    assert categories == set(range(len(CORRUPTION_PROBABILITIES)))


def test_missing_current_groups_use_only_past_values():
    history = torch.zeros(2, 99)
    history[:, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(16)
    history[0, 0] = 0.1
    history[1, 0] = 0.2
    history_available = torch.ones(2, 17, dtype=torch.bool)
    current = history[-1].clone()
    current[0] = 99.0
    current_available = torch.ones(17, dtype=torch.bool)
    current_available[0] = False
    filled = _fill_current_from_history(
        current.numpy(), current_available.numpy(), history.numpy(), history_available.numpy())
    assert filled[0] == pytest.approx(0.2)
    assert filled[0] != 99.0


def test_fixed_anchor_normalization_is_rigid_transform_equivariant_with_jitter():
    batch_size, history_length, future_steps = 2, 30, 4
    states = torch.from_numpy(_states(history_length + future_steps + 1))
    history = states[:history_length].unsqueeze(0).repeat(batch_size, 1, 1)
    raw = states[history_length].unsqueeze(0).repeat(batch_size, 1)
    current = states[history_length].unsqueeze(0).repeat(batch_size, 1)
    future = states[history_length + 1:history_length + 1 + future_steps]
    future = future.unsqueeze(0).repeat(batch_size, 1, 1)
    jitter = torch.zeros(batch_size, 51)
    jitter[:, :3] = torch.tensor([0.03, -0.02, 0.01])
    jitter[:, 3:6] = torch.tensor([0.15, -0.08, 0.05])
    raw = apply_delta(raw, jitter)
    candidate = apply_delta(raw, jitter * -0.35)
    available = torch.ones(batch_size, 17, dtype=torch.bool)
    base = {
        "history": history,
        "raw": raw,
        "candidate": candidate,
        "history_available": available[:, None].expand(-1, history_length, -1),
        "raw_available": available,
        "candidate_available": available,
        "history_times": torch.arange(-history_length, 0).float()[None].expand(batch_size, -1) / 30,
        "history_mask": torch.ones(batch_size, history_length, dtype=torch.bool),
        "dt": torch.full((batch_size,), 1 / 30),
        "handedness": torch.ones(batch_size),
        "current": current,
        "current_valid": torch.ones(batch_size, dtype=torch.bool),
        "future": future,
        "future_valid": torch.ones(batch_size, future_steps, dtype=torch.bool),
        "future_times": torch.arange(1, future_steps + 1).float()[None].expand(batch_size, -1) / 30,
    }
    axis = torch.tensor([0.4, -0.2, 0.7])
    axis = axis / axis.norm()
    cross = torch.tensor([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = torch.eye(3) + torch.sin(torch.tensor(1.1)) * cross + (1 - torch.cos(torch.tensor(1.1))) * (cross @ cross)
    translation = torch.tensor([1.2, -0.7, 2.1]).expand(batch_size, 3)
    moved = dict(base)
    for key in ("history", "raw", "candidate", "current", "future"):
        moved[key] = to_world(base[key], rotation.expand(batch_size, 3, 3), translation)

    local_base, target_base = normalize_motion_batch(base)
    local_moved, target_moved = normalize_motion_batch(moved)
    for key in ("history", "raw", "candidate"):
        torch.testing.assert_close(local_base[key], local_moved[key], atol=2e-5, rtol=2e-5)
    for key in ("current", "future"):
        torch.testing.assert_close(target_base[key], target_moved[key], atol=2e-5, rtol=2e-5)

    anchor_rotation = rotation.expand(batch_size, 3, 3)
    predicted_base = to_world(
        apply_delta(local_base["candidate"], jitter),
        *make_anchor(base["raw"], base["history"], base["raw_available"], base["history_available"]),
    )
    predicted_moved = to_world(
        apply_delta(local_moved["candidate"], jitter),
        *make_anchor(moved["raw"], moved["history"], moved["raw_available"], moved["history_available"]),
    )
    torch.testing.assert_close(
        predicted_moved,
        to_world(predicted_base, anchor_rotation, translation),
        atol=3e-5,
        rtol=3e-5,
    )


def test_motion_training_forward_receives_no_targets_or_corruption_labels(tmp_path):
    manifest = _manifest(tmp_path)
    sample = default_collate([HandLoopMotionDataset(manifest, "train", seed=4)[0]])
    output, _ = _distribution_batch(future_steps=60)

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.keys = None

        def forward(self, **kwargs):
            self.keys = set(kwargs)
            return output

    model = Recorder()
    result = motion_training_step(model, sample)
    assert torch.isfinite(result["loss"])
    assert "future" not in model.keys
    assert "current" not in model.keys
    assert "corruption_category" not in model.keys
    assert "future_times" in model.keys
    assert "future_valid" not in model.keys
    assert sample["future"].shape[1] == 60


def test_split_validation_rejects_participant_leakage_across_all_kinds():
    manifest = {
        "schema": "handloop_v2",
        "train": [{"path": "a.npz", "participant": "P1", "source": "oak", "kind": "motion"}],
        "calibration": [{"path": "b.npz", "participant": "P2", "source": "oak", "kind": "motion"}],
        "test": [{"path": "c.npz", "participant": "P1", "source": "oak", "kind": "visual"}],
    }
    with pytest.raises(ValueError, match="Participant leakage"):
        validate_manifest(manifest, required_kind="motion")


def test_cross_manifest_split_union_rejects_visual_motion_leakage():
    motion = {
        "train": [{"participant": "P1", "source": "hot3d"}],
        "calibration": [],
        "test": [],
    }
    visual = {
        "train": [],
        "calibration": [{"participant": "P1", "source": "HOT3D"}],
        "test": [],
    }
    with pytest.raises(ValueError, match="Participant leakage across manifests"):
        validate_cross_manifest_splits(motion, visual)


def test_motion_manifest_resolves_relative_archive_paths_from_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    for split in ("train", "calibration", "test"):
        for entry in manifest[split]:
            entry["path"] = Path(entry["path"]).name
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dataset = HandLoopMotionDataset(manifest_path, "train")
    assert dataset.sequences[0].path == (tmp_path / "train.npz").resolve()
    assert len(dataset) > 0


def test_batch_stream_restores_exact_epoch_order_and_cursor(tmp_path):
    dataset = HandLoopMotionDataset(_manifest(tmp_path), "train", seed=13)
    stream = HandLoopBatchStream(dataset, 1, seed=21, shuffle=True)
    first = next(stream)
    state = stream.state_dict()
    expected = next(stream)

    resumed = HandLoopBatchStream(dataset, 1, seed=21, shuffle=True)
    resumed.load_state_dict(state)
    actual = next(resumed)
    assert first["current"].shape == actual["current"].shape
    for key in ("history", "raw", "candidate", "current", "future", "future_valid"):
        torch.testing.assert_close(expected[key], actual[key])


def test_joint_motion_replay_samples_sources_with_replacement_and_equal_probability(tmp_path):
    manifest = _manifest(tmp_path)
    second = _write_sequence(
        tmp_path / "second_source.npz", participant="P4", count=80)
    second["source"] = "other_library"
    manifest["train"].append(second)
    dataset = HandLoopMotionDataset(manifest, "train")
    stream = HandLoopBatchStream(
        dataset, 1, seed=99, shuffle=True, sampling_mode="source_balanced")
    counts = {source: 0 for source in dataset.source_indices}
    selected = []
    for _ in range(200):
        next(stream)
        sample_index = stream.order[stream.offset - 1]
        selected.append(sample_index)
        window = dataset.windows[sample_index]
        source = dataset.sequences[window.sequence_index].source.casefold()
        counts[source] += 1
    assert len(set(selected)) < len(selected)
    assert all(75 < count < 125 for count in counts.values())


def test_schedule_and_checkpoint_restore_model_optimizer_and_rng(tmp_path):
    assert learning_rate_factor(0, 100, 20) == pytest.approx(0.05)
    assert learning_rate_factor(19, 100, 20) == pytest.approx(1.0)
    assert learning_rate_factor(100, 100, 20) == pytest.approx(0.1)
    assert [HandLoopTrainer.task_for_step(i) for i in range(8)] == [
        "visual", "visual", "visual", "motion", "visual", "visual", "visual", "motion"
    ]

    system = SimpleNamespace(
        hawor=nn.Linear(2, 2),
        observer=SimpleNamespace(twm=nn.Linear(2, 2), visual=nn.Linear(2, 2)),
    )
    groups = parameter_groups_for_system(system)
    joint_optimizer = make_optimizer(
        groups, {"hawor": 1e-6, "twm": 1e-5, "visual": 1e-4})
    assert [group["group_name"] for group in joint_optimizer.param_groups] == [
        "hawor", "twm", "visual"
    ]
    with pytest.raises(ValueError, match="non-empty HaWoR/TWM/visual groups"):
        make_optimizer(
            {"hawor": list(system.hawor.parameters()), "twm": list(system.observer.twm.parameters()),
             "visual": []},
            {"hawor": 1e-6, "twm": 1e-5, "visual": 1e-4},
        )

    model = nn.Linear(2, 1)
    optimizer = make_optimizer({"twm": model.parameters()}, {"twm": 1e-3})
    trainer = HandLoopTrainer(model, optimizer, total_steps=10)
    (model(torch.ones(1, 2)).square().mean()).backward()
    optimizer.step()
    trainer.global_step = 1
    manifest_hash = "train-calibration-test"
    path = tmp_path / "handloop.pt"
    save_checkpoint(
        path,
        model=model,
        trainer=trainer,
        optimizer=optimizer,
        stage="motion_pretrain",
        config={"architecture": "handloop_v2", "twm": {"width": 8}},
        split_fingerprint=manifest_hash,
    )
    expected_torch = torch.rand(3)
    expected_python = random.random()
    expected_numpy = np.random.random(3)

    restored_model = nn.Linear(2, 1)
    restored_optimizer = make_optimizer({"twm": restored_model.parameters()}, {"twm": 1e-3})
    restored_trainer = HandLoopTrainer(restored_model, restored_optimizer, total_steps=10)
    checkpoint = load_checkpoint(
        path,
        model=restored_model,
        trainer=restored_trainer,
        optimizer=restored_optimizer,
        stage="motion_pretrain",
        split_fingerprint=manifest_hash,
    )
    assert checkpoint["architecture"] == "handloop_v2"
    assert restored_trainer.global_step == 1
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)

    with pytest.raises(ValueError, match="manifest/split differs"):
        load_checkpoint(
            path,
            model=restored_model,
            trainer=restored_trainer,
            optimizer=restored_optimizer,
            stage="motion_pretrain",
            split_fingerprint="different",
        )
