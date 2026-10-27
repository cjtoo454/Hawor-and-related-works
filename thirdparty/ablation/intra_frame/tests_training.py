"""Focused contract tests for the independent intra-frame visual trainer."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
sys.path[:0] = [str(HERE), str(ROOT), str(TRY), str(TRY / "scripts"),
                str(ROOT / "thirdparty" / "ablation" / "twm_only")]

from prepare_training import (  # noqa: E402
    PreparedSequence,
    calibration_samples,
    context_indices,
    history_indices,
    observation_record,
    validate_split_manifests,
)
from train_visual import (  # noqa: E402
    build_raw_history_args,
    freeze_except_visual,
    require_finite_metrics,
)


def make_sequence(
    *,
    segments: list[int] | None = None,
    times: list[float] | None = None,
    visual_valid: list[bool] | None = None,
) -> PreparedSequence:
    count = len(segments or [0] * 4)
    segments = segments or [0] * count
    times = times or [index / 30.0 for index in range(count)]
    visual_valid = visual_valid or [True] * count
    raw = np.zeros((count, 99), dtype=np.float32)
    raw[:, 0] = np.arange(count, dtype=np.float32) + 1
    return PreparedSequence(
        sequence="P0002_unit",
        split="train",
        archive_path=Path("unused.npz"),
        image_paths=np.asarray([f"frame-{index}.jpg" for index in range(count)]),
        boxes=np.tile(np.asarray([[0, 0, 32, 32]], dtype=np.float32), (count, 1)),
        image_size_wh=np.asarray([32, 32], dtype=np.float32),
        focal_length=np.full(count, 50, dtype=np.float32),
        timestamp_sec=np.asarray(times, dtype=np.float64),
        segment_id=np.asarray(segments, dtype=np.int64),
        rotation_w2c=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        translation_w2c=np.zeros((count, 3), dtype=np.float32),
        visual_valid=np.asarray(visual_valid, dtype=bool),
        gt_valid=np.ones(count, dtype=bool),
        visual_legacy=np.zeros((count, 109), dtype=np.float32),
        target_legacy=np.zeros((count, 109), dtype=np.float32),
        raw_world=raw,
    )


class TrainingDataContractTests(unittest.TestCase):
    def test_contexts_stay_inside_the_raw_slam_segment(self) -> None:
        sequence = make_sequence(segments=[0] * 17 + [1] * 3)
        ids, current = context_indices(sequence, 18)
        self.assertEqual(current, 1)
        self.assertEqual(ids[:3].tolist(), [17, 18, 19])
        self.assertTrue(np.all(ids >= 17))
        self.assertEqual(ids[-1], 19)

    def test_observation_record_has_no_gt_fields(self) -> None:
        sequence = make_sequence()
        record = observation_record(sequence, 2)
        self.assertEqual(record["context_indices"].tolist(), list(range(4)) + [3] * 12)
        self.assertFalse({"target_legacy", "target_state", "gt_valid", "gt_state"} & record.keys())

    def test_history_uses_only_prior_raw_states_and_marks_padding(self) -> None:
        sequence = make_sequence()
        current_raw = torch.zeros(1, 99)
        current_raw[:, 0] = 99
        available = torch.ones(1, 17, dtype=torch.bool)
        first = build_raw_history_args(
            [(sequence, 2)], current_raw, available, torch.device("cpu"), history_length=3)
        sequence.target_legacy.fill(1234)
        second = build_raw_history_args(
            [(sequence, 2)], current_raw, available, torch.device("cpu"), history_length=3)
        self.assertEqual(first["history"].shape, (1, 3, 99))
        self.assertEqual(first["history_mask"].tolist(), [[False, True, True]])
        self.assertEqual(first["history"][0, 1:, 0].tolist(), [1.0, 2.0])
        torch.testing.assert_close(
            first["history_times"][0, 1:], torch.tensor([-1 / 15, -1 / 30]))
        torch.testing.assert_close(first["dt"], torch.tensor([1 / 30]))
        self.assertTrue(torch.equal(first["history"], second["history"]))
        self.assertTrue(torch.equal(first["history_mask"], second["history_mask"]))
        self.assertFalse(first["dispersion_known"].any())
        self.assertFalse(first["history_dispersion"].any())
        self.assertFalse((first["history"][..., 0] == 99).any())

    def test_history_resets_at_segment_boundary(self) -> None:
        sequence = make_sequence(segments=[0, 0, 1, 1])
        self.assertEqual(history_indices(sequence, 3, 3), [2])
        current_raw = torch.zeros(1, 99)
        available = torch.ones(1, 17, dtype=torch.bool)
        history = build_raw_history_args(
            [(sequence, 3)], current_raw, available, torch.device("cpu"), history_length=3)
        self.assertEqual(history["history_mask"].tolist(), [[False, False, True]])
        self.assertEqual(history["history"][0, -1, 0].item(), 3.0)

    def test_calibration_uses_gt_and_raw_visual_valid_frames(self) -> None:
        sequence = make_sequence(visual_valid=[False, True, False, True])
        selected = calibration_samples([sequence])
        self.assertEqual(selected, [(sequence, 3)])

    def test_nonfinite_calibration_score_is_rejected(self) -> None:
        require_finite_metrics({"loss": torch.tensor(1.0), "joint_m": 0.02}, "unit test")
        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            require_finite_metrics({"loss": torch.tensor(float("nan"))}, "unit test")

    def test_participant_split_validation_rejects_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = {"train": ["P0002_train"], "calibration": ["P0015_cal"]}
            export = dict(split)
            protocol = {"test_sequences": ["P0001_test"]}
            paths = [root / name for name in ("split.json", "export.json", "protocol.json")]
            for path, value in zip(paths, (split, export, protocol)):
                path.write_text(json.dumps(value), encoding="utf-8")
            train, calibration, _ = validate_split_manifests(
                *paths, verify_frozen_protocol=False)
            self.assertEqual(train, ["P0002_train"])
            self.assertEqual(calibration, ["P0015_cal"])
            split["calibration"] = ["P0002_calibration"]
            export["calibration"] = list(split["calibration"])
            paths[0].write_text(json.dumps(split), encoding="utf-8")
            paths[1].write_text(json.dumps(export), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Participant leakage"):
                validate_split_manifests(*paths, verify_frozen_protocol=False)


class TrainableScopeTests(unittest.TestCase):
    def test_only_visual_parameters_update(self) -> None:
        system = nn.Module()
        system.hawor = nn.Linear(3, 3)
        system.geometry = nn.Linear(3, 3)
        system.observer = nn.Module()
        system.observer.visual = nn.Linear(3, 1)
        system.observer.twm = nn.Linear(3, 3)
        system.observer.config = SimpleNamespace(
            rounds=2, reobserve=True, twm=SimpleNamespace(history_length=30))
        hawor_before = {key: value.detach().clone() for key, value in system.hawor.state_dict().items()}
        twm_before = {key: value.detach().clone() for key, value in system.observer.twm.state_dict().items()}
        geometry_before = {key: value.detach().clone() for key, value in system.geometry.state_dict().items()}
        freeze_except_visual(system)
        optimizer = torch.optim.SGD(system.observer.visual.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        system.observer.visual(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        self.assertTrue(any(parameter.grad is not None for parameter in system.observer.visual.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in system.hawor.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in system.observer.twm.parameters()))
        self.assertEqual(hawor_before.keys(), system.hawor.state_dict().keys())
        self.assertEqual(twm_before.keys(), system.observer.twm.state_dict().keys())
        for key, value in hawor_before.items():
            self.assertTrue(torch.equal(value, system.hawor.state_dict()[key]))
        for key, value in twm_before.items():
            self.assertTrue(torch.equal(value, system.observer.twm.state_dict()[key]))
        for key, value in geometry_before.items():
            self.assertTrue(torch.equal(value, system.geometry.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
