from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_twm_onset  # noqa: E402
import run_pipeline  # noqa: E402
import run_twm_only as runner  # noqa: E402
from dual_loop.handloop_state import pack_state  # noqa: E402
from dual_loop.handloop_training import HandLoopTrainer, save_checkpoint  # noqa: E402
from dual_loop.handloop_twm import TWMConfig, TrajectoryRestorer  # noqa: E402


def state(position, root=None):
    batch = len(position)
    rotations = torch.eye(3).expand(batch, 16, 3, 3).clone()
    if root is not None:
        rotations[:, 0] = root
    return pack_state(torch.as_tensor(position, dtype=torch.float32), rotations)


class RawHistoryProbe(nn.Module):
    def __init__(self, history_length=4, offset=0.2):
        super().__init__()
        self.config = SimpleNamespace(history_length=history_length)
        self.offset = offset
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({key: value.detach().clone() for key, value in kwargs.items()
                           if torch.is_tensor(value)})
        restored = kwargs["candidate"].clone()
        restored[:, 0] += self.offset
        return {"current": {"state": restored}}


class TWMOnlyTests(unittest.TestCase):
    def test_one_call_per_frame_and_history_commits_raw(self):
        angle = torch.tensor(torch.pi / 2)
        root = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        raw = state([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], root)
        model = RawHistoryProbe()
        restored = runner.restore_raw_sequence(
            model,
            raw,
            torch.ones(2, 17, dtype=torch.bool),
            torch.tensor([0.0, 1 / 30]),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
        )
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1]["history"].shape, (1, 1, 99))
        # The first model output is +0.2m in the rotated anchor's X axis.
        torch.testing.assert_close(restored[0, :3], torch.tensor([0.0, 0.2, 0.0]), atol=1e-5, rtol=0)
        # Frame 2 sees the first raw pose at local origin, not the restored output.
        torch.testing.assert_close(model.calls[1]["history"][0, 0, :3], torch.zeros(3), atol=1e-6, rtol=0)

    def test_zero_history_is_supported_without_negative_zero_slice(self):
        raw = state([[0.0, 0.0, 0.0]])
        model = RawHistoryProbe(history_length=0)
        restored = runner.restore_raw_sequence(
            model, raw, torch.ones(1, 17, dtype=torch.bool), torch.tensor([1.0]),
            torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long))
        self.assertEqual(restored.shape, (1, 99))
        self.assertEqual(model.calls[0]["history"].shape, (1, 0, 99))

    def test_real_training_checkpoint_loads_restricted_and_runs_sequence(self):
        config = TWMConfig(width=8, layers=1, heads=1, state_size=4, conv_kernel=2,
                           expand=1, experts=2, history_length=2, modes=2, future_steps=3)
        twm = TrajectoryRestorer(config)
        config_dict = {"architecture": "handloop_v2", "twm": asdict(config), "visual": {}}

        class JointContainer(nn.Module):
            def __init__(self, child):
                super().__init__()
                self.hawor = nn.Linear(2, 2)
                self.observer = nn.Module()
                self.observer.twm = child

        cases = (
            ("motion_pretrain", twm),
            ("joint_visual", JointContainer(TrajectoryRestorer(config))),
        )
        with tempfile.TemporaryDirectory() as directory:
            for stage, model in cases:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                trainer = HandLoopTrainer(model, optimizer, total_steps=2,
                                          warmup_steps=0, global_step=1)
                checkpoint_path = Path(directory) / f"{stage}.pt"
                save_checkpoint(
                    checkpoint_path, model=model, trainer=trainer, optimizer=optimizer,
                    stage=stage, config=config_dict, split_fingerprint="fixture")
                loaded, checkpoint = runner.load_twm(checkpoint_path, "cpu")
                self.assertEqual(checkpoint["stage"], stage)
                source_model = model if stage == "motion_pretrain" else model.observer.twm
                source = source_model.state_dict()
                for key, value in source.items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)
                if stage == "joint_visual":
                    frontend_state = runner.frontend_state_from_checkpoint(
                        runner.load_checkpoint_restricted(checkpoint_path))
                    self.assertEqual(set(frontend_state), {"weight", "bias"})
                    nn.Linear(2, 2).load_state_dict(frontend_state, strict=True)
                prediction = runner.restore_raw_sequence(
                    loaded,
                    state([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]),
                    torch.ones(3, 17, dtype=torch.bool),
                    torch.tensor([0.0, 1 / 30, 2 / 30]),
                    torch.zeros(3, dtype=torch.long),
                    torch.ones(3, dtype=torch.long),
                )
                self.assertEqual(prediction.shape, (3, 99))
                self.assertTrue(torch.isfinite(prediction).all())

    def test_untrained_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "global_step=0"):
            runner.validate_trained_checkpoint({
                "format": "handloop_training", "format_version": 2,
                "trainer": {"global_step": 0},
            })

    def test_onset_scorer_uses_shared_matching_and_returns_mae(self):
        times = np.arange(12, dtype=np.float64) / 30

        def joints(onset):
            value = np.zeros((len(times), 21, 3), dtype=np.float32)
            value[:, 9, 1] = 1.0
            value[:, 8, 0] = 0.4
            value[onset:onset + 3, 8, 0] = 0.1
            return value

        result = eval_twm_onset.score_twm_onsets(
            joints(3), joints(2), times, np.ones(len(times), bool), np.ones(len(times), bool),
            prediction_threshold=0.2, reference_threshold=0.2, min_frames=3,
            onset_tolerance_sec=0.2)
        self.assertEqual(result["metrics"]["tp"], 1)
        self.assertAlmostEqual(result["metrics"]["onset_mae_ms"], 1000 / 30, places=4)

    def test_table_row_requires_matching_complete_video_sets(self):
        inference = {"sequence_a": {
            "processed_frames": 20, "total_frames": 20,
            "checkpoint_stage": "motion_pretrain", "checkpoint_sha256": "abc",
            "hawor_checkpoint_sha256": "def", "frontend_source_sha256": "def",
            "frontend_weights_source": "fixture",
            "checkpoint_global_step": 1,
            "method": "TWM only; raw HaWoR feedback", "oracle_assisted": True,
            "baseline_frontend_scope_matches_frozen_pinch_protocol": True,
        }}
        wmpjpe = {"videos": [{"video": "sequence_a"}], "aggregate": {
            "w_mpjpe_mm": 10.0, "wa_mpjpe_mm": 9.0, "pa_mpjpe_mm": 8.0,
            "accel_m_s2": 1.0, "rte_percent": 2.0,
        }}
        onset = {"per_video": [{"sequence": "sequence_a"}], "aggregate": {
            "onset_mae_ms": 33.3, "f1": 0.5, "reference_events": 4,
        }}
        row = run_pipeline.build_table_row(["sequence_a"], inference, wmpjpe, onset,
                                           frozen_videos=["sequence_a"])
        self.assertEqual(row["onset_mae_ms"], 33.3)
        self.assertIn("W-MPJPE (mm)", run_pipeline.render_table_row(row))
        with self.assertRaisesRegex(ValueError, "Video-set mismatch"):
            run_pipeline.build_table_row(["sequence_a", "sequence_b"], inference, wmpjpe, onset,
                                         frozen_videos=["sequence_a", "sequence_b"])
        inference["sequence_a"]["processed_frames"] = 19
        with self.assertRaisesRegex(ValueError, "Incomplete inference"):
            run_pipeline.build_table_row(["sequence_a"], inference, wmpjpe, onset,
                                         frozen_videos=["sequence_a"])
        inference["sequence_a"]["processed_frames"] = 20
        onset["aggregate"]["onset_mae_ms"] = None
        with self.assertRaisesRegex(ValueError, "finite onset_mae_ms"):
            run_pipeline.build_table_row(["sequence_a"], inference, wmpjpe, onset,
                                         frozen_videos=["sequence_a"])

    def test_wmpjpe_details_are_bound_to_prediction_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sequence_a"
            sequence.mkdir()
            prediction = sequence / "try_predictions.npz"
            details = sequence / "evaluation_details.npz"
            metadata = sequence / "try_inference.json"
            prediction.write_bytes(b"twm prediction")
            details.write_bytes(b"wmpjpe details")
            metadata.write_text(json.dumps({
                "method": "TWM only; raw HaWoR feedback", "checkpoint_sha256": "abc",
            }))
            run_pipeline.bind_evaluation_sources(root, ["sequence_a"])
            _, manifest_path = eval_twm_onset.verify_evaluation_source(details, prediction, metadata)
            self.assertTrue(manifest_path.is_file())
            prediction.write_bytes(b"changed prediction")
            with self.assertRaisesRegex(ValueError, "source binding mismatch"):
                eval_twm_onset.verify_evaluation_source(details, prediction, metadata)

    def test_cli_help_for_all_entry_points(self):
        for name in ("run_twm_only.py", "eval_twm_onset.py", "run_pipeline.py"):
            result = subprocess.run(
                [sys.executable, str(HERE / name), "--help"],
                check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
