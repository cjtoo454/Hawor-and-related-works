from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from runtime import (  # noqa: E402
    INTRA_ARCHITECTURE,
    INTRA_FORMAT,
    INTRA_FORMAT_VERSION,
    raw_history_batch,
    validate_intra_checkpoint,
    verify_baseline_inputs,
)
from eval_pipeline import (  # noqa: E402
    EXPECTED_METHOD,
    EXPECTED_PROTOCOL_SHA256,
    PROTOCOL,
    build_table_row,
    _normalize_wmpjpe_method,
    _validate_protocol,
)
from dual_loop.handloop_model import HandLoopConfig, HandLoopObserver  # noqa: E402
from dual_loop.handloop_state import pack_state  # noqa: E402
from dual_loop.handloop_twm import TWMConfig, TrajectoryRestorer  # noqa: E402


def identity_state(positions: torch.Tensor) -> torch.Tensor:
    rotations = torch.eye(3).expand(len(positions), 16, 3, 3).clone()
    return pack_state(positions.float(), rotations)


class FixedVisualDelta(nn.Module):
    def __init__(self, x_delta: float = 0.05) -> None:
        super().__init__()
        self.x_delta = x_delta

    def forward(self, *, raw_camera: torch.Tensor, **kwargs) -> torch.Tensor:
        result = raw_camera.new_zeros(len(raw_camera), 51)
        result[:, 0] = self.x_delta
        return result


class CandidateTWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, history, raw, candidate, history_available, raw_available,
                candidate_available, history_times, dt, handedness, history_mask,
                future_times=None):
        self.calls += 1
        batch = len(candidate)
        return {
            "current": {
                "state": candidate,
                "sigma": candidate.new_ones(batch, 3),
                "kappa": candidate.new_ones(batch, 16),
            },
            "future": None,
        }


class IntraRuntimeTests(unittest.TestCase):
    def test_raw_history_is_causal_and_resets_on_segment_boundary(self):
        raw = identity_state(torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                                           [3., 0., 0.], [4., 0., 0.]]))
        available = torch.ones(5, 17, dtype=torch.bool)
        times = torch.arange(5, dtype=torch.float64) / 30
        segments = torch.tensor([0, 0, 0, 1, 1])
        hands = torch.ones(5, dtype=torch.long)

        history = raw_history_batch(raw, available, times, segments, hands,
                                    torch.tensor([2, 3, 4]), 3)
        self.assertEqual(tuple(history["history"].shape), (3, 3, 99))
        # The history row for frame 2 contains only frame 0 and frame 1 raw inputs.
        torch.testing.assert_close(history["history"][0, 1:, 0], torch.tensor([0., 1.]))
        self.assertEqual(history["history_mask"][0].tolist(), [False, True, True])
        # Segment 1 begins a new history; frame 3 cannot see any frame 0..2 state.
        self.assertFalse(history["history_mask"][1].any())
        self.assertEqual(history["history_mask"][2].tolist(), [False, False, True])
        torch.testing.assert_close(history["history"][2, -1], raw[3])
        self.assertTrue(torch.equal(history["history_dispersion"],
                                    torch.zeros_like(history["history_dispersion"])))
        self.assertFalse(history["dispersion_known"].any())

    def test_forward_step_reprojects_the_updated_position_exactly_twice(self):
        twm = CandidateTWM()
        observer = HandLoopObserver(
            HandLoopConfig(rounds=2, reobserve=True, twm=TWMConfig()),
            visual=FixedVisualDelta(0.05),
            twm=twm,
        ).eval()
        raw = identity_state(torch.tensor([[1., 2., 3.]]))
        empty_state = raw.new_empty(1, 0, 99)
        empty_available = torch.empty(1, 0, 17, dtype=torch.bool)
        empty_mask = torch.empty(1, 0, dtype=torch.bool)
        empty_times = raw.new_empty(1, 0)
        projected_positions = []

        def projector(state):
            projected_positions.append(state[:, :3].detach().clone())
            grid = state[:, None, :2].expand(-1, 16, -1)
            return grid, torch.ones(len(state), 16, dtype=torch.bool)

        result = observer.forward_step(
            patches=raw.new_zeros(1, 192, 8),
            raw_state=raw,
            history=empty_state,
            raw_available=torch.ones(1, 17, dtype=torch.bool),
            history_available=empty_available,
            history_mask=empty_mask,
            history_times=empty_times,
            dt=raw.new_full((1,), 1 / 30),
            handedness=torch.ones(1, dtype=torch.long),
            history_dispersion=raw.new_empty(1, 0, 17),
            dispersion_known=empty_mask,
            rotation_w2c=torch.eye(3)[None],
            translation_w2c=raw.new_zeros(1, 3),
            projector=projector,
            image_available=torch.ones(1, dtype=torch.bool),
        )
        self.assertEqual(twm.calls, 2)
        self.assertEqual(len(projected_positions), 2)
        torch.testing.assert_close(projected_positions[0], raw[:, :3])
        torch.testing.assert_close(projected_positions[1][:, 0], raw[:, 0] + 0.05)
        torch.testing.assert_close(result["state"][:, 0], raw[:, 0] + 0.1)

    def test_real_tiny_twm_runs_two_reobserving_rounds(self):
        config = TWMConfig(width=8, layers=1, heads=1, state_size=4, conv_kernel=2,
                           expand=1, experts=2, history_length=2, modes=2, future_steps=1)
        twm = TrajectoryRestorer(config)
        observer = HandLoopObserver(
            HandLoopConfig(rounds=2, reobserve=True, twm=config),
            visual=FixedVisualDelta(0.001),
            twm=twm,
        ).eval()
        calls = []
        hook = twm.register_forward_hook(lambda _module, _inputs, _output: calls.append(1))
        raw = identity_state(torch.tensor([[0.1, 0.2, 0.8]]))
        projected = []

        def projector(state):
            projected.append(state.detach().clone())
            return state[:, None, :2].expand(-1, 16, -1), torch.ones(
                len(state), 16, dtype=torch.bool)

        with torch.inference_mode():
            result = observer.forward_step(
                patches=raw.new_zeros(1, 192, 8),
                raw_state=raw,
                history=raw.new_empty(1, 0, 99),
                raw_available=torch.ones(1, 17, dtype=torch.bool),
                history_available=torch.empty(1, 0, 17, dtype=torch.bool),
                history_mask=torch.empty(1, 0, dtype=torch.bool),
                history_times=raw.new_empty(1, 0),
                dt=raw.new_full((1,), 1 / 30),
                handedness=torch.ones(1, dtype=torch.long),
                history_dispersion=raw.new_empty(1, 0, 17),
                dispersion_known=torch.empty(1, 0, dtype=torch.bool),
                rotation_w2c=torch.eye(3)[None],
                translation_w2c=raw.new_zeros(1, 3),
                projector=projector,
                image_available=torch.ones(1, dtype=torch.bool),
            )
        hook.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(projected), 2)
        self.assertEqual(result["state"].shape, (1, 99))
        self.assertTrue(torch.isfinite(result["state"]).all())

    def test_visual_checkpoint_requires_trained_state_and_base_hashes(self):
        checkpoint = {
            "architecture": INTRA_ARCHITECTURE,
            "format": INTRA_FORMAT,
            "format_version": INTRA_FORMAT_VERSION,
            "trainer": {"global_step": 1},
            "config": {"architecture": "handloop_v2", "twm": {}, "visual": {}},
            "visual": {"weight": torch.ones(1)},
            "base_twm_sha256": "a" * 64,
            "base_hawor_sha256": "b" * 64,
        }
        step, config, visual = validate_intra_checkpoint(checkpoint)
        self.assertEqual(step, 1)
        self.assertIn("twm", config)
        self.assertEqual(set(visual), {"weight"})
        checkpoint["trainer"]["global_step"] = 0
        with self.assertRaisesRegex(ValueError, "global_step=0"):
            validate_intra_checkpoint(checkpoint)

    def test_formal_row_requires_all_frozen_videos_and_complete_inference(self):
        sequences = json.loads(
            (ROOT / "diagnostics/pinch_hawor_vs_ema_10videos_20261018_final"
             / "protocol_frozen.json").read_text())["test_sequences"]
        inference = {
            sequence: {
                "method": EXPECTED_METHOD,
                "processed_frames": 100,
                "total_frames": 100,
                "baseline_raw_input_verification": {"raw_state_world": {}},
                "twm_checkpoint_sha256": "a" * 64,
                "hawor_checkpoint_sha256": "b" * 64,
                "visual_checkpoint_sha256": "c" * 64,
                "visual_global_step": 1,
            }
            for sequence in sequences
        }
        counts = {sequence: 3450 for sequence in sequences}
        counts[sequences[-1]] += 4
        wmpjpe = {
            "method": EXPECTED_METHOD,
            "videos": [{"video": sequence, "method": EXPECTED_METHOD}
                       for sequence in sequences],
            "aggregate": {
                "w_mpjpe_mm": 10., "wa_mpjpe_mm": 9., "pa_mpjpe_mm": 8.,
                "accel_m_s2": 1., "rte_percent": 2.,
            },
        }
        onset = {
            "method": EXPECTED_METHOD,
            "per_video": [{"sequence": sequence} for sequence in sequences],
            "aggregate": {"onset_mae_ms": 20., "f1": 0.5, "reference_events": 4},
            "prediction_threshold": 0.275,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "reference_type": "frozen geometry proxy",
        }
        row = build_table_row(sequences, inference, wmpjpe, onset, counts)
        self.assertEqual(row["evaluated_right_hand_frames"], 34504)
        self.assertEqual(row["method"], EXPECTED_METHOD)
        inference[sequences[0]]["processed_frames"] = 99
        with self.assertRaisesRegex(ValueError, "Incomplete videos"):
            build_table_row(sequences, inference, wmpjpe, onset, counts)

    def test_protocol_threshold_or_sequence_tampering_is_rejected_by_hash(self):
        original = json.loads(PROTOCOL.read_text())
        for key, value in (
            ("prediction_threshold", original["prediction_threshold"] + 0.01),
            ("test_sequences", original["test_sequences"][:-1]),
        ):
            changed = dict(original)
            changed[key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "protocol.json"
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    _validate_protocol(changed, path)

    def test_wmpjpe_summaries_keep_evaluator_label_and_use_intra_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video_a"
            video.mkdir()
            (root / "evaluation_summary_1videos.json").write_text(
                '{"aggregate":{},"videos":[{"video":"video_a",'
                '"method":"Dual-Loop State Reconciliation (HaWoR front-end)"}]}')
            (video / "evaluation_summary.json").write_text(
                '{"method":"Dual-Loop State Reconciliation (HaWoR front-end)"}')
            _normalize_wmpjpe_method(root, ["video_a"])
            aggregate = json.loads((root / "evaluation_summary_1videos.json").read_text())
            per_video = json.loads((video / "evaluation_summary.json").read_text())
            self.assertEqual(aggregate["method"], EXPECTED_METHOD)
            self.assertEqual(aggregate["videos"][0]["method"], EXPECTED_METHOD)
            self.assertEqual(aggregate["videos"][0]["evaluator_method"],
                             "Dual-Loop State Reconciliation (HaWoR front-end)")
            self.assertEqual(per_video["method"], EXPECTED_METHOD)
            self.assertEqual(per_video["evaluator_method"],
                             "Dual-Loop State Reconciliation (HaWoR front-end)")

    def test_archived_raw_frontend_inputs_are_bound_to_first_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "sequence"
            baseline.mkdir()
            raw = np.zeros((2, 99), dtype=np.float32)
            betas = np.ones((2, 10), dtype=np.float32)
            timestamps = np.array([1_000_000_000, 1_033_000_000], dtype=np.int64)
            seconds = timestamps.astype(np.float64) / 1e9
            segments = np.zeros(2, dtype=np.int64)
            (baseline / "try_inference.json").write_text(
                '{"method":"TWM only; raw HaWoR feedback",'
                '"checkpoint_sha256":"t","hawor_checkpoint_sha256":"h",'
                '"frontend_source_sha256":"h","track_sha256":"tracks",'
                '"oracle_assisted":true,'
                '"baseline_frontend_scope_matches_frozen_pinch_protocol":true}')
            np.savez_compressed(
                baseline / "twm_only_trace.npz",
                raw_state_world=raw, betas=betas, timestamp_ns=timestamps,
                timestamp_sec=seconds, segment_id=segments,
            )
            inputs = {
                "track_sha256": "tracks", "oracle_assisted": True,
                "ns": timestamps, "seconds": seconds, "segments": segments,
            }
            verified = verify_baseline_inputs(
                baseline, sequence="sequence", inputs=inputs, raw_world=raw,
                betas=betas, twm_checkpoint_sha256="t", hawor_checkpoint_sha256="h")
            self.assertEqual(verified["raw_state_world"]["max_abs_error"], 0.0)
            changed = raw.copy()
            changed[1, 0] = 0.01
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                verify_baseline_inputs(
                    baseline, sequence="sequence", inputs=inputs, raw_world=changed,
                    betas=betas, twm_checkpoint_sha256="t", hawor_checkpoint_sha256="h")

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "run_intra.py"), "--help"],
            check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--visual_checkpoint", result.stdout)
        self.assertIn("--twm_checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
