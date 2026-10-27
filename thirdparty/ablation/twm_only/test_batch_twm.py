from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_twm_only as runner  # noqa: E402
from dual_loop.handloop_state import pack_state, so3_exp  # noqa: E402
from dual_loop.handloop_twm import TWMConfig, TrajectoryRestorer  # noqa: E402


class CountingTWM(nn.Module):
    """Count model calls while delegating to the real tiny TWM implementation."""

    def __init__(self, model: TrajectoryRestorer) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.batch_sizes: list[int] = []

    def forward(self, **kwargs):
        self.batch_sizes.append(len(kwargs["raw"]))
        return self.model(**kwargs)


def raw_states(count: int) -> torch.Tensor:
    positions = torch.stack((
        torch.arange(count, dtype=torch.float32) * 0.004,
        torch.sin(torch.arange(count, dtype=torch.float32) * 0.2) * 0.01,
        torch.zeros(count),
    ), dim=-1)
    angles = torch.zeros(count, 16, 3)
    angles[:, 0, 2] = torch.arange(count, dtype=torch.float32) * 0.006
    rotations = so3_exp(angles)
    return pack_state(positions, rotations)


def discontinuity_inputs():
    # Exercises normal frames, a >250ms gap, a segment change, a repeated
    # timestamp, a handedness change, and both unavailable and partial states.
    seconds = torch.tensor([
        0.0, 1 / 30, 2 / 30, 0.32, 0.32 + 1 / 30,
        0.32 + 2 / 30, 0.32 + 3 / 30, 0.32 + 3 / 30,
        0.32 + 4 / 30, 0.32 + 5 / 30, 0.32 + 6 / 30,
    ])
    segments = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    hands = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    available = torch.ones(len(seconds), 17, dtype=torch.bool)
    available[2] = False
    available[4, :2] = False
    available[6, 9:] = False
    return raw_states(len(seconds)), available, seconds, segments, hands


class BatchedRawSequenceTests(unittest.TestCase):
    def test_real_twm_batch_matches_sequential_across_resets_and_missing_states(self):
        torch.manual_seed(4021)
        base = TrajectoryRestorer(TWMConfig(
            width=8, layers=1, heads=1, state_size=4, conv_kernel=2,
            expand=1, experts=2, history_length=4, modes=2, future_steps=2,
        )).eval()
        with torch.no_grad():
            base.delta_head.weight.normal_(std=0.01)
            base.delta_head.bias.normal_(std=0.001)
        model = CountingTWM(base).eval()
        raw, available, seconds, segments, hands = discontinuity_inputs()

        with torch.inference_mode():
            expected = runner.restore_raw_sequence(
                model, raw, available, seconds, segments, hands)
        self.assertEqual(model.batch_sizes, [1] * len(raw))

        model.batch_sizes.clear()
        with torch.inference_mode():
            actual = runner.restore_raw_sequence_batched(
                model, raw, available, seconds, segments, hands, batch_size=3)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(model.batch_sizes, [3, 3, 3, 2])

    def test_real_twm_batch_matches_sequential_with_zero_history(self):
        torch.manual_seed(921)
        base = TrajectoryRestorer(TWMConfig(
            width=8, layers=1, heads=1, state_size=4, conv_kernel=2,
            expand=1, experts=2, history_length=0, modes=2, future_steps=0,
        )).eval()
        with torch.no_grad():
            base.delta_head.weight.normal_(std=0.01)
            base.delta_head.bias.normal_(std=0.001)
        model = CountingTWM(base).eval()
        raw, available, seconds, segments, hands = discontinuity_inputs()

        with torch.inference_mode():
            expected = runner.restore_raw_sequence(
                model, raw, available, seconds, segments, hands)
        model.batch_sizes.clear()
        with torch.inference_mode():
            actual = runner.restore_raw_sequence_batched(
                model, raw, available, seconds, segments, hands, batch_size=4)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(model.batch_sizes, [4, 4, 3])


if __name__ == "__main__":
    unittest.main()
