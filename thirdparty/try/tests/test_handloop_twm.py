from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from dual_loop.handloop_state import pack_state, so3_exp
from dual_loop.handloop_twm import (
    TWMConfig,
    TrajectoryRestorer,
    _history_intervals,
)


def _state(batch: int, time: int | None = None) -> torch.Tensor:
    leading = (batch,) if time is None else (batch, time)
    position = torch.randn(*leading, 3) * 0.1
    tangent = torch.randn(*leading, 16, 3) * 0.2
    return pack_state(position, so3_exp(tangent))


def _inputs(batch: int = 2, history_length: int = 3) -> dict[str, torch.Tensor]:
    return {
        "history": _state(batch, history_length),
        "raw": _state(batch),
        "candidate": _state(batch),
        "history_available": torch.ones(batch, history_length, 17, dtype=torch.bool),
        "raw_available": torch.ones(batch, 17, dtype=torch.bool),
        "candidate_available": torch.ones(batch, 17, dtype=torch.bool),
        "history_times": torch.tensor(
            [[-0.08, -0.04, -0.02][:history_length]] * batch,
            dtype=torch.float32,
        ),
        "dt": torch.full((batch,), 0.02),
        "handedness": torch.arange(batch) % 2,
        "history_mask": torch.ones(batch, history_length, dtype=torch.bool),
    }


def _model() -> TrajectoryRestorer:
    torch.manual_seed(7)
    return TrajectoryRestorer(
        TWMConfig(
            width=16,
            layers=1,
            heads=4,
            state_size=4,
            conv_kernel=3,
            expand=1,
            experts=3,
            history_length=4,
            modes=2,
            future_steps=4,
        )
    )


def test_padding_slots_do_not_change_current_prediction() -> None:
    model = _model().eval()
    inputs = _inputs()
    with torch.no_grad():
        expected = model(**inputs)["current"]

        padded = {
            key: value.clone()
            for key, value in inputs.items()
        }
        pad_state = _state(2, 1)
        padded["history"] = torch.cat((pad_state, inputs["history"]), dim=1)
        padded["history_available"] = torch.cat(
            (torch.zeros(2, 1, 17, dtype=torch.bool), inputs["history_available"]),
            dim=1,
        )
        padded["history_times"] = torch.cat(
            (torch.full((2, 1), 0.7), inputs["history_times"]), dim=1
        )
        padded["history_mask"] = torch.cat(
            (torch.zeros(2, 1, dtype=torch.bool), inputs["history_mask"]), dim=1
        )
        actual = model(**padded)["current"]

    for key in ("state", "sigma", "kappa"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-6)


def test_actual_frame_interval_changes_conditioned_uncertainty() -> None:
    model = _model().eval()
    inputs = _inputs()
    with torch.no_grad():
        short = model(**inputs)["current"]
        inputs["dt"] = torch.full_like(inputs["dt"], 0.35)
        long = model(**inputs)["current"]

    assert not torch.allclose(short["sigma"], long["sigma"], atol=1e-7, rtol=1e-7)
    assert torch.all(short["sigma"] > 0.001)
    assert torch.all(short["sigma"] < 1.0)
    assert torch.all(short["kappa"] >= 0.1)
    assert torch.all(short["kappa"] <= 10000.0)


def test_history_intervals_follow_real_preceding_samples() -> None:
    times = torch.tensor([[-0.12, 0.0, -0.10, -0.04, 0.0]])
    mask = torch.tensor([[True, False, True, True, False]])
    intervals = _history_intervals(times, mask, torch.tensor([0.04]))

    torch.testing.assert_close(
        intervals,
        torch.tensor([[0.02, 0.0, 0.02, 0.06, 0.0]]),
    )


def test_future_modes_have_shapes_and_backpropagate_through_current_center() -> None:
    model = _model()
    inputs = _inputs()
    future_times = torch.tensor([[0.03, 0.07, 0.12, 0.2]] * 2)
    inputs["candidate"].requires_grad_()
    outputs = model(**inputs, future_times=future_times)
    current = outputs["current"]
    future = outputs["future"]

    assert current["state"].shape == (2, 99)
    assert current["sigma"].shape == (2, 3)
    assert current["kappa"].shape == (2, 16)
    assert future is not None
    assert future["state"].shape == (2, 2, 4, 99)
    assert future["sigma"].shape == (2, 2, 4, 3)
    assert future["kappa"].shape == (2, 2, 4, 16)
    assert future["logits"].shape == (2, 2)
    assert outputs["balance_loss"].ndim == 0

    current["state"].retain_grad()
    future["state"].square().mean().backward()
    assert current["state"].grad is not None
    assert torch.isfinite(current["state"].grad).all()
    assert current["state"].grad.abs().sum() > 0


def test_no_evidence_keeps_maximum_uncertainty_and_future_is_optional() -> None:
    model = _model().eval()
    inputs = _inputs(history_length=0)
    inputs["history_available"] = torch.zeros(2, 0, 17, dtype=torch.bool)
    inputs["history_times"] = torch.zeros(2, 0)
    inputs["history_mask"] = torch.zeros(2, 0, dtype=torch.bool)
    inputs["raw_available"].zero_()
    inputs["candidate_available"].zero_()
    output = model(**inputs)

    assert output["future"] is None
    torch.testing.assert_close(output["current"]["sigma"], torch.ones(2, 3))
    torch.testing.assert_close(output["current"]["kappa"], torch.full((2, 16), 0.1))
    assert "future_labels" not in inspect.signature(model.forward).parameters
