"""CPU regression tests for centered intra-frame feedback."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "try"))

from dual_loop.handloop_model import HandLoopConfig, HandLoopObserver  # noqa: E402
from dual_loop.handloop_state import (  # noqa: E402
    apply_delta, make_anchor, so3_log, to_local, to_world, unpack_state,
    pack_state,
)
from feedback_v4 import (  # noqa: E402
    INTRA_ARCHITECTURE, bound_visual_delta, install_feedback_v4,
)


def _state(position: tuple[float, float, float]) -> torch.Tensor:
    point = torch.tensor(position, dtype=torch.float32).reshape(1, 3)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 16, 3, 3).clone()
    return pack_state(point, rotations)


def _inputs(*, raw_available: bool = True, image_available: bool = True):
    history = torch.cat((_state((0.1, 0.2, 0.3)), _state((0.2, 0.1, 0.4))), dim=0)[None]
    return {
        "patches": torch.zeros(1, 192, 8),
        "raw_state": _state((0.6, -0.3, 0.8)),
        "history": history,
        "raw_available": torch.full((1, 17), raw_available, dtype=torch.bool),
        "history_available": torch.ones(1, 2, 17, dtype=torch.bool),
        "history_mask": torch.ones(1, 2, dtype=torch.bool),
        "history_times": torch.tensor([[-0.06, -0.03]]),
        "dt": torch.tensor([1 / 30]),
        "handedness": torch.zeros(1, dtype=torch.long),
        "history_dispersion": torch.zeros(1, 2, 17),
        "dispersion_known": torch.zeros(1, 2, dtype=torch.bool),
        "rotation_w2c": torch.eye(3).reshape(1, 3, 3),
        "translation_w2c": torch.zeros(1, 3),
        "image_available": torch.tensor([image_available]),
    }


class _Visual(nn.Module):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.first = nn.Parameter(first.clone())
        self.second = nn.Parameter(second.clone())
        self.calls = 0
        self.observed = []
        self.register_buffer("delta_scale", torch.ones(51))

    def forward(self, **kwargs):
        self.observed.append(kwargs["candidate_camera"])
        delta = self.first if self.calls == 0 else self.second
        self.calls += 1
        return delta[None].expand(kwargs["candidate_camera"].shape[0], -1)


class _NonlinearTWM(nn.Module):
    """Differentiable, nonlinear candidate response with noncommuting rotation."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.4))
        self.candidates = []
        self.raw_inputs = []
        self.history_inputs = []

    def forward(self, history, raw, candidate, *args):
        self.candidates.append(candidate)
        self.raw_inputs.append(raw.detach().clone())
        self.history_inputs.append(history.detach().clone())
        position, rotations = unpack_state(candidate)
        residual = torch.zeros(candidate.shape[0], 51, device=candidate.device, dtype=candidate.dtype)
        residual[:, 0] = 0.015 + self.gain * position[:, 0].square()
        residual[:, 1] = -0.007 + self.gain * position[:, 1].square()
        residual[:, 5] = 0.13 + self.gain * rotations[:, 0, 0, 1].square()
        state = apply_delta(candidate, residual)
        return {
            "current": {
                "state": state,
                "sigma": state.new_full((state.shape[0], 3), 0.1),
                "kappa": state.new_ones(state.shape[0], 16),
            },
            "future": None,
            "balance_loss": state.new_zeros(()),
        }


class _System(nn.Module):
    def __init__(self, visual, twm):
        super().__init__()
        self.hawor = nn.Linear(1, 1)
        self.observer = HandLoopObserver(
            HandLoopConfig(rounds=2, reobserve=True), visual=visual, twm=twm)


def _projector(valid=None):
    def project(state):
        grid = state[:, None, :2].expand(-1, 16, -1)
        validity = torch.ones(state.shape[0], 16, dtype=torch.bool, device=state.device)
        if valid is not None:
            validity &= valid.to(device=state.device)
        return grid, validity

    return project


def _run(visual, twm, *, inputs=None, valid=None):
    system = _System(visual, twm)
    keys = set(visual.state_dict())
    observer = install_feedback_v4(system)
    result = observer.forward_step(**(_inputs() if inputs is None else inputs),
                                   projector=_projector(valid))
    assert set(visual.state_dict()) == keys
    return observer, result, system


def test_zero_visual_is_one_twm_update_and_fallback_keeps_raw_history():
    zero = torch.zeros(51)
    visual, twm = _Visual(zero, zero), _NonlinearTWM()
    inputs = _inputs(raw_available=False)
    raw_before, history_before = inputs["raw_state"].clone(), inputs["history"].clone()
    observer, output, system = _run(visual, twm, inputs=inputs)

    anchor_r, anchor_p = make_anchor(
        inputs["history"][:, -1], inputs["history"], inputs["history_available"][:, -1],
        inputs["history_available"])
    initial = to_local(inputs["history"][:, -1], anchor_r, anchor_p)
    torch.testing.assert_close(twm.candidates[0], initial)
    assert len(twm.candidates) == 3
    torch.testing.assert_close(twm.candidates[1], twm.candidates[2])
    assert torch.equal(output["current"]["state"], output["rounds"][0]["state"])
    torch.testing.assert_close(output["state"], to_world(
        output["current"]["state"], output["anchor_rotation"], output["anchor_position"]))
    assert observer.architecture == INTRA_ARCHITECTURE
    assert observer.config.rounds == 2 and observer.config.reobserve
    assert observer.feedback_metadata["distribution_nll_valid"] is False
    assert output["distribution_semantics"]["nll_valid"] is False
    torch.testing.assert_close(inputs["raw_state"], raw_before)
    torch.testing.assert_close(inputs["history"], history_before)
    assert all(torch.equal(item, twm.raw_inputs[0]) for item in twm.raw_inputs)
    assert all(torch.equal(item, twm.history_inputs[0]) for item in twm.history_inputs)
    assert not any(p.requires_grad for p in observer.twm.parameters())
    assert not any(p.requires_grad for p in system.hawor.parameters())


def test_second_round_zero_is_exact_identity_with_reference_gradient_cancellation():
    first, second = torch.zeros(51), torch.zeros(51)
    first[0] = 0.0003
    visual, twm = _Visual(first, second), _NonlinearTWM()
    _, output, _ = _run(visual, twm)
    s1 = output["rounds"][0]["state"]
    s2 = output["current"]["state"]
    assert torch.equal(s2, s1)
    assert not torch.equal(twm.candidates[0], twm.candidates[1])
    torch.testing.assert_close(twm.candidates[1], twm.candidates[2])
    grad_s1 = torch.autograd.grad(s1[:, 0].sum(), visual.first, retain_graph=True)[0]
    grad_s2 = torch.autograd.grad(s2[:, 0].sum(), visual.first, retain_graph=True)[0]
    torch.testing.assert_close(grad_s2, grad_s1, atol=1e-5, rtol=1e-4)
    grad_second = torch.autograd.grad(s2[:, 0].sum(), visual.second)[0]
    assert torch.isfinite(grad_second).all() and grad_second[0].abs() > 0
    assert twm.gain.grad is None and not twm.gain.requires_grad


def test_nonzero_second_round_uses_left_so3_difference_and_current_state():
    first, second = torch.zeros(51), torch.zeros(51)
    first[3] = 0.003
    second[0] = 0.00025
    second[4] = 0.004
    second[6] = -0.008
    visual, twm = _Visual(first, second), _NonlinearTWM()
    _, output, _ = _run(visual, twm)
    s1 = output["rounds"][0]["state"]
    candidate = output["candidate_distribution"]["state"]
    reference = output["reference_distribution"]["state"]
    pplus, rplus = unpack_state(candidate)
    pref, rref = unpack_state(reference)
    ps1, rs1 = unpack_state(s1)
    p2, r2 = unpack_state(output["current"]["state"])
    torch.testing.assert_close(p2, ps1 + pplus - pref, atol=1e-6, rtol=1e-5)
    expected_rotations = (rplus @ rref.transpose(-1, -2)) @ rs1
    torch.testing.assert_close(r2, expected_rotations, atol=1e-5, rtol=1e-5)
    assert so3_log(rplus[:, 0] @ rref[:, 0].transpose(-1, -2)).norm() > 0
    assert output["current"]["state"] is not candidate
    torch.testing.assert_close(output["state"], to_world(
        output["current"]["state"], output["anchor_rotation"], output["anchor_position"]))
    torch.testing.assert_close(visual.observed[1], output["projection_states"][1])


def test_both_visual_rounds_receive_gradient_through_frozen_twm():
    first, second = torch.zeros(51), torch.zeros(51)
    first[0], second[0] = 0.0002, 0.00015
    visual, twm = _Visual(first, second), _NonlinearTWM()
    _, output, _ = _run(visual, twm)
    output["current"]["state"][:, 0].sum().backward()
    assert visual.first.grad is not None and visual.first.grad[0].abs() > 0
    assert visual.second.grad is not None and visual.second.grad[0].abs() > 0
    assert torch.isfinite(visual.first.grad).all()
    assert torch.isfinite(visual.second.grad).all()
    assert twm.gain.grad is None


def test_invalid_projection_and_missing_image_block_their_corrections():
    full = torch.full((51,), 0.01)
    valid = torch.ones(16, dtype=torch.bool)
    valid[0] = False
    valid[3] = False
    visual, twm = _Visual(full, full), _NonlinearTWM()
    _, output, _ = _run(visual, twm, valid=valid)
    for delta in output["visual_deltas_camera"]:
        assert torch.equal(delta[:, :6], torch.zeros_like(delta[:, :6]))
        assert torch.equal(delta[:, 12:15], torch.zeros_like(delta[:, 12:15]))
        assert delta[:, 6:9].abs().sum() > 0

    visual, twm = _Visual(full, full), _NonlinearTWM()
    _, output, _ = _run(visual, twm, inputs=_inputs(image_available=False))
    assert all(torch.count_nonzero(delta) == 0 for delta in output["visual_deltas_camera"])
    assert torch.equal(output["current"]["state"], output["rounds"][0]["state"])


def test_nonfinite_valid_delta_raises_but_masked_nonfinite_is_ignored():
    nan_delta = torch.zeros(51)
    nan_delta[0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        _run(_Visual(nan_delta, torch.zeros(51)), _NonlinearTWM())
    invalid_root = torch.ones(16, dtype=torch.bool)
    invalid_root[0] = False
    _, output, _ = _run(_Visual(nan_delta, torch.zeros(51)), _NonlinearTWM(),
                        valid=invalid_root)
    assert torch.isfinite(output["state"]).all()


def test_bound_scale_small_signal_gradient_and_future_rejection():
    raw = torch.zeros(1, 51, requires_grad=True)
    bounded = bound_visual_delta(raw)
    bounded[0, 0].backward()
    torch.testing.assert_close(raw.grad[0, 0], torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    large = bound_visual_delta(torch.full((1, 51), 1000.0))
    assert large[:, :3].norm(dim=-1).max() <= 0.001 + 1e-7
    assert large[:, 3:6].norm(dim=-1).max() <= 0.01 + 1e-7
    assert large[:, 6:].reshape(1, 15, 3).norm(dim=-1).max() <= 0.05 + 1e-7
    visual, twm = _Visual(torch.zeros(51), torch.zeros(51)), _NonlinearTWM()
    system = _System(visual, twm)
    observer = install_feedback_v4(system)
    torch.testing.assert_close(visual.delta_scale[:3], torch.full((3,), 0.001))
    torch.testing.assert_close(visual.delta_scale[3:6], torch.full((3,), 0.01))
    torch.testing.assert_close(visual.delta_scale[6:], torch.full((45,), 0.05))
    assert visual.first.requires_grad and visual.second.requires_grad
    with pytest.raises(ValueError, match="future_times"):
        observer.forward_step(**_inputs(), projector=_projector(), future_times=torch.ones(1, 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
