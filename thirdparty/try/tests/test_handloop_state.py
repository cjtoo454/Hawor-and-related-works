from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


DUAL_LOOP = Path(__file__).resolve().parents[1]
if str(DUAL_LOOP) not in sys.path:
    sys.path.insert(0, str(DUAL_LOOP))

from dual_loop.handloop_probability import (  # noqa: E402
    dispersion,
    geometric_loss,
    log_prob,
    matrix_fisher_log_normalizer,
    mixture_log_prob,
)
from dual_loop.handloop_state import (  # noqa: E402
    apply_delta,
    legacy_to_state,
    make_anchor,
    pack_state,
    project_state,
    so3_exp,
    so3_log,
    state_to_legacy,
    to_local,
    to_world,
    unpack_state,
)
from dual_loop.state import pack_state as pack_legacy_state  # noqa: E402
from dual_loop.state import unpack_state as unpack_legacy_state  # noqa: E402


def _random_rotations(*leading: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    vector = torch.randn(*leading, 3, dtype=dtype) * 0.7
    return so3_exp(vector)


def _state(batch: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    p = torch.randn(batch, 3, dtype=dtype)
    rotations = _random_rotations(batch, 16, dtype=dtype)
    return pack_state(p, rotations)


def test_anchor_transform_is_equivariant_over_time_and_uses_newest_valid_history() -> None:
    torch.manual_seed(4)
    world = torch.randn(2, 4, 3, dtype=torch.float64)
    roots = _random_rotations(2, 4, 16, dtype=torch.float64)
    states = pack_state(world, roots)
    anchor_r = _random_rotations(2, dtype=torch.float64)
    anchor_p = torch.randn(2, 3, dtype=torch.float64)

    local = to_local(states, anchor_r, anchor_p)
    recovered = to_world(local, anchor_r, anchor_p)
    torch.testing.assert_close(recovered, project_state(states), atol=1e-10, rtol=1e-10)

    raw = states[:, 0].detach().requires_grad_()
    history = states.detach().clone().requires_grad_()
    available = torch.zeros(2, 4, 17, dtype=torch.bool)
    available[0, 1:, :2] = True
    raw_available = torch.ones(2, 17, dtype=torch.bool)
    raw_r, raw_p = make_anchor(raw, history, raw_available, available)
    _, history_r = unpack_state(history)
    _, raw_rotations = unpack_state(raw)
    torch.testing.assert_close(raw_r[0], history_r[0, 3, 0])
    torch.testing.assert_close(raw_p[0], history[0, 3, :3])
    torch.testing.assert_close(raw_r[1], raw_rotations[1, 0])
    torch.testing.assert_close(raw_p[1], raw[1, :3])
    assert not raw_r.requires_grad and not raw_p.requires_grad


def test_so3_exp_log_and_delta_have_finite_near_zero_gradients_and_left_compose() -> None:
    vector = torch.tensor([1e-8, -2e-8, 3e-8], dtype=torch.float64, requires_grad=True)
    recovered = so3_log(so3_exp(vector))
    torch.testing.assert_close(recovered, vector, atol=1e-14, rtol=1e-7)
    recovered.sum().backward()
    assert vector.grad is not None and torch.isfinite(vector.grad).all()
    torch.testing.assert_close(vector.grad, torch.ones_like(vector), atol=1e-7, rtol=1e-7)

    near_pi_vectors = torch.tensor([
        [0.0, 0.0, 0.0],
        [math.pi - 0.005, 0.0, 0.0],
        [-math.pi + 0.005, 0.0, 0.0],
        [0.0, math.pi - 0.005, 0.0],
        [0.0, -math.pi + 0.005, 0.0],
        [0.0, 0.0, math.pi - 0.005],
        [0.0, 0.0, -math.pi + 0.005],
    ], dtype=torch.float64, requires_grad=True)
    near_pi_rotations = so3_exp(near_pi_vectors)
    near_pi_logs = so3_log(near_pi_rotations)
    torch.testing.assert_close(near_pi_logs, near_pi_vectors, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(so3_exp(near_pi_logs), near_pi_rotations, atol=1e-10, rtol=1e-10)
    near_pi_logs.sum().backward()
    assert near_pi_vectors.grad is not None and torch.isfinite(near_pi_vectors.grad).all()
    torch.testing.assert_close(
        near_pi_vectors.grad, torch.ones_like(near_pi_vectors), atol=1e-5, rtol=1e-5
    )

    original = _state(1)
    delta = torch.zeros(1, 51, dtype=torch.float64)
    delta[0, 3:6] = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    updated = apply_delta(original, delta)
    _, before = unpack_state(original)
    _, after = unpack_state(updated)
    expected_root = so3_exp(delta[0, 3:6]) @ before[0, 0]
    torch.testing.assert_close(after[0, 0], expected_root, atol=1e-10, rtol=1e-10)

    degenerate = torch.zeros(1, 99, dtype=torch.float64)
    projected = project_state(degenerate)
    _, matrices = unpack_state(projected)
    torch.testing.assert_close(torch.linalg.det(matrices), torch.ones(1, 16, dtype=torch.float64))


class _OffsetGeometry:
    """Small differentiable stand-in for MANO's joints(state) contract."""

    def joints(self, state: torch.Tensor) -> torch.Tensor:
        translation, root, hand, betas = unpack_legacy_state(state)
        wrist_offset = torch.stack((
            0.1 + 0.02 * betas[:, 0],
            0.03 * root[:, 0, 0, 0],
            0.04 * hand[:, 0, 1, 1],
        ), dim=-1)
        wrist = translation + wrist_offset
        offsets = torch.arange(21, dtype=state.dtype, device=state.device)[None, :, None]
        return wrist[:, None, :] + offsets * state.new_tensor([0.001, 0.002, 0.003])


def test_legacy_bridge_uses_actual_wrist_and_remains_differentiable() -> None:
    torch.manual_seed(11)
    geometry = _OffsetGeometry()
    batch = 3
    rotations = _random_rotations(batch, 16)
    translation = torch.randn(batch, 3, dtype=torch.float64)
    betas = torch.randn(batch, 10, dtype=torch.float64)
    legacy = pack_legacy_state(translation, rotations[:, :1], rotations[:, 1:], betas)

    state = legacy_to_state(legacy, geometry)
    expected_wrist = geometry.joints(legacy)[:, 0]
    torch.testing.assert_close(state[:, :3], expected_wrist)
    rebuilt = state_to_legacy(state, betas, geometry)
    torch.testing.assert_close(geometry.joints(rebuilt)[:, 0], expected_wrist)

    differentiable_state = state.detach().requires_grad_()
    differentiable_betas = betas.detach().requires_grad_()
    geometry.joints(state_to_legacy(differentiable_state, differentiable_betas, geometry)).sum().backward()
    assert differentiable_state.grad is not None
    assert torch.isfinite(differentiable_state.grad).all()
    assert differentiable_betas.grad is not None
    assert torch.isfinite(differentiable_betas.grad).all()


def test_matrix_fisher_normalizer_integrates_and_stays_finite_at_high_concentration() -> None:
    torch.manual_seed(17)
    kappa = 0.8
    # Normalized Gaussian quaternions give Haar-uniform SO(3) samples.
    quaternion = torch.randn(50_000, 4, dtype=torch.float64)
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y.square() + z.square()), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x.square() + z.square()), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x.square() + y.square()),
    ), dim=-1).reshape(-1, 3, 3)
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    log_z = matrix_fisher_log_normalizer(torch.tensor(kappa, dtype=torch.float64))
    integral = torch.exp(-kappa * (3 - trace) - log_z).mean()
    assert integral.item() == pytest.approx(1.0, abs=0.025)

    high = torch.tensor(10_000.0, dtype=torch.float64, requires_grad=True)
    high_log_z = matrix_fisher_log_normalizer(high)
    x_high = 2 * high.detach()
    asymptotic = (
        -0.5 * torch.log(2 * math.pi * x_high)
        + torch.log(
            1 / (2 * x_high)
            + 3 / (16 * x_high.square())
            + 45 / (256 * x_high.pow(3))
            + 25 / (96 * x_high.pow(4))
        )
    )
    torch.testing.assert_close(high_log_z, asymptotic, atol=2e-9, rtol=2e-9)
    high_log_z.backward()
    assert high.grad is not None and torch.isfinite(high.grad)


def test_probability_geometry_and_dispersion_shapes() -> None:
    center = _state(2)
    target = _state(2)
    distribution = {
        "state": center,
        "sigma": torch.full((2, 3), 0.05, dtype=torch.float64),
        "kappa": torch.full((2, 16), 1.5, dtype=torch.float64),
    }
    density = log_prob(distribution, target)
    assert density.shape == (2,) and torch.isfinite(density).all()
    loss = geometric_loss(center, target)
    assert loss.shape == (2,) and (loss >= 0).all()
    spread = dispersion(distribution)
    assert spread.shape == (2, 17)
    torch.testing.assert_close(spread[:, 0], torch.full((2,), 0.05, dtype=torch.float64))


def test_mixture_scores_unmasked_trajectory_and_ignores_invalid_tail() -> None:
    torch.manual_seed(23)
    batch, modes, future_steps = 2, 3, 4
    center = torch.randn(batch, modes, future_steps, 3, dtype=torch.float64) * 0.02
    rotations = _random_rotations(batch, modes, future_steps, 16)
    future_state = pack_state(center, rotations)
    future = {
        "state": future_state,
        "sigma": torch.full((batch, modes, future_steps, 3), 0.04, dtype=torch.float64),
        "kappa": torch.full((batch, modes, future_steps, 16), 2.0, dtype=torch.float64),
        "logits": torch.tensor([[0.3, -0.2, 0.8], [-0.5, 0.7, 0.1]], dtype=torch.float64),
    }
    target = _state(batch * future_steps).reshape(batch, future_steps, 99)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    masked = target.clone()
    masked[~valid] = float("nan")

    scored = mixture_log_prob(future, masked, valid)
    clean_scored = mixture_log_prob(future, target, valid)
    torch.testing.assert_close(scored, clean_scored)

    step_log_prob = log_prob(future, target[:, None])
    expected = torch.logsumexp(
        torch.log_softmax(future["logits"], dim=-1)
        + torch.where(valid[:, None], step_log_prob, torch.zeros_like(step_log_prob)).sum(-1),
        dim=-1,
    )
    torch.testing.assert_close(scored, expected)
    empty = mixture_log_prob(future, masked, torch.zeros_like(valid))
    torch.testing.assert_close(empty, torch.zeros(batch, dtype=torch.float64))

    train_future = {
        key: value.detach().clone().requires_grad_()
        for key, value in future.items()
    }
    padded_target = masked.detach().clone().requires_grad_()
    negative_log_likelihood = -mixture_log_prob(train_future, padded_target, valid).mean()
    negative_log_likelihood.backward()
    for value in train_future.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert padded_target.grad is not None and torch.isfinite(padded_target.grad).all()
    torch.testing.assert_close(
        padded_target.grad[~valid], torch.zeros_like(padded_target.grad[~valid])
    )
