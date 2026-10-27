"""CPU unit tests for bounded two-round intra-frame feedback."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
TRY = ROOT / "try"
sys.path.insert(0, str(TRY))

from dual_loop.handloop_model import HandLoopConfig, HandLoopObserver  # noqa: E402
from dual_loop.handloop_state import (  # noqa: E402
    apply_delta,
    make_anchor,
    pack_state,
    to_local,
    to_world,
    unpack_state,
)
from feedback_v2 import (  # noqa: E402
    INTRA_ARCHITECTURE,
    bound_visual_delta,
    install_feedback_v2,
)


def _state(position: tuple[float, float, float]) -> torch.Tensor:
    point = torch.tensor(position, dtype=torch.float32).reshape(1, 3)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 16, 3, 3).clone()
    return pack_state(point, rotations)


def _inputs(*, raw_available: bool = True) -> dict[str, torch.Tensor]:
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
        "image_available": torch.ones(1, dtype=torch.bool),
    }


class _SequenceVisual(nn.Module):
    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__()
        self.outputs = outputs
        self.calls = 0
        self.candidates: list[torch.Tensor] = []

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        self.candidates.append(kwargs["candidate_camera"].detach().clone())
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return self.outputs[index].to(kwargs["candidate_camera"]).expand(
            kwargs["candidate_camera"].shape[0], -1)


class _LearnableVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(51))

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.delta[None].expand(kwargs["candidate_camera"].shape[0], -1)


class _ResidualTWM(nn.Module):
    def __init__(self, translation: float = 0.01) -> None:
        super().__init__()
        residual = torch.zeros(51)
        residual[0] = translation
        self.residual = nn.Parameter(residual)
        self.candidates: list[torch.Tensor] = []
        self.raw_inputs: list[torch.Tensor] = []
        self.history_inputs: list[torch.Tensor] = []

    def forward(
        self,
        history: torch.Tensor,
        raw: torch.Tensor,
        candidate: torch.Tensor,
        history_available: torch.Tensor,
        raw_available: torch.Tensor,
        candidate_available: torch.Tensor,
        history_times: torch.Tensor,
        dt: torch.Tensor,
        handedness: torch.Tensor,
        history_mask: torch.Tensor,
        future_times: torch.Tensor | None = None,
    ) -> dict[str, object]:
        self.candidates.append(candidate.detach().clone())
        self.raw_inputs.append(raw.detach().clone())
        self.history_inputs.append(history.detach().clone())
        state = apply_delta(candidate, self.residual[None].expand(candidate.shape[0], -1))
        return {
            "current": {
                "state": state,
                "sigma": state.new_full((state.shape[0], 3), 0.1),
                "kappa": state.new_ones(state.shape[0], 16),
            },
            "future": None,
            "balance_loss": state.new_zeros(()),
        }


class _ToySystem(nn.Module):
    def __init__(self, visual: nn.Module, twm: nn.Module) -> None:
        super().__init__()
        self.hawor = nn.Linear(1, 1)
        self.observer = HandLoopObserver(
            HandLoopConfig(rounds=2, reobserve=True), visual=visual, twm=twm)


def _projector(seen: list[torch.Tensor]):
    def project(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(state.detach().clone())
        position = state[:, None, :2].expand(-1, 16, -1)
        valid = torch.ones(state.shape[0], 16, dtype=torch.bool, device=state.device)
        return position, valid

    return project


def _run(system: _ToySystem, inputs: dict[str, torch.Tensor], seen: list[torch.Tensor]):
    return system.observer.forward_step(**inputs, projector=_projector(seen))


def test_zero_visual_two_rounds_match_one_raw_valid_twm_update() -> None:
    visual = _SequenceVisual([torch.zeros(51)])
    twm = _ResidualTWM(translation=0.013)
    system = _ToySystem(visual, twm)
    visual_keys = set(system.observer.visual.state_dict())
    observer = install_feedback_v2(system)
    assert set(observer.visual.state_dict()) == visual_keys
    inputs = _inputs(raw_available=True)
    raw_before = inputs["raw_state"].clone()
    history_before = inputs["history"].clone()
    projector_states: list[torch.Tensor] = []

    result = _run(system, inputs, projector_states)

    anchor_rotation, anchor_position = make_anchor(
        inputs["raw_state"], inputs["history"], inputs["raw_available"],
        inputs["history_available"])
    initial_anchor = to_local(inputs["raw_state"], anchor_rotation, anchor_position)
    one_update = apply_delta(initial_anchor, twm.residual.detach()[None])
    expected = to_world(one_update, anchor_rotation, anchor_position)
    torch.testing.assert_close(result["state"], expected)
    assert len(twm.candidates) == 2
    torch.testing.assert_close(twm.candidates[0], initial_anchor)
    torch.testing.assert_close(twm.candidates[1], initial_anchor)
    assert len(result["rounds"]) == 2
    assert len(result["visual_candidates"]) == len(result["visual_deltas_camera"]) == 2
    torch.testing.assert_close(result["visual_deltas_camera"][0], torch.zeros(1, 51))
    torch.testing.assert_close(result["visual_deltas_camera"][1], torch.zeros(1, 51))
    torch.testing.assert_close(inputs["raw_state"], raw_before)
    torch.testing.assert_close(inputs["history"], history_before)
    torch.testing.assert_close(twm.raw_inputs[0], twm.raw_inputs[1])
    torch.testing.assert_close(twm.history_inputs[0], twm.history_inputs[1])
    assert not any(parameter.requires_grad for parameter in observer.twm.parameters())
    assert not any(parameter.requires_grad for parameter in system.hawor.parameters())
    assert observer.architecture == INTRA_ARCHITECTURE


def test_missing_raw_uses_original_latest_history_fallback() -> None:
    visual = _SequenceVisual([torch.zeros(51)])
    twm = _ResidualTWM(translation=0.0)
    system = _ToySystem(visual, twm)
    install_feedback_v2(system)
    inputs = _inputs(raw_available=False)
    projector_states: list[torch.Tensor] = []

    _run(system, inputs, projector_states)

    anchor_rotation, anchor_position = make_anchor(
        inputs["history"][:, -1], inputs["history"], inputs["history_available"][:, -1],
        inputs["history_available"])
    expected = to_local(inputs["history"][:, -1], anchor_rotation, anchor_position)
    torch.testing.assert_close(twm.candidates[0], expected)
    torch.testing.assert_close(twm.candidates[1], expected)


def test_second_round_zero_visual_keeps_first_restored_state() -> None:
    first = torch.zeros(51)
    first[0] = 8.0
    visual = _SequenceVisual([first, torch.zeros(51)])
    twm = _ResidualTWM(translation=0.01)
    system = _ToySystem(visual, twm)
    install_feedback_v2(system)
    inputs = _inputs()
    projector_states: list[torch.Tensor] = []

    result = _run(system, inputs, projector_states)

    assert len(projector_states) == 2
    assert not torch.allclose(projector_states[0], projector_states[1])
    torch.testing.assert_close(result["rounds"][0]["state"], result["rounds"][1]["state"])
    round_one_world = to_world(
        result["rounds"][0]["state"], result["anchor_rotation"], result["anchor_position"])
    torch.testing.assert_close(result["state"], round_one_world)
    torch.testing.assert_close(result["visual_candidates"][0], result["visual_candidates"][1])
    assert not torch.allclose(visual.candidates[0], visual.candidates[1])


def test_vector_norm_bounds_and_so3_rotations() -> None:
    raw_delta = torch.full((2, 51), 10000.0)
    bounded = bound_visual_delta(raw_delta)
    assert torch.all(torch.linalg.vector_norm(bounded[:, :3], dim=-1) <= 0.0025 + 1e-7)
    assert torch.all(torch.linalg.vector_norm(bounded[:, 3:6], dim=-1) <= 0.025 + 1e-7)
    fingers = bounded[:, 6:].reshape(2, 15, 3)
    assert torch.all(torch.linalg.vector_norm(fingers, dim=-1) <= 0.05 + 1e-7)

    updated = apply_delta(_state((0.0, 0.0, 0.0)).expand(2, -1), bounded)
    _, rotations = unpack_state(updated)
    identity = torch.eye(3).expand(2, 16, 3, 3)
    torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
    assert torch.all(torch.linalg.det(rotations) > 0.999)


def test_small_visual_delta_keeps_physical_scale_and_unit_gradient() -> None:
    raw_delta = torch.zeros(1, 51)
    raw_delta[0, 0] = 1e-6
    raw_delta.requires_grad_()

    bounded = bound_visual_delta(raw_delta)
    torch.testing.assert_close(bounded[0, 0], raw_delta[0, 0], atol=1e-10, rtol=1e-6)
    bounded[0, 0].backward()
    assert raw_delta.grad is not None
    torch.testing.assert_close(raw_delta.grad[0, 0], torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_gradients_reach_visual_through_frozen_twm() -> None:
    visual = _LearnableVisual()
    twm = _ResidualTWM(translation=0.01)
    system = _ToySystem(visual, twm)
    observer = install_feedback_v2(system)
    inputs = _inputs()
    projector_states: list[torch.Tensor] = []

    result = _run(system, inputs, projector_states)
    result["state"][:, 0].sum().backward()

    assert observer.visual.delta.grad is not None
    assert torch.isfinite(observer.visual.delta.grad).all()
    assert observer.visual.delta.grad[0].abs() > 0
    assert not twm.residual.requires_grad
    assert twm.residual.grad is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
