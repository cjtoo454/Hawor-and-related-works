from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from dual_loop.handloop_model import HandLoopConfig, HandLoopObserver
from dual_loop.handloop_state import apply_delta, pack_state, so3_exp, to_world, unpack_state
from dual_loop.handloop_twm import TWMConfig
from dual_loop.handloop_visual import HandLoopVisual, _time_features


def _state(batch: int, time: int = 0, position: tuple[float, float, float] = (0, 0, 0)) -> torch.Tensor:
    leading = (batch,) if time == 0 else (batch, time)
    p = torch.tensor(position, dtype=torch.float32).expand(*leading, 3).clone()
    tangent = torch.zeros(*leading, 16, 3)
    tangent[..., 0, :] = torch.tensor((0.08, -0.04, 0.03))
    return pack_state(p, so3_exp(tangent))


class _ZeroVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[torch.Tensor] = []
        self.grids: list[torch.Tensor] = []

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        self.candidates.append(kwargs["candidate_camera"].detach().clone())
        self.grids.append(kwargs["projected_grid"].detach().clone())
        return kwargs["candidate_camera"].new_zeros(kwargs["candidate_camera"].shape[0], 51)


class _PassThroughTWM(nn.Module):
    def __init__(self, shift_per_call: float = 0.0) -> None:
        super().__init__()
        self.calls = 0
        self.shift_per_call = shift_per_call
        self.future_flags: list[bool] = []

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
        self.calls += 1
        self.future_flags.append(future_times is not None)
        state = candidate
        if self.shift_per_call:
            delta = candidate.new_zeros(candidate.shape[0], 51)
            delta[:, 0] = self.shift_per_call
            state = apply_delta(candidate, delta)
        batch = candidate.shape[0]
        return {
            "current": {
                "state": state,
                "sigma": candidate.new_full((batch, 3), 0.1),
                "kappa": candidate.new_ones(batch, 16),
            },
            "future": None,
            "balance_loss": candidate.new_zeros(()),
        }


def _observer_inputs(batch: int = 1, history_length: int = 0) -> dict[str, torch.Tensor]:
    history = _state(batch, max(1, history_length))[:, :history_length]
    return {
        "patches": torch.zeros(batch, 192, 8),
        "raw_state": _state(batch, position=(0.2, -0.1, 0.7)),
        "history": history,
        "raw_available": torch.ones(batch, 17, dtype=torch.bool),
        "history_available": torch.ones(batch, history_length, 17, dtype=torch.bool),
        "history_mask": torch.ones(batch, history_length, dtype=torch.bool),
        "history_times": torch.linspace(-0.08, -0.02, history_length).expand(batch, -1).clone(),
        "dt": torch.full((batch,), 0.02),
        "handedness": torch.zeros(batch, dtype=torch.long),
        "history_dispersion": torch.zeros(batch, history_length, 17),
        "dispersion_known": torch.zeros(batch, history_length, dtype=torch.bool),
        "rotation_w2c": torch.eye(3).expand(batch, 3, 3).clone(),
        "translation_w2c": torch.zeros(batch, 3),
        "image_available": torch.ones(batch, dtype=torch.bool),
    }


def _camera_projector(
    rotation_w2c: torch.Tensor,
    translation_w2c: torch.Tensor,
    seen: list[torch.Tensor] | None = None,
):
    def project(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p, _ = unpack_state(state)
        camera_p = (rotation_w2c @ p.unsqueeze(-1)).squeeze(-1) + translation_w2c
        if seen is not None:
            seen.append(state.detach().clone())
        offset = torch.arange(16, dtype=state.dtype, device=state.device)[None] * 0.002
        grid = torch.stack((
            torch.tanh(camera_p[:, :1] + offset),
            torch.tanh(camera_p[:, 1:2] - offset),
        ), dim=-1)
        return grid, torch.ones(state.shape[0], 16, dtype=torch.bool, device=state.device)
    return project


def test_visual_missing_image_and_history_stays_finite_and_zero() -> None:
    visual = HandLoopVisual(width=8, heads=2, feature_dim=8, blocks=1)
    assert all(
        block.global_attention is block.local_attention
        for block in visual.blocks
    )
    batch, time = 2, 2
    raw = _state(batch)
    invalid_history = torch.full((batch, time, 99), float("nan"))
    result = visual(
        patches=torch.full((batch, 192, 8), float("nan")),
        raw_camera=raw,
        candidate_camera=raw,
        history_camera=invalid_history,
        history_dispersion=torch.full((batch, time, 17), float("nan")),
        history_available=torch.zeros(batch, time, 17, dtype=torch.bool),
        history_mask=torch.zeros(batch, time, dtype=torch.bool),
        history_times=torch.full((batch, time), float("nan")),
        dispersion_known=torch.zeros(batch, time, dtype=torch.bool),
        projected_grid=torch.full((batch, 16, 2), float("nan")),
        projection_valid=torch.zeros(batch, 16, dtype=torch.bool),
        handedness=torch.tensor((0, 1)),
        image_available=torch.zeros(batch, dtype=torch.bool),
    )
    assert result.shape == (batch, 51)
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(result))


def test_history_dispersion_keeps_position_rms_and_logs_known_kappa() -> None:
    class _CaptureProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value: torch.Tensor | None = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.value = value.detach().clone()
            return value.new_zeros(*value.shape[:-1], 8)

    visual = HandLoopVisual(width=8, heads=2, feature_dim=8, blocks=1)
    capture = _CaptureProjection()
    visual.history_projection = capture
    history_dispersion = torch.tensor([[[0.25, 2.0, 4.0] + [1.0] * 14]])
    known = torch.ones(1, 1, dtype=torch.bool)
    visual._history_tokens(
        _state(1, 1), history_dispersion,
        torch.ones(1, 1, 17, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1), known,
    )
    assert capture.value is not None
    torch.testing.assert_close(
        capture.value[0, 0, 0, 11:13], torch.tensor((0.25, torch.log(torch.tensor(2.0))))
    )
    torch.testing.assert_close(
        capture.value[0, 0, 1, 11:13], torch.tensor((0.0, torch.log(torch.tensor(4.0))))
    )


def test_single_history_time_preserves_age_relative_to_current_frame() -> None:
    recent = _time_features(torch.tensor([[-0.03]]), torch.float32)
    older = _time_features(torch.tensor([[-0.24]]), torch.float32)
    assert not torch.equal(recent, older)
    torch.testing.assert_close(recent[..., 0], torch.tensor([[-0.03]]))
    torch.testing.assert_close(older[..., 0], torch.tensor([[-0.24]]))


def test_observer_runs_two_rounds_and_reprojects_updated_state() -> None:
    visual = _ZeroVisual()
    twm = _PassThroughTWM(shift_per_call=0.01)
    observer = HandLoopObserver(
        HandLoopConfig(rounds=2), visual=visual, twm=twm).eval()
    inputs = _observer_inputs()
    projected: list[torch.Tensor] = []
    inputs["projector"] = _camera_projector(
        inputs["rotation_w2c"], inputs["translation_w2c"], projected)  # type: ignore[assignment]
    with torch.no_grad():
        output = observer.forward_step(**inputs)  # type: ignore[arg-type]

    assert twm.calls == 2
    assert len(output["rounds"]) == 2
    assert len(projected) == 2
    assert len(visual.candidates) == 2
    assert torch.linalg.vector_norm(projected[1][:, :3] - projected[0][:, :3]) > 0.009
    final_position, _ = unpack_state(output["state"])
    torch.testing.assert_close(final_position[:, 0], torch.tensor((0.22,)), atol=5e-5, rtol=0)


def test_initial_state_selects_each_available_group_from_raw_then_latest_history() -> None:
    visual = _ZeroVisual()
    observer = HandLoopObserver(
        HandLoopConfig(rounds=1), visual=visual, twm=_PassThroughTWM()).eval()
    raw_position = torch.tensor([[0.5, -0.2, 0.9]])
    raw_rotations = torch.eye(3).expand(1, 16, 3, 3).clone()
    raw_rotations[:, 0] = so3_exp(torch.tensor([[0.0, 0.0, -0.4]]))
    raw = pack_state(raw_position, raw_rotations)

    history_position = torch.tensor([[[0.1, 0.0, 0.7], [0.2, 0.0, 0.8]]])
    history_rotations = torch.eye(3).expand(1, 2, 16, 3, 3).clone()
    history_rotations[:, 0, 0] = so3_exp(torch.tensor([[0.0, 0.0, 0.2]]))
    history_rotations[:, 1, 0] = so3_exp(torch.tensor([[0.0, 0.0, 0.5]]))
    history_rotations[:, 0, 1] = so3_exp(torch.tensor([[0.1, 0.0, 0.0]]))
    history_rotations[:, 1, 1] = so3_exp(torch.tensor([[0.3, 0.0, 0.0]]))
    history = pack_state(history_position, history_rotations)
    raw_available = torch.zeros(1, 17, dtype=torch.bool)
    raw_available[:, 0] = True
    history_available = torch.zeros(1, 2, 17, dtype=torch.bool)
    history_available[:, :, 0] = True
    history_available[:, 0, 1] = True
    history_available[:, :, 2] = True
    inputs = _observer_inputs(history_length=2)
    inputs["raw_state"] = raw
    inputs["raw_available"] = raw_available
    inputs["history"] = history
    inputs["history_available"] = history_available
    with torch.no_grad():
        observer.forward_step(
            **inputs,
            projector=_camera_projector(
                inputs["rotation_w2c"], inputs["translation_w2c"]),
        )

    selected_position, selected_rotations = unpack_state(visual.candidates[0])
    torch.testing.assert_close(selected_position, raw_position)
    torch.testing.assert_close(selected_rotations[:, 0], history_rotations[:, 0, 0])
    torch.testing.assert_close(selected_rotations[:, 1], history_rotations[:, 1, 1])
    torch.testing.assert_close(
        selected_rotations[:, 2:], torch.eye(3).expand(1, 14, 3, 3))


def test_observer_is_equivariant_to_world_rigid_transform() -> None:
    observer = HandLoopObserver(
        HandLoopConfig(rounds=2), visual=_ZeroVisual(), twm=_PassThroughTWM()).eval()
    inputs = _observer_inputs(history_length=1)
    inputs["history"] = _state(1, 1, position=(0.16, -0.08, 0.68))
    inputs["history_times"] = torch.tensor([[-0.03]])
    q = so3_exp(torch.tensor([[0.2, -0.1, 0.3]]))
    shift = torch.tensor([[0.4, -0.3, 0.2]])
    rotation_w2c = so3_exp(torch.tensor([[-0.15, 0.12, 0.08]]))
    translation_w2c = torch.tensor([[0.03, -0.04, 0.2]])
    inputs["rotation_w2c"] = rotation_w2c
    inputs["translation_w2c"] = translation_w2c
    camera_projector = _camera_projector(rotation_w2c, translation_w2c)
    with torch.no_grad():
        baseline = observer.forward_step(**inputs, projector=camera_projector)  # type: ignore[arg-type]

        transformed = dict(inputs)
        transformed["raw_state"] = to_world(inputs["raw_state"], q, shift)
        transformed["history"] = to_world(inputs["history"], q, shift)
        transformed_rotation = rotation_w2c @ q.transpose(-1, -2)
        transformed_translation = translation_w2c - (
            transformed_rotation @ shift.unsqueeze(-1)).squeeze(-1)
        transformed["rotation_w2c"] = transformed_rotation
        transformed["translation_w2c"] = transformed_translation
        transformed_projector = _camera_projector(
            transformed_rotation, transformed_translation)
        actual = observer.forward_step(
            **transformed, projector=transformed_projector)  # type: ignore[arg-type]

    expected_state = to_world(baseline["state"], q, shift)
    torch.testing.assert_close(actual["state"], expected_state, atol=2e-6, rtol=2e-6)


def test_observer_integrates_small_real_twm_and_optional_future() -> None:
    config = HandLoopConfig.from_dict({
        "architecture": "handloop_v2",
        "rounds": 2,
        "visual": {"width": 8, "heads": 2, "feature_dim": 8, "blocks": 1},
        "twm": {
            "width": 8, "layers": 1, "heads": 2, "state_size": 2,
            "conv_kernel": 2, "expand": 1, "experts": 2,
            "history_length": 2, "modes": 2, "future_steps": 2,
        },
    })
    observer = HandLoopObserver(config).eval()
    inputs = _observer_inputs(history_length=1)
    inputs["future_times"] = torch.tensor([[0.04, 0.08]])  # type: ignore[assignment]
    with torch.no_grad():
        output = observer.forward_step(
            **inputs,
            projector=_camera_projector(
                inputs["rotation_w2c"], inputs["translation_w2c"]),
        )  # type: ignore[arg-type]

    assert output["state"].shape == (1, 99)
    assert len(output["rounds"]) == 2
    assert output["future"] is not None
    assert output["future"]["state"].shape == (1, 2, 2, 99)


def test_ablation_freezes_first_projection_and_history_commit_resets_stale_track() -> None:
    visual = _ZeroVisual()
    twm = _PassThroughTWM(shift_per_call=0.01)
    observer = HandLoopObserver(
        HandLoopConfig(rounds=2, reobserve=False), visual=visual, twm=twm).eval()
    inputs = _observer_inputs()
    projected: list[torch.Tensor] = []
    with torch.no_grad():
        observer.forward_step(
            **inputs,
            projector=_camera_projector(
                inputs["rotation_w2c"], inputs["translation_w2c"], projected),
        )
    assert twm.calls == 2
    assert len(projected) == 1
    torch.testing.assert_close(visual.grids[0], visual.grids[1])

    state = _state(1).requires_grad_()
    empty = _observer_inputs()
    committed = HandLoopObserver.commit_history(
        empty["history"], empty["history_available"], empty["history_dispersion"],
        empty["dispersion_known"], empty["history_times"], empty["history_mask"],
        torch.empty(1, 0, dtype=torch.long), state,
        empty["raw_available"], torch.ones(1, 17), torch.tensor([0.0]),
        torch.tensor([0]), max_length=4, dispersion_is_known=False,
    )
    next_state = HandLoopObserver.commit_history(
        committed["history"], committed["history_available"],
        committed["history_dispersion"], committed["dispersion_known"],
        committed["history_times"], committed["history_mask"],
        committed["history_handedness"], state, empty["raw_available"],
        torch.ones(1, 17), torch.tensor([0.4]), torch.tensor([1]), max_length=4,
    )
    assert next_state["reset"].item()
    assert next_state["history_mask"].tolist() == [[False, True]]
    assert not next_state["history"].requires_grad
    assert next_state["dispersion_known"].tolist() == [[False, True]]
