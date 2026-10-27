from __future__ import annotations

import torch
from torch import nn

from dual_loop.model import DualLoopConfig, DualLoopObserver
from dual_loop.networks import TrajectoryWorldModel, VisualQualityEstimator
from dual_loop.state import STATE_DIM, pack_state, unpack_state


def identity_state(batch: int = 2) -> torch.Tensor:
    root = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, 1, 1, 1)
    hand = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, 15, 1, 1)
    return pack_state(torch.zeros(batch, 3), root, hand, torch.zeros(batch, 10))


def test_state_roundtrip() -> None:
    state = identity_state()
    translation, root, hand, betas = unpack_state(state)
    rebuilt = pack_state(translation, root, hand, betas)
    assert state.shape == (2, STATE_DIM)
    assert torch.allclose(state, rebuilt, atol=1e-6)


def test_identity_safe_initialization() -> None:
    config = DualLoopConfig(correction_steps=2)
    model = DualLoopObserver(config).eval()
    visual = identity_state(batch=1)
    history = visual[:, None]
    image = torch.zeros(1, 3, 64, 64)
    observed_states = []

    def projector(state: torch.Tensor):
        observed_states.append(state.detach().clone())
        grid = torch.zeros(state.shape[0], 21, 2)
        visible = torch.ones(state.shape[0], 21, dtype=torch.bool)
        return grid, visible

    with torch.inference_mode():
        output = model.forward_step(
            image, visual, history, projector,
            torch.ones(1, dtype=torch.bool), torch.ones(1, 1))
    assert output["state"].shape == (1, STATE_DIM)
    assert output["gates"].shape == (1, 2)
    assert torch.allclose(output["state"], visual, atol=1e-6)
    assert len(observed_states) == config.correction_steps + 1


def test_invisible_final_projection_forces_zero_quality() -> None:
    model = DualLoopObserver(DualLoopConfig(correction_steps=2)).eval()
    visual = identity_state(batch=1)

    def projector(state: torch.Tensor):
        return (
            torch.zeros(state.shape[0], 21, 2),
            torch.zeros(state.shape[0], 21, dtype=torch.bool),
        )

    with torch.inference_mode():
        output = model.forward_step(
            torch.zeros(1, 3, 64, 64), visual, visual[:, None], projector,
            torch.ones(1, dtype=torch.bool), torch.zeros(1, 1))
    assert output["quality"].item() == 0.0
    assert output["visual_quality"].item() == 0.0
    assert output["visibility_fraction"].item() == 0.0
    assert not output["final_visibility"].any()
    assert output["visual_quality_logit"].shape == (1, 1)
    assert torch.isfinite(output["prior_variance"]).all()


def test_quality_hint_has_no_hard_confidence_floor() -> None:
    estimator = VisualQualityEstimator(
        STATE_DIM, feature_dim=8, hidden_dim=16, evidence_dim=12)
    with torch.no_grad():
        estimator.network[-1].weight.zero_()
        estimator.network[-1].bias.fill_(-8.0)
    state = identity_state(batch=1)
    quality = estimator(
        state, state, torch.zeros(1, 8), torch.ones(1, 1),
        torch.ones(1, dtype=torch.bool), local_evidence=torch.zeros(1, 12),
        visibility=torch.ones(1, 21, dtype=torch.bool))
    assert quality.item() < 0.01


def test_missing_frontend_can_use_visible_local_evidence() -> None:
    model = DualLoopObserver(DualLoopConfig(correction_steps=1)).eval()
    visual = identity_state(batch=1)

    def projector(state: torch.Tensor):
        return (
            torch.zeros(state.shape[0], 21, 2),
            torch.ones(state.shape[0], 21, dtype=torch.bool),
        )

    with torch.inference_mode():
        output = model.forward_step(
            torch.zeros(1, 3, 64, 64), visual, visual[:, :0], projector,
            torch.zeros(1, dtype=torch.bool), torch.zeros(1, 1))
    assert output["visual_quality"].item() > 0.0
    assert output["gates"].item() > 0.0
    assert 0.0 < output["quality"].item() < 0.20


def test_reprojection_count_tracks_correction_setting_and_final_state() -> None:
    class MovingCorrector(nn.Module):
        def forward(
            self,
            candidate: torch.Tensor,
            visual_state: torch.Tensor,
            prior_state: torch.Tensor,
            evidence: torch.Tensor,
            visual_quality: torch.Tensor,
            prior_logvar: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            moved = candidate.clone()
            moved[..., 0] += 0.01
            return moved, visual_quality

    for reobserve, expected_x in ((True, (0.0, 0.01, 0.02)),
                                  (False, (0.0, 0.02))):
        config = DualLoopConfig(
            correction_steps=2, intra_frame_reobserve=reobserve)
        model = DualLoopObserver(config).eval()
        model.corrector = MovingCorrector()
        visual = identity_state(batch=1)
        observed_x = []

        def projector(state: torch.Tensor):
            observed_x.append(float(state[0, 0]))
            return (
                torch.zeros(state.shape[0], 21, 2),
                torch.ones(state.shape[0], 21, dtype=torch.bool),
            )

        with torch.inference_mode():
            model.forward_step(
                torch.zeros(1, 3, 64, 64), visual, visual[:, None], projector,
                torch.ones(1, dtype=torch.bool), torch.ones(1, 1))
        assert len(observed_x) == len(expected_x)
        assert torch.allclose(
            torch.tensor(observed_x), torch.tensor(expected_x), atol=1e-6)


def test_world_model_prior_is_gauge_equivariant_and_supports_horizons() -> None:
    model = TrajectoryWorldModel(hidden_dim=16).eval()
    batch, frames = 2, 3
    translation = torch.zeros(batch, frames, 3)
    translation[..., 0] = torch.arange(frames, dtype=torch.float32) * 0.02
    root = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(batch, frames, 1, 1, 1)
    hand = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(batch, frames, 15, 1, 1)
    betas = torch.zeros(batch, frames, 10)
    history = pack_state(translation, root, hand, betas)

    angle = torch.tensor(0.7)
    rotation = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    offset = torch.tensor([0.3, -0.4, 0.2])
    moved_translation = torch.matmul(
        rotation, translation.unsqueeze(-1)).squeeze(-1) + offset
    moved_root = rotation.reshape(1, 1, 1, 3, 3) @ root
    moved_history = pack_state(moved_translation, moved_root, hand, betas)

    with torch.inference_mode():
        prediction, logvar = model(history, steps=3)
        moved_prediction, moved_logvar = model(moved_history, steps=3)
        one_step, one_step_logvar = model(history)
        rolling_history = history
        repeated_predictions = []
        repeated_logvars = []
        for _ in range(3):
            repeated_prediction, repeated_logvar = model(rolling_history)
            repeated_predictions.append(repeated_prediction)
            repeated_logvars.append(repeated_logvar)
            rolling_history = torch.cat(
                (rolling_history, repeated_prediction[:, None]), dim=1
            )[:, -frames:]

    predicted_translation, predicted_root, predicted_hand, predicted_betas = unpack_state(
        prediction)
    expected_translation = torch.matmul(
        rotation, predicted_translation.unsqueeze(-1)).squeeze(-1) + offset
    expected_root = rotation.reshape(1, 1, 1, 3, 3) @ predicted_root
    moved_translation, moved_root, moved_hand, moved_betas = unpack_state(moved_prediction)
    assert torch.allclose(moved_translation, expected_translation, atol=1e-5)
    assert torch.allclose(moved_root, expected_root, atol=1e-5)
    assert torch.allclose(moved_hand, predicted_hand, atol=1e-5)
    assert torch.allclose(moved_betas, predicted_betas, atol=1e-5)
    assert torch.allclose(moved_logvar, logvar, atol=1e-5)
    assert torch.allclose(prediction, torch.stack(repeated_predictions, dim=1), atol=1e-6)
    assert torch.allclose(logvar, torch.stack(repeated_logvars, dim=1), atol=1e-6)
    assert one_step.shape == (batch, STATE_DIM)
    assert one_step_logvar.shape == (batch, STATE_DIM)
    assert torch.allclose(
        logvar[..., :3], logvar[..., :1].expand_as(logvar[..., :3]))

