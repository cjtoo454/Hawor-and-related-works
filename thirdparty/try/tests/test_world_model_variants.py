from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


TRY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRY_ROOT / "scripts"))
sys.path.insert(0, str(TRY_ROOT))

from dual_loop.networks import (  # noqa: E402
    TrajectoryWorldModel,
    WORLD_MODEL_FORMAT_VERSION,
)
from dual_loop.state import (  # noqa: E402
    STATE_DIM,
    axis_angle_to_matrix,
    pack_state,
    project_state,
)
from train import world_model_variant_from_checkpoint  # noqa: E402
from train_world_model import (  # noqa: E402
    CV_RESIDUAL_LOSS_CONFIG,
    LEGACY_LOSS_CONFIG,
    _validate_resume_metadata,
    cv_residual_objective,
    translation_regression_loss,
)


def _history(batch: int = 2, frames: int = 4) -> torch.Tensor:
    times = torch.arange(frames, dtype=torch.float32)
    translation = torch.zeros(batch, frames, 3)
    translation[..., 0] = times * 0.015
    translation[..., 1] = times.square() * 0.001
    root_angles = torch.zeros(batch, frames, 1, 3)
    root_angles[..., 1] = times.reshape(1, frames, 1) * 0.04
    root_angles[..., 2] = times.reshape(1, frames, 1) * -0.02
    root = axis_angle_to_matrix(root_angles)
    hand_angles = torch.zeros(batch, frames, 15, 3)
    hand_angles[..., 0] = times.reshape(1, frames, 1) * 0.01
    hand = axis_angle_to_matrix(hand_angles)
    return pack_state(translation, root, hand, torch.zeros(batch, frames, 10))


def test_cv_residual_zero_initialization_matches_deployed_recursive_cv() -> None:
    model = TrajectoryWorldModel(hidden_dim=16, variant="cv_residual").eval()
    history = _history()
    horizon = 10
    last, previous = history[:, -1], history[:, -2]
    step_values = torch.arange(1, horizon + 1, dtype=history.dtype)[None, :, None]
    anchored_cv = project_state(last[:, None] + step_values * (last - previous)[:, None])

    with torch.inference_mode():
        prediction, logvar = model(history, steps=horizon)
        rolling_history = history
        repeated_predictions = []
        for _ in range(horizon):
            repeated, _ = model(rolling_history)
            repeated_predictions.append(repeated)
            rolling_history = torch.cat(
                (rolling_history, repeated[:, None]), dim=1
            )[:, -history.shape[1]:]

    assert prediction.shape == logvar.shape == (2, horizon, STATE_DIM)
    assert torch.equal(prediction, torch.stack(repeated_predictions, dim=1))
    # Translation is linear and exactly matches the established evaluator CV
    # baseline; rotations follow the deployed recursive projection semantics.
    assert torch.allclose(prediction[..., :3], anchored_cv[..., :3], atol=1e-7)
    assert torch.equal(prediction[..., 99:], anchored_cv[..., 99:])
    assert not torch.equal(prediction[..., 3:99], anchored_cv[..., 3:99])


def test_cv_residual_objective_trains_late_steps_and_separates_variance_gradient() -> None:
    prediction = _history(batch=1, frames=5).detach().requires_grad_(True)
    target = prediction.detach().clone()
    target[:, 3, 0] += 0.02
    target[:, 3, 3:9] += 0.04
    target[:, 3, 99:] += 0.03
    logvar = torch.zeros_like(prediction, requires_grad=True)

    loss, terms = cv_residual_objective(prediction, logvar, target)
    uncertainty_mean_gradient = torch.autograd.grad(
        terms["uncertainty"], prediction, allow_unused=True, retain_graph=True
    )[0]
    loss.backward()

    assert torch.isfinite(loss)
    assert set(terms) == {"translation", "rotation", "shape", "uncertainty"}
    assert prediction.grad is not None
    assert prediction.grad[:, 3].abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[:, :3]) == 0
    assert torch.count_nonzero(prediction.grad[:, 4:]) == 0
    assert logvar.grad is not None and logvar.grad.abs().sum() > 0
    assert uncertainty_mean_gradient is None


def test_zero_initialized_residual_head_receives_gradient_and_rollout_matches_calls() -> None:
    model = TrajectoryWorldModel(hidden_dim=16, variant="cv_residual").train()
    history = _history(batch=1)
    prediction, logvar = model(history)
    target = prediction.detach().clone()
    target[..., 0] += 0.02
    loss, _ = cv_residual_objective(prediction[:, None], logvar[:, None], target[:, None])
    loss.backward()
    assert model.residual_head.weight.grad is not None
    assert model.residual_head.weight.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        model.residual_head.weight.fill_(0.002)
        model.residual_head.bias.fill_(0.01)
        multi_prediction, multi_logvar = model(history, steps=5)
        rolling_history = history
        repeated_predictions = []
        repeated_logvars = []
        for _ in range(5):
            one_prediction, one_logvar = model(rolling_history)
            repeated_predictions.append(one_prediction)
            repeated_logvars.append(one_logvar)
            rolling_history = torch.cat(
                (rolling_history, one_prediction[:, None]), dim=1
            )[:, -history.shape[1]:]

    assert torch.equal(multi_prediction, torch.stack(repeated_predictions, dim=1))
    assert torch.equal(multi_logvar, torch.stack(repeated_logvars, dim=1))


def test_translation_regression_is_independent_of_log_variance() -> None:
    prediction = _history(batch=1, frames=2)
    target = prediction.clone()
    target[..., 0] += 0.01
    direct = translation_regression_loss(prediction, target)

    # The direct translation term accepts no variance input and is included at
    # a fixed weight in the CV residual objective.
    assert torch.allclose(direct, torch.tensor(0.005 / 3))
    assert CV_RESIDUAL_LOSS_CONFIG["translation_weight"] == 10.0


def test_variant_checkpoint_and_resume_metadata_are_explicit_and_compatible() -> None:
    legacy = TrajectoryWorldModel(hidden_dim=16)
    cv_residual = TrajectoryWorldModel(hidden_dim=16, variant="cv_residual")
    assert legacy.variant == "legacy"
    assert "delta_head.weight" in legacy.state_dict()
    assert "residual_head.weight" not in legacy.state_dict()
    with pytest.raises(RuntimeError):
        cv_residual.load_state_dict(legacy.state_dict(), strict=True)

    assert world_model_variant_from_checkpoint({
        "stage": "world_model_pretrain",
        "model_config": {"state_dim": STATE_DIM, "hidden_dim": 256},
    }) == ("legacy", 256)
    assert world_model_variant_from_checkpoint({
        "stage": "world_model_pretrain",
        "variant": "cv_residual",
        "world_model_format_version": WORLD_MODEL_FORMAT_VERSION["cv_residual"],
        "model_config": {
            "state_dim": STATE_DIM,
            "hidden_dim": 256,
            "variant": "cv_residual",
            "format_version": WORLD_MODEL_FORMAT_VERSION["cv_residual"],
        },
    }) == ("cv_residual", 256)

    previous = {
        "model_config": {"hidden_dim": 256},
        "loss_config": LEGACY_LOSS_CONFIG,
    }
    current = {
        "model_config": {
            "hidden_dim": 256,
            "variant": "legacy",
            "format_version": WORLD_MODEL_FORMAT_VERSION["legacy"],
        },
        "variant": "legacy",
        "loss_config": LEGACY_LOSS_CONFIG,
    }
    _validate_resume_metadata(previous, current, "test")
    with pytest.raises(ValueError, match="variant|model_config|loss_config"):
        _validate_resume_metadata(
            current,
            {
                **current,
                "variant": "cv_residual",
                "model_config": {
                    **current["model_config"],
                    "variant": "cv_residual",
                    "format_version": WORLD_MODEL_FORMAT_VERSION["cv_residual"],
                },
                "loss_config": CV_RESIDUAL_LOSS_CONFIG,
            },
            "test",
        )
