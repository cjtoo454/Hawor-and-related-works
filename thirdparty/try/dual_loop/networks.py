"""Trainable components for the dual-loop observer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .state import (
    STATE_DIM,
    project_state,
    pack_state,
    state_delta_scale,
    unpack_state,
)


WORLD_MODEL_VARIANTS = ("legacy", "cv_residual")
WORLD_MODEL_FORMAT_VERSION = {"legacy": 1, "cv_residual": 2}


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class JointReObserver(nn.Module):
    """Read image evidence again at the candidate state's projected joints."""

    def __init__(self, feature_dim: int = 64, evidence_dim: int = 128) -> None:
        super().__init__()
        self.joint_encoder = nn.Sequential(
            nn.Linear(feature_dim + 3, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
            nn.GELU(),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        joint_grid: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        grid = joint_grid.unsqueeze(-2)
        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        joint_input = torch.cat(
            (sampled, joint_grid, visibility.unsqueeze(-1).float()), dim=-1)
        encoded = self.joint_encoder(joint_input)
        weights = visibility.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled


class TrajectoryWorldModel(nn.Module):
    """Causal motion prior evaluated in a first-frame wrist-local gauge.

    Callers pass ordinary world-frame states. Internally the first wrist
    translation and root rotation are the origin and orientation of the
    sequence gauge. Predictions are transformed back to world coordinates
    before returning, so training and inference share the same convention.
    Translation uncertainty is isotropic in world coordinates.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 256,
        variant: str = "legacy",
    ) -> None:
        super().__init__()
        if state_dim != STATE_DIM:
            raise ValueError(f"This implementation expects state_dim={STATE_DIM}")
        if variant not in WORLD_MODEL_VARIANTS:
            raise ValueError(
                f"variant must be one of {WORLD_MODEL_VARIANTS}, got {variant!r}")
        self.state_dim = state_dim
        self.variant = variant
        gru_input_dim = state_dim
        if variant == "cv_residual":
            # Keep the original state sequence while giving the residual head
            # an explicit, normalized one-frame velocity feature.
            self.residual_head = nn.Linear(hidden_dim + state_dim, state_dim)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            self.register_buffer(
                "residual_scale",
                state_delta_scale(translation=0.03, rotation=0.08, shape=0.02),
            )
        else:
            self.delta_head = nn.Linear(hidden_dim, state_dim)
        self.gru = nn.GRU(gru_input_dim, hidden_dim, batch_first=True)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
        if variant == "legacy":
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    @staticmethod
    def _to_local(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        translation, root, hand, betas = unpack_state(history)
        anchor_rotation = root[:, 0, 0]
        anchor_translation = translation[:, 0]
        inverse_anchor = anchor_rotation.transpose(-1, -2)
        local_translation = torch.matmul(
            inverse_anchor[:, None],
            (translation - anchor_translation[:, None]).unsqueeze(-1),
        ).squeeze(-1)
        local_root = torch.matmul(inverse_anchor[:, None, None], root)
        local_history = pack_state(local_translation, local_root, hand, betas)
        return project_state(local_history), anchor_rotation, anchor_translation

    @staticmethod
    def _to_world(
        local_state: torch.Tensor,
        anchor_rotation: torch.Tensor,
        anchor_translation: torch.Tensor,
    ) -> torch.Tensor:
        translation, root, hand, betas = unpack_state(local_state)
        local_translation = torch.matmul(
            anchor_rotation[:, None], translation.unsqueeze(-1)
        ).squeeze(-1)
        world_translation = local_translation + anchor_translation[:, None]
        world_root = torch.matmul(anchor_rotation[:, None, None], root)
        return project_state(pack_state(world_translation, world_root, hand, betas))

    @staticmethod
    def _world_logvar(logvar: torch.Tensor) -> torch.Tensor:
        # A scalar translation variance is invariant to the anchor rotation.
        # Root 6D rotation channels are likewise grouped under the same
        # rotation transform; hand pose and shape remain in their original
        # coordinates.
        logvar = logvar.clone()
        logvar[..., :3] = logvar[..., :3].mean(dim=-1, keepdim=True)
        logvar[..., 3:9] = logvar[..., 3:9].mean(dim=-1, keepdim=True)
        return logvar

    @staticmethod
    def _to_local_state(
        state: torch.Tensor,
        anchor_rotation: torch.Tensor,
        anchor_translation: torch.Tensor,
    ) -> torch.Tensor:
        translation, root, hand, betas = unpack_state(state)
        inverse_anchor = anchor_rotation.transpose(-1, -2)
        local_translation = torch.matmul(
            inverse_anchor, (translation - anchor_translation).unsqueeze(-1)
        ).squeeze(-1)
        local_root = torch.matmul(inverse_anchor[:, None], root)
        return project_state(pack_state(local_translation, local_root, hand, betas))

    def _predict_one(
        self,
        history: torch.Tensor,
        *,
        cv_anchor: torch.Tensor | None = None,
        cv_step: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_history, anchor_rotation, anchor_translation = self._to_local(history)
        output, _ = self.gru(local_history)
        hidden = output[:, -1]
        if self.variant == "legacy":
            local_current = project_state(local_history[:, -1] + self.delta_head(hidden))
            world_prediction = self._to_world(
                local_current[:, None], anchor_rotation, anchor_translation)[:, 0]
        else:
            if cv_anchor is None:
                cv_anchor = history
            if cv_step < 1:
                raise ValueError("cv_step must be at least one")
            current = cv_anchor[:, -1]
            previous = cv_anchor[:, -2] if cv_anchor.shape[1] >= 2 else current
            baseline = project_state(current + cv_step * (current - previous))

            if local_history.shape[1] >= 2:
                velocity = (local_history[:, -1] - local_history[:, -2])
            else:
                velocity = torch.zeros_like(local_history[:, -1])
            velocity = (velocity / self.residual_scale).clamp(-10.0, 10.0)
            residual = torch.tanh(self.residual_head(
                torch.cat((hidden, velocity), dim=-1)))
            residual = residual * self.residual_scale * (cv_step ** 0.5)

            local_baseline = self._to_local_state(
                baseline, anchor_rotation, anchor_translation)
            local_candidate = project_state(local_baseline + residual)
            world_candidate = self._to_world(
                local_candidate[:, None], anchor_rotation, anchor_translation)[:, 0]
            zero_residual = residual.eq(0).all(dim=-1, keepdim=True)
            # Return the exact baseline at initialization while keeping a
            # straight-through gradient through the projected candidate.
            world_prediction = torch.where(
                zero_residual,
                baseline + (world_candidate - world_candidate.detach()),
                world_candidate,
            )
        logvar = self.logvar_head(hidden).clamp(-8.0, 4.0)
        return world_prediction, self._world_logvar(logvar)

    def forward(
        self, history: torch.Tensor, steps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[1] < 1:
            raise ValueError("history must have shape [batch, time, state_dim]")
        if history.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, got {history.shape[-1]}"
            )
        if steps < 1:
            raise ValueError("steps must be at least one")
        if steps == 1:
            return self._predict_one(history, cv_anchor=history, cv_step=1)

        # Re-encode the finite world-frame history at every horizon step. This
        # matches inference, where each corrected prediction becomes history
        # and the next frame establishes its own local gauge.
        history_limit = history.shape[1]
        rolling_history = history
        predictions = []
        logvars = []
        for _ in range(steps):
            prediction, logvar = self._predict_one(
                rolling_history, cv_anchor=rolling_history, cv_step=1)
            predictions.append(prediction)
            logvars.append(logvar)
            rolling_history = torch.cat(
                (rolling_history, prediction.unsqueeze(1)), dim=1
            )[:, -history_limit:]
        return torch.stack(predictions, dim=1), torch.stack(logvars, dim=1)


class VisualQualityEstimator(nn.Module):
    """Estimate confidence from global and candidate-local image evidence.

    ``quality_hint`` is an input feature for calibration, not a confidence floor.
    The returned probability is additionally constrained by the fraction of
    projected joints with visible image support.
    """

    def __init__(
        self,
        state_dim: int,
        feature_dim: int,
        hidden_dim: int = 128,
        evidence_dim: int = 128,
    ) -> None:
        super().__init__()
        self.evidence_dim = evidence_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim + feature_dim + evidence_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        visual_state: torch.Tensor,
        prior_state: torch.Tensor,
        global_feature: torch.Tensor,
        quality_hint: torch.Tensor,
        valid: torch.Tensor,
        local_evidence: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        candidate_state: torch.Tensor | None = None,
        return_logit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return calibrated quality and, optionally, its trainable raw logit.

        The original five positional arguments remain supported. When local
        evidence or visibility is omitted, zero evidence and full visibility
        are assumed for callers using that legacy form.
        """
        if candidate_state is None:
            candidate_state = visual_state
        innovation = (candidate_state - prior_state).abs()
        innovation_mean = innovation.mean(dim=-1, keepdim=True)
        if local_evidence is None:
            local_evidence = global_feature.new_zeros(
                global_feature.shape[0], self.evidence_dim)
        if visibility is None:
            visible_fraction = global_feature.new_ones(global_feature.shape[0], 1)
        else:
            visible_fraction = visibility.float().mean(dim=-1, keepdim=True)
        hint = quality_hint.reshape(quality_hint.shape[0], -1)[:, :1]
        valid_feature = valid.float().reshape(valid.shape[0], -1)[:, :1]
        inputs = torch.cat(
            (innovation, global_feature, local_evidence, innovation_mean,
             hint, valid_feature, visible_fraction), dim=-1)
        logit = self.network(inputs)
        quality = torch.sigmoid(logit) * visible_fraction
        if return_logit:
            return quality, logit
        return quality


class StateCorrector(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        evidence_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = state_dim * 3 + evidence_dim + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim + 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("delta_scale", state_delta_scale())

    def forward(
        self,
        candidate: torch.Tensor,
        visual_state: torch.Tensor,
        prior_state: torch.Tensor,
        evidence: torch.Tensor,
        visual_quality: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uncertainty = prior_logvar.mean(dim=-1, keepdim=True).exp()
        inputs = torch.cat(
            (candidate, visual_state, prior_state, evidence,
             visual_quality, uncertainty), dim=-1)
        raw = self.network(inputs)
        delta = torch.tanh(raw[..., :-1]) * self.delta_scale
        gate = torch.sigmoid(raw[..., -1:]) * visual_quality
        corrected = project_state(candidate + gate * delta)
        return corrected, gate

