"""End-to-end dual-loop state reconciliation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .networks import (
    ImageEncoder,
    JointReObserver,
    StateCorrector,
    TrajectoryWorldModel,
    VisualQualityEstimator,
)
from .state import STATE_DIM


Projector = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class DualLoopConfig:
    state_dim: int = STATE_DIM
    feature_dim: int = 64
    evidence_dim: int = 128
    hidden_dim: int = 256
    history_length: int = 16
    correction_steps: int = 2
    image_size: int = 256
    intra_frame_reobserve: bool = True
    corrected_state_feedback: bool = True
    quality_decay_on_missing: float = 0.92
    min_trace_quality: float = 0.20
    world_model_variant: str = "legacy"


class DualLoopObserver(nn.Module):
    def __init__(self, config: DualLoopConfig) -> None:
        super().__init__()
        if config.state_dim != STATE_DIM:
            raise ValueError(f"This implementation expects state_dim={STATE_DIM}")
        self.config = config
        self.image_encoder = ImageEncoder(config.feature_dim)
        self.reobserver = JointReObserver(config.feature_dim, config.evidence_dim)
        self.world_model = TrajectoryWorldModel(
            config.state_dim, config.hidden_dim, variant=config.world_model_variant)
        self.quality_estimator = VisualQualityEstimator(
            config.state_dim, config.feature_dim, config.hidden_dim // 2,
            config.evidence_dim)
        self.corrector = StateCorrector(
            config.state_dim, config.evidence_dim, config.hidden_dim)

    def predict_prior(
        self,
        history: torch.Tensor,
        fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.shape[1] == 0:
            return fallback, torch.zeros_like(fallback)
        return self.world_model(history[:, -self.config.history_length:])

    def forward_step(
        self,
        image: torch.Tensor,
        visual_state: torch.Tensor,
        history: torch.Tensor,
        projector: Projector,
        visual_valid: torch.Tensor,
        quality_hint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prior, prior_logvar = self.predict_prior(history, visual_state)
        valid_column = visual_valid.bool().reshape(visual_state.shape[0], 1)
        candidate = torch.where(valid_column, visual_state, prior)
        feature_map = self.image_encoder(image)
        global_feature = feature_map.mean(dim=(-2, -1))
        valid_feature = valid_column.to(dtype=visual_state.dtype)
        quality_hint = quality_hint.reshape(visual_state.shape[0], -1)[:, :1]
        cached_evidence = None
        cached_visibility = None
        gates = []
        for correction_index in range(self.config.correction_steps):
            should_observe = (
                cached_evidence is None or self.config.intra_frame_reobserve)
            if should_observe:
                grid, visibility = projector(candidate)
                cached_evidence = self.reobserver(feature_map, grid, visibility)
                cached_visibility = visibility.bool()
            assert cached_evidence is not None and cached_visibility is not None
            quality, _ = self.quality_estimator(
                visual_state, prior, global_feature, quality_hint, valid_feature,
                local_evidence=cached_evidence, visibility=cached_visibility,
                candidate_state=candidate, return_logit=True)
            candidate, gate = self.corrector(
                candidate, visual_state, prior, cached_evidence,
                quality, prior_logvar)
            gates.append(gate)

        # Quality describes the corrected state, so its visibility and local
        # evidence must be sampled at the final projection as well.
        final_grid, final_visibility = projector(candidate)
        final_evidence = self.reobserver(feature_map, final_grid, final_visibility)
        visual_quality, visual_quality_logit = self.quality_estimator(
            visual_state, prior, global_feature, quality_hint, valid_feature,
            local_evidence=final_evidence, visibility=final_visibility,
            candidate_state=candidate, return_logit=True)
        visibility_fraction = final_visibility.float().mean(dim=-1, keepdim=True)

        # With no valid frontend state, retain only a low-confidence prior
        # trace. The current frame can still provide local evidence to the
        # corrector, while invisibility always forces trace quality to zero.
        prior_quality = self.config.min_trace_quality * torch.exp(
            -prior_logvar.exp().mean(dim=-1, keepdim=True))
        prior_quality = prior_quality * visibility_fraction
        final_quality = torch.where(valid_column, visual_quality, prior_quality)
        return {
            "state": candidate,
            "prior": prior,
            "prior_logvar": prior_logvar,
            "prior_variance": prior_logvar.exp(),
            "quality": final_quality,
            "visual_quality": visual_quality,
            "visual_quality_logit": visual_quality_logit,
            "final_visibility": final_visibility.bool(),
            "visibility_fraction": visibility_fraction,
            "gates": (
                torch.cat(gates, dim=-1)
                if gates else candidate.new_empty(candidate.shape[0], 0)
            ),
        }

