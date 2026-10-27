"""Centered two-round intra-frame feedback for the HandLoop observer.

The second visual update is applied to the state it observed. A reference TWM
call removes the TWM's response to that state before the innovation is added.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import nn

from dual_loop.handloop_model import (
    HandLoopObserver,
    _camera_delta_to_anchor,
    _last_available_initial_state,
    _world_to_camera,
)
from dual_loop.handloop_state import (
    apply_delta, make_anchor, so3_log, to_local, to_world, unpack_state,
)


INTRA_ARCHITECTURE = "handloop_intra_v4"
DEFAULT_TRANSLATION_BOUND = 0.001
DEFAULT_ROOT_ROTATION_BOUND = 0.01
DEFAULT_FINGER_ROTATION_BOUND = 0.05
Projector = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _validate_bound(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _bounded_vector(vector: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    """Bound a 3-vector radially while keeping small physical deltas unchanged."""
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scaled_norm = norm / maximum_norm
    eps = torch.finfo(vector.dtype).eps
    # tanh(r) / r tends to one at zero. Scaling by the configured bound makes
    # the output norm max*tanh(norm/max), preserving the input's physical units
    # for small deltas while smoothly saturating at max.
    ratio = torch.where(
        scaled_norm > eps,
        torch.tanh(scaled_norm) / scaled_norm.clamp_min(eps),
        torch.ones_like(norm),
    )
    return vector * ratio


def bound_visual_delta(
    delta: torch.Tensor,
    *,
    translation_bound: float = DEFAULT_TRANSLATION_BOUND,
    root_rotation_bound: float = DEFAULT_ROOT_ROTATION_BOUND,
    finger_rotation_bound: float = DEFAULT_FINGER_ROTATION_BOUND,
) -> torch.Tensor:
    """Apply tanh bounds to position, root, and each finger tangent vector.

    The input layout is the HandLoop 51D camera tangent: translation (3), root
    rotation (3), then fifteen independent finger rotation vectors (15x3).
    Each configured value bounds a vector norm, rather than each coordinate.
    """
    if delta.ndim != 2 or delta.shape[-1] != 51:
        raise ValueError("visual delta must have shape [B,51]")
    if not delta.is_floating_point():
        raise TypeError("visual delta must use a floating-point dtype")
    bounds = (
        _validate_bound("translation_bound", translation_bound),
        _validate_bound("root_rotation_bound", root_rotation_bound),
        _validate_bound("finger_rotation_bound", finger_rotation_bound),
    )
    if not torch.isfinite(delta).all():
        raise ValueError("valid visual delta contains NaN or Inf")
    position = _bounded_vector(delta[:, :3], bounds[0])
    root_rotation = _bounded_vector(delta[:, 3:6], bounds[1])
    finger_rotation = _bounded_vector(
        delta[:, 6:].reshape(delta.shape[0], 15, 3), bounds[2]
    ).flatten(1)
    return torch.cat((position, root_rotation, finger_rotation), dim=-1)


def _apply_delta_exact_zero(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Keep an exact zero update while retaining its tangent derivative."""
    updated = apply_delta(state, delta)
    zero = (delta == 0).all(dim=-1, keepdim=True)
    correction = updated - state
    return torch.where(zero, state, updated) + zero.to(state.dtype) * (
        correction - correction.detach())


def _twm_state(result: dict[str, Any]) -> torch.Tensor:
    if not isinstance(result, dict) or "current" not in result:
        raise TypeError("TrajectoryRestorer must return a dict with 'current'")
    distribution = result["current"]
    if not isinstance(distribution, dict) or "state" not in distribution:
        raise ValueError("TWM current distribution must contain 'state'")
    return distribution["state"]


def _innovation(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    candidate_position, candidate_rotation = unpack_state(candidate)
    reference_position, reference_rotation = unpack_state(reference)
    rotation_delta = so3_log(candidate_rotation @ reference_rotation.transpose(-1, -2))
    return torch.cat((candidate_position - reference_position, rotation_delta.flatten(1)), dim=-1)


class CenteredFeedbackObserver(HandLoopObserver):
    """Two-round observer whose second innovation is centered on its observation."""

    architecture = INTRA_ARCHITECTURE

    def __init__(
        self,
        base_observer: HandLoopObserver,
        *,
        translation_bound: float = DEFAULT_TRANSLATION_BOUND,
        root_rotation_bound: float = DEFAULT_ROOT_ROTATION_BOUND,
        finger_rotation_bound: float = DEFAULT_FINGER_ROTATION_BOUND,
    ) -> None:
        if not isinstance(base_observer, HandLoopObserver):
            raise TypeError("base_observer must be a HandLoopObserver")
        # Reuse the exact visual and TWM objects and their state_dict keys.
        base_observer.config.rounds = 2
        base_observer.config.reobserve = True
        super().__init__(base_observer.config, visual=base_observer.visual,
                         twm=base_observer.twm)
        self.translation_bound = _validate_bound("translation_bound", translation_bound)
        self.root_rotation_bound = _validate_bound("root_rotation_bound", root_rotation_bound)
        self.finger_rotation_bound = _validate_bound("finger_rotation_bound", finger_rotation_bound)
        scale = getattr(self.visual, "delta_scale", None)
        if scale is not None:
            if scale.shape != (51,):
                raise ValueError("visual.delta_scale must have shape [51]")
            with torch.no_grad():
                scale.copy_(scale.new_tensor(
                    [self.translation_bound] * 3
                    + [self.root_rotation_bound] * 3
                    + [self.finger_rotation_bound] * 45))
        self.feedback_metadata = {
            "architecture": self.architecture,
            "rounds": 2,
            "reobserve": True,
            "delta_bounds": {
                "translation_m": self.translation_bound,
                "root_rotation_rad": self.root_rotation_bound,
                "finger_rotation_rad": self.finger_rotation_bound,
            },
            "visual_head_scale": {
                "translation_m": self.translation_bound,
                "root_rotation_rad": self.root_rotation_bound,
                "finger_rotation_rad": self.finger_rotation_bound,
            },
            "distribution_nll_valid": False,
        }

    def forward_step(
        self,
        patches: torch.Tensor,
        raw_state: torch.Tensor,
        history: torch.Tensor,
        raw_available: torch.Tensor,
        history_available: torch.Tensor,
        history_mask: torch.Tensor,
        history_times: torch.Tensor,
        dt: torch.Tensor,
        handedness: torch.Tensor,
        history_dispersion: torch.Tensor,
        dispersion_known: torch.Tensor,
        rotation_w2c: torch.Tensor,
        translation_w2c: torch.Tensor,
        projector: Projector,
        image_available: torch.Tensor,
        future_times: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Restore one frame with a centered second-round TWM innovation."""
        if self.config.rounds != 2 or not self.config.reobserve:
            raise ValueError("handloop_intra_v4 requires exactly two reobserving rounds")
        if future_times is not None:
            raise ValueError("handloop_intra_v4 does not support future_times")
        if raw_state.ndim != 2 or raw_state.shape[-1] != 99:
            raise ValueError("raw_state must have shape [B,99]")
        batch = raw_state.shape[0]
        if history.ndim != 3 or history.shape[0] != batch or history.shape[-1] != 99:
            raise ValueError("history must have shape [B,T,99]")
        time = history.shape[1]
        if raw_available.shape != (batch, 17):
            raise ValueError("raw_available must have shape [B,17]")
        if history_available.shape != (batch, time, 17):
            raise ValueError("history_available must have shape [B,T,17]")
        if history_mask.shape != (batch, time) or history_times.shape != (batch, time):
            raise ValueError("history_mask and history_times must have shape [B,T]")
        if history_dispersion.shape != (batch, time, 17):
            raise ValueError("history_dispersion must have shape [B,T,17]")
        if dispersion_known.shape != (batch, time):
            raise ValueError("dispersion_known must have shape [B,T]")
        if rotation_w2c.shape != (batch, 3, 3) or translation_w2c.shape != (batch, 3):
            raise ValueError("camera transforms must have shapes [B,3,3] and [B,3]")
        if image_available.shape != (batch,):
            raise ValueError("image_available must have shape [B]")

        # Match the original observer's raw/history fallback and anchor choice.
        history = history.detach()
        history_dispersion = history_dispersion.detach()
        history_available = history_available.bool() & history_mask.bool()[..., None]
        initial_world, initial_available = _last_available_initial_state(
            raw_state, history, raw_available, history_available, history_mask)
        anchor_rotation, anchor_position = make_anchor(
            initial_world, history, initial_available, history_available)
        anchor_rotation = anchor_rotation.detach()
        anchor_position = anchor_position.detach()

        raw_anchor = to_local(raw_state, anchor_rotation, anchor_position)
        history_anchor = to_local(history, anchor_rotation, anchor_position)
        initial_anchor = to_local(initial_world, anchor_rotation, anchor_position)
        projection_state_anchor = initial_anchor
        current_available = initial_available
        history_camera = _world_to_camera(history, rotation_w2c, translation_w2c)
        raw_camera = _world_to_camera(raw_state, rotation_w2c, translation_w2c)
        twm_handedness = handedness.reshape(batch).long().clamp(0, 1)

        round_outputs = []
        projection_states = []
        visual_candidates = []
        visual_deltas_camera = []
        visual_deltas_anchor = []
        first_result: dict[str, Any] | None = None
        candidate_result: dict[str, Any] | None = None
        reference_result: dict[str, Any] | None = None

        for round_index in range(2):
            projection_state_world = to_world(
                projection_state_anchor, anchor_rotation, anchor_position)
            projected_grid, projection_valid = projector(projection_state_world)
            if projected_grid.shape != (batch, 16, 2):
                raise ValueError("projector must return grid [B,16,2]")
            if projection_valid.shape != (batch, 16):
                raise ValueError("projector must return validity [B,16]")
            valid_projection = projection_valid.bool() & torch.isfinite(projected_grid).all(-1)
            candidate_camera = _world_to_camera(
                projection_state_world, rotation_w2c, translation_w2c)
            raw_delta_camera = self.visual(
                patches=patches,
                raw_camera=raw_camera,
                candidate_camera=candidate_camera,
                history_camera=history_camera,
                history_dispersion=history_dispersion,
                history_available=history_available,
                history_mask=history_mask,
                history_times=history_times,
                dispersion_known=dispersion_known,
                projected_grid=projected_grid,
                projection_valid=projection_valid,
                handedness=handedness,
                image_available=image_available,
            )
            if raw_delta_camera.shape != (batch, 51):
                raise ValueError("visual module must return delta [B,51]")
            root_valid = valid_projection[:, :1] & image_available.bool()[:, None]
            finger_valid = valid_projection[:, 1:] & image_available.bool()[:, None]
            valid_delta = torch.cat((
                root_valid.expand(-1, 6),
                finger_valid.repeat_interleave(3, dim=-1)), dim=-1)
            if (valid_delta & ~torch.isfinite(raw_delta_camera)).any():
                raise ValueError("valid visual delta contains NaN or Inf")
            raw_delta_camera = torch.where(
                valid_delta, raw_delta_camera, torch.zeros_like(raw_delta_camera))
            delta_camera = bound_visual_delta(
                raw_delta_camera,
                translation_bound=self.translation_bound,
                root_rotation_bound=self.root_rotation_bound,
                finger_rotation_bound=self.finger_rotation_bound,
            )
            delta_anchor = _camera_delta_to_anchor(
                delta_camera, rotation_w2c, anchor_rotation)

            visual_candidate_anchor = _apply_delta_exact_zero(projection_state_anchor, delta_anchor)
            candidate_result = self.twm(
                history_anchor, raw_anchor, visual_candidate_anchor,
                history_available, raw_available.bool(), current_available,
                history_times, dt, twm_handedness, history_mask.bool(), None)
            candidate_state = _twm_state(candidate_result)
            round_outputs.append(candidate_result["current"])
            projection_states.append(projection_state_world)
            visual_candidates.append(to_world(
                visual_candidate_anchor, anchor_rotation, anchor_position))
            visual_deltas_camera.append(delta_camera)
            visual_deltas_anchor.append(delta_anchor)
            if round_index == 0:
                first_result = candidate_result
                projection_state_anchor = candidate_state
            else:
                reference_result = self.twm(
                    history_anchor, raw_anchor, projection_state_anchor,
                    history_available, raw_available.bool(), current_available,
                    history_times, dt, twm_handedness, history_mask.bool(), None)
                reference_state = _twm_state(reference_result)
                innovation = _innovation(candidate_state, reference_state)
                final_anchor = _apply_delta_exact_zero(projection_state_anchor, innovation)
                # When the observed delta is exactly zero, both TWM inputs are
                # identical. Preserve the first restored state bit-for-bit,
                # while the tangent derivative still passes through innovation.
                zero_second = (delta_anchor == 0).all(dim=-1, keepdim=True)
                correction = final_anchor - projection_state_anchor
                final_anchor = torch.where(
                    zero_second, projection_state_anchor, final_anchor) + (
                        zero_second.to(final_anchor.dtype)
                        * (correction - correction.detach()))

        assert first_result is not None and candidate_result is not None
        assert reference_result is not None
        current_distribution = dict(candidate_result["current"])
        current_distribution["state"] = final_anchor
        semantics = {
            "state": "centered_twm_innovation_in_anchor_frame",
            "sigma_kappa": "candidate_twm_proxy; not calibrated for centered state",
            "nll_valid": False,
        }
        current_distribution["distribution_semantics"] = semantics
        final_state_world = to_world(final_anchor, anchor_rotation, anchor_position)
        sigma = current_distribution.get("sigma", current_distribution.get("sigma3"))
        kappa = current_distribution.get("kappa", current_distribution.get("kappa16"))
        if sigma is None or kappa is None:
            raise ValueError("TWM current distribution must contain sigma and kappa")
        dispersion = torch.cat((
            sigma.square().mean(dim=-1, keepdim=True).sqrt(), kappa), dim=-1)
        return {
            "state": final_state_world,
            "current": current_distribution,
            "future": None,
            "anchor_rotation": anchor_rotation,
            "anchor_position": anchor_position,
            "dispersion": dispersion,
            "rounds": round_outputs,
            "balance_loss": candidate_result.get(
                "balance_loss", final_state_world.new_zeros(())),
            "projection_states": projection_states,
            "visual_candidates": visual_candidates,
            "visual_deltas_camera": visual_deltas_camera,
            "visual_deltas_anchor": visual_deltas_anchor,
            "reference_distribution": reference_result["current"],
            "candidate_distribution": candidate_result["current"],
            "distribution_semantics": semantics,
            "feedback_metadata": self.feedback_metadata,
            "innovation_anchor": innovation,
        }


def install_feedback_v4(
    system: nn.Module,
    *,
    translation_bound: float = DEFAULT_TRANSLATION_BOUND,
    root_rotation_bound: float = DEFAULT_ROOT_ROTATION_BOUND,
    finger_rotation_bound: float = DEFAULT_FINGER_ROTATION_BOUND,
) -> CenteredFeedbackObserver:
    """Install centered feedback on a system returned by ``build_system``.

    HaWoR and TWM parameters are frozen, while the visual head stays trainable.
    No ``no_grad`` context is introduced around TWM, so gradients still flow
    from its output to visual candidate inputs.
    """
    base_observer = getattr(system, "observer", None)
    if not isinstance(base_observer, HandLoopObserver):
        raise TypeError("system.observer must be the original HandLoopObserver")
    observer = CenteredFeedbackObserver(
        base_observer,
        translation_bound=translation_bound,
        root_rotation_bound=root_rotation_bound,
        finger_rotation_bound=finger_rotation_bound,
    )
    if hasattr(system, "hawor"):
        system.hawor.requires_grad_(False)
    observer.twm.requires_grad_(False)
    observer.visual.requires_grad_(True)
    system.observer = observer
    return observer


__all__ = [
    "INTRA_ARCHITECTURE",
    "DEFAULT_TRANSLATION_BOUND",
    "DEFAULT_ROOT_ROTATION_BOUND",
    "DEFAULT_FINGER_ROTATION_BOUND",
    "CenteredFeedbackObserver",
    "bound_visual_delta",
    "install_feedback_v4",
]
