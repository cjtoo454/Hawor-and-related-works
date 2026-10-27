"""Two-loop HandLoop state restoration and per-frame history feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from .handloop_state import (
    apply_delta,
    make_anchor,
    pack_state,
    to_local,
    to_world,
    unpack_state,
)
from .handloop_twm import TWMConfig, TrajectoryRestorer
from .handloop_visual import HandLoopVisual


Projector = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class HandLoopConfig:
    """HandLoop architecture and nested visual/TWM configuration."""

    architecture: str = "handloop_v2"
    rounds: int = 2
    reobserve: bool = True
    twm: TWMConfig | dict[str, Any] = field(default_factory=TWMConfig)
    visual: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.architecture != "handloop_v2":
            raise ValueError(
                f"architecture must be 'handloop_v2', got {self.architecture!r}")
        if self.rounds < 1:
            raise ValueError("rounds must be at least one")
        if isinstance(self.twm, dict):
            self.twm = TWMConfig(**self.twm)
        self.visual = dict(self.visual)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "HandLoopConfig":
        """Build a config from checkpoint/runtime dictionaries."""
        values = dict(values)
        architecture = values.pop("architecture", values.pop("version", "handloop_v2"))
        rounds = values.pop("rounds", values.pop("correction_steps", None))
        reobserve = values.pop(
            "reobserve", values.pop("intra_frame_reobserve", None))
        visual = dict(values.pop("visual", {}))
        if rounds is None:
            rounds = visual.pop("rounds", 2)
        if reobserve is None:
            reobserve = visual.pop("reobserve", True)
        twm = values.pop("twm", {})
        # Ignore legacy metadata fields that do not affect this architecture.
        values.pop("history_length", None)
        values.pop("state_dim", None)
        return cls(
            architecture=architecture,
            rounds=int(rounds),
            reobserve=bool(reobserve),
            twm=twm,
            visual=visual,
        )


def _identity_state(reference: torch.Tensor) -> torch.Tensor:
    batch = reference.shape[0]
    position = reference.new_zeros(batch, 3)
    rotations = torch.eye(3, dtype=reference.dtype, device=reference.device)
    rotations = rotations.expand(batch, 16, 3, 3)
    return pack_state(position, rotations)


def _last_available_initial_state(
    raw_state: torch.Tensor,
    history: torch.Tensor,
    raw_available: torch.Tensor,
    history_available: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use raw groups first, then the latest available group, then identity."""
    batch, time = history.shape[:2]
    raw_available = raw_available.bool()
    valid = history_available.bool() & history_mask.bool()[..., None]
    if time:
        slots = torch.arange(time, device=history.device).view(1, time, 1)
        latest = torch.where(valid, slots, -1).amax(dim=1)
        has_history = latest >= 0
        batch_slots = torch.arange(batch, device=history.device)[:, None]
        group_states = history[batch_slots, latest.clamp_min(0)]
    else:
        has_history = torch.zeros(batch, 17, dtype=torch.bool, device=raw_state.device)
        group_states = raw_state.new_zeros(batch, 17, 99)

    identity = _identity_state(raw_state)
    position = torch.where(
        raw_available[:, :1], raw_state[:, :3],
        torch.where(has_history[:, :1], group_states[:, 0, :3], identity[:, :3]))
    raw_rotations = raw_state[:, 3:].reshape(batch, 16, 6)
    history_rotations = group_states[:, 1:17, 3:].reshape(batch, 16, 16, 6)
    identity_rotations = identity[:, 3:].reshape(batch, 16, 6)
    rotation_parts = []
    for rotation_index in range(16):
        group_index = rotation_index + 1
        rotation_parts.append(torch.where(
            raw_available[:, group_index:group_index + 1],
            raw_rotations[:, rotation_index],
            torch.where(
                has_history[:, group_index:group_index + 1],
                history_rotations[:, rotation_index, rotation_index],
                identity_rotations[:, rotation_index],
            ),
        ))
    initial = torch.cat((position, torch.stack(rotation_parts, dim=1).flatten(1)), dim=-1)
    initial_available = raw_available | has_history
    return initial, initial_available


def _world_to_camera(
    state: torch.Tensor,
    rotation_w2c: torch.Tensor,
    translation_w2c: torch.Tensor,
) -> torch.Tensor:
    position, rotations = unpack_state(state)
    extra = state.ndim - 2
    rotation = rotation_w2c
    translation = translation_w2c
    for _ in range(extra):
        rotation = rotation.unsqueeze(1)
        translation = translation.unsqueeze(1)
    camera_position = (rotation @ position.unsqueeze(-1)).squeeze(-1) + translation
    camera_rotations = rotations.clone()
    camera_rotations[..., 0, :, :] = rotation @ rotations[..., 0, :, :]
    return pack_state(camera_position, camera_rotations)


def _camera_delta_to_anchor(
    delta_camera: torch.Tensor,
    rotation_w2c: torch.Tensor,
    anchor_rotation: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame position/root tangents into the fixed anchor."""
    camera_to_anchor = anchor_rotation.transpose(-1, -2) @ rotation_w2c.transpose(-1, -2)
    delta_anchor = delta_camera.clone()
    delta_anchor[:, :3] = (camera_to_anchor @ delta_camera[:, :3, None]).squeeze(-1)
    delta_anchor[:, 3:6] = (camera_to_anchor @ delta_camera[:, 3:6, None]).squeeze(-1)
    return delta_anchor


class HandLoopObserver(nn.Module):
    """Run shared visual/TWM updates in one fixed reference frame."""

    def __init__(
        self,
        config: HandLoopConfig | dict[str, Any] | None = None,
        *,
        visual: nn.Module | None = None,
        twm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = HandLoopConfig()
        elif isinstance(config, dict):
            config = HandLoopConfig.from_dict(config)
        self.config = config
        self.visual = visual or HandLoopVisual(**config.visual)
        self.twm = twm or TrajectoryRestorer(config.twm)

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
        """Restore one frame; history is fixed and the final state is not committed here."""
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

        # Completed history is a fixed, detached input throughout both rounds.
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
        current_anchor = to_local(initial_world, anchor_rotation, anchor_position)
        current_available = initial_available
        history_camera = _world_to_camera(
            history, rotation_w2c, translation_w2c)
        raw_camera = _world_to_camera(raw_state, rotation_w2c, translation_w2c)

        first_grid: torch.Tensor | None = None
        first_valid: torch.Tensor | None = None
        round_outputs = []
        final_result: dict[str, Any] | None = None
        twm_handedness = handedness.reshape(batch).long().clamp(0, 1)
        for round_index in range(self.config.rounds):
            candidate_world = to_world(current_anchor, anchor_rotation, anchor_position)
            if round_index == 0 or self.config.reobserve:
                projected_grid, projection_valid = projector(candidate_world)
                if projected_grid.shape != (batch, 16, 2):
                    raise ValueError("projector must return grid [B,16,2]")
                if projection_valid.shape != (batch, 16):
                    raise ValueError("projector must return validity [B,16]")
                if round_index == 0:
                    first_grid, first_valid = projected_grid, projection_valid
            else:
                assert first_grid is not None and first_valid is not None
                projected_grid, projection_valid = first_grid, first_valid

            candidate_camera = _world_to_camera(
                candidate_world, rotation_w2c, translation_w2c)
            delta_camera = self.visual(
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
            if delta_camera.shape != (batch, 51):
                raise ValueError("visual module must return delta [B,51]")
            delta_anchor = _camera_delta_to_anchor(
                delta_camera, rotation_w2c, anchor_rotation)
            candidate_anchor = apply_delta(current_anchor, delta_anchor)
            future_query = future_times if round_index == self.config.rounds - 1 else None
            final_result = self.twm(
                history_anchor,
                raw_anchor,
                candidate_anchor,
                history_available,
                raw_available.bool(),
                current_available,
                history_times,
                dt,
                twm_handedness,
                history_mask.bool(),
                future_query,
            )
            if not isinstance(final_result, dict) or "current" not in final_result:
                raise TypeError("TrajectoryRestorer must return a dict with 'current'")
            current_distribution = final_result["current"]
            if "state" not in current_distribution:
                raise ValueError("TWM current distribution must contain 'state'")
            current_anchor = current_distribution["state"]
            round_outputs.append(current_distribution)

        assert final_result is not None
        final_distribution = final_result["current"]
        final_state_world = to_world(
            final_distribution["state"], anchor_rotation, anchor_position)
        sigma = final_distribution.get("sigma", final_distribution.get("sigma3"))
        kappa = final_distribution.get("kappa", final_distribution.get("kappa16"))
        if sigma is None or kappa is None:
            raise ValueError("TWM current distribution must contain sigma and kappa")
        dispersion = torch.cat((
            sigma.square().mean(dim=-1, keepdim=True).sqrt(), kappa), dim=-1)
        return {
            "state": final_state_world,
            "current": final_distribution,
            "future": final_result.get("future"),
            "anchor_rotation": anchor_rotation,
            "anchor_position": anchor_position,
            "dispersion": dispersion,
            "rounds": round_outputs,
            "balance_loss": final_result.get(
                "balance_loss", final_state_world.new_zeros(())),
        }

    @staticmethod
    def commit_history(
        history: torch.Tensor,
        history_available: torch.Tensor,
        history_dispersion: torch.Tensor,
        dispersion_known: torch.Tensor,
        history_times: torch.Tensor,
        history_mask: torch.Tensor,
        history_handedness: torch.Tensor,
        state: torch.Tensor,
        state_available: torch.Tensor,
        dispersion: torch.Tensor,
        frame_time: torch.Tensor,
        handedness: torch.Tensor,
        *,
        max_length: int = 30,
        reset: torch.Tensor | None = None,
        dispersion_is_known: torch.Tensor | bool = True,
        max_gap: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        """Append one detached completed state, resetting stale tracks first.

        The caller invokes this exactly once after the final round. It returns
        the updated history tensors, including explicit state and dispersion
        known masks, and never stores candidates or predicted future states.
        """
        if state.ndim != 2 or state.shape[-1] != 99:
            raise ValueError("state must have shape [B,99]")
        batch = state.shape[0]
        if state_available.shape != (batch, 17) or dispersion.shape != (batch, 17):
            raise ValueError("state_available and dispersion must have shape [B,17]")
        if frame_time.shape != (batch,) or handedness.shape != (batch,):
            raise ValueError("frame_time and handedness must have shape [B]")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        old_mask = history_mask.bool()
        had_history = old_mask.any(dim=1)
        if old_mask.shape[1]:
            slots = torch.arange(old_mask.shape[1], device=old_mask.device).expand_as(old_mask)
            last_index = torch.where(old_mask, slots, -1).amax(dim=1).clamp_min(0)
            batch_index = torch.arange(batch, device=state.device)
            last_time = history_times[batch_index, last_index]
            last_handedness = history_handedness[batch_index, last_index]
        else:
            last_time = frame_time.new_zeros(batch)
            last_handedness = handedness.new_zeros(batch)
        should_reset = torch.zeros(batch, dtype=torch.bool, device=state.device)
        if reset is not None:
            should_reset |= reset.bool().reshape(batch)
        gap = frame_time - last_time
        should_reset |= had_history & ((gap > max_gap) | (gap < 0))
        should_reset |= had_history & (last_handedness != handedness)
        retained_mask = old_mask & ~should_reset[:, None]
        retained_history = history.detach()
        retained_available = history_available.detach().bool()
        retained_dispersion = history_dispersion.detach()
        retained_known = dispersion_known.detach().bool()
        retained_times = history_times.detach()
        retained_handedness = history_handedness.detach()
        new_state = torch.cat((retained_history, state.detach()[:, None]), dim=1)
        new_available = torch.cat(
            (retained_available, state_available.detach().bool()[:, None]), dim=1)
        new_dispersion = torch.cat(
            (retained_dispersion, dispersion.detach()[:, None]), dim=1)
        known = torch.as_tensor(
            dispersion_is_known, dtype=torch.bool, device=state.device)
        if known.ndim == 0:
            known = known.expand(batch)
        new_known = torch.cat((retained_known, known.reshape(batch, 1)), dim=1)
        new_times = torch.cat((retained_times, frame_time.detach()[:, None]), dim=1)
        new_handedness = torch.cat(
            (retained_handedness, handedness.detach()[:, None]), dim=1)
        new_mask = torch.cat((retained_mask, torch.ones_like(frame_time, dtype=torch.bool)[:, None]), dim=1)
        return {
            "history": new_state[:, -max_length:].detach(),
            "history_available": new_available[:, -max_length:].detach(),
            "history_dispersion": new_dispersion[:, -max_length:].detach(),
            "dispersion_known": new_known[:, -max_length:].detach(),
            "history_times": new_times[:, -max_length:].detach(),
            "history_mask": new_mask[:, -max_length:].detach(),
            "history_handedness": new_handedness[:, -max_length:].detach(),
            "reset": should_reset,
        }
