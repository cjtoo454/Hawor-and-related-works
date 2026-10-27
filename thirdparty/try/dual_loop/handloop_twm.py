"""Numerical trajectory world model for hand-state restoration.

The temporal block is a small, differentiable reference implementation of the
selective SSM used by Mamba-1. It keeps the selective input-dependent time
step, state-space parameters, causal depthwise convolution, and gated output;
it does not substitute a recurrent neural network or attention when the CUDA
extension is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .handloop_state import apply_delta, pack_state, so3_exp, unpack_state


STATE_SIZE = 99
JOINT_COUNT = 16
AVAILABILITY_GROUPS = 17
TIME_FREQUENCIES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass
class TWMConfig:
    width: int = 256
    layers: int = 6
    heads: int = 4
    state_size: int = 64
    conv_kernel: int = 4
    expand: int = 2
    experts: int = 4
    history_length: int = 30
    modes: int = 4
    future_steps: int = 60


def _time_features(tau: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
    """Encode physical seconds and the interval leading to each token."""
    pieces = [
        tau.unsqueeze(-1),
        interval.unsqueeze(-1),
        torch.log1p(interval.clamp_min(0)).unsqueeze(-1),
    ]
    for frequency in TIME_FREQUENCIES:
        phase = tau * (2.0 * math.pi * frequency)
        pieces.extend((torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)))
    return torch.cat(pieces, dim=-1)


def _history_intervals(
    times: torch.Tensor, mask: torch.Tensor, current_dt: torch.Tensor
) -> torch.Tensor:
    """Get each history row's preceding real timestamp gap, ignoring padding.

    The first real row has no in-window predecessor, so it uses the gap to the
    next real row as its interval estimate (or current_dt for a one-row window).
    """
    batch, length = times.shape
    if length == 0:
        return times.new_zeros(batch, 0)

    next_time = times.new_zeros(batch)
    has_next = torch.zeros(batch, dtype=torch.bool, device=times.device)
    next_intervals: list[torch.Tensor] = [times.new_zeros(batch) for _ in range(length)]
    for index in range(length - 1, -1, -1):
        valid = mask[:, index]
        regular_gap = (next_time - times[:, index]).clamp_min(0.0)
        interval = torch.where(has_next, regular_gap, current_dt)
        next_intervals[index] = torch.where(valid, interval, torch.zeros_like(interval))
        next_time = torch.where(valid, times[:, index], next_time)
        has_next = has_next | valid

    previous_time = times.new_zeros(batch)
    has_previous = torch.zeros(batch, dtype=torch.bool, device=times.device)
    intervals: list[torch.Tensor] = []
    for index in range(length):
        valid = mask[:, index]
        preceding_gap = (times[:, index] - previous_time).clamp_min(0.0)
        interval = torch.where(has_previous, preceding_gap, next_intervals[index])
        intervals.append(torch.where(valid, interval, torch.zeros_like(interval)))
        previous_time = torch.where(valid, times[:, index], previous_time)
        has_previous = has_previous | valid
    return torch.stack(intervals, dim=1)


class _StateSlotEncoder(nn.Module):
    """Encode one state source into its own set of sixteen joint tokens."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.root = nn.Linear(9, width)
        self.joint = nn.Linear(6, width)
        self.global_availability = nn.Linear(AVAILABILITY_GROUPS, width)
        self.local_availability = nn.Linear(2, width)

    def forward(self, state: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        leading = state.shape[:-1]
        position = state[..., :3] / 0.05
        rotation_6d = state[..., 3:].reshape(*leading, JOINT_COUNT, 6)

        root_token = self.root(torch.cat((position, rotation_6d[..., :1, :].squeeze(-2)), -1))
        joint_tokens = self.joint(rotation_6d[..., 1:, :])
        tokens = torch.cat((root_token.unsqueeze(-2), joint_tokens), dim=-2)

        availability = available.to(dtype=tokens.dtype)
        global_context = self.global_availability(availability).unsqueeze(-2)
        root_local = torch.stack((availability[..., 0], availability[..., 1]), dim=-1)
        joint_valid = availability[..., 2:]
        joint_local = torch.stack((joint_valid, torch.zeros_like(joint_valid)), dim=-1)
        local = torch.cat((root_local.unsqueeze(-2), joint_local), dim=-2)
        return tokens + global_context + self.local_availability(local)


class _SelectiveMamba(nn.Module):
    """Mamba-1 selective SSM evaluated by a differentiable PyTorch scan."""

    def __init__(
        self, width: int, state_size: int, conv_kernel: int, expand: int
    ) -> None:
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.inner = width * expand
        self.dt_rank = max(1, math.ceil(width / 16))

        self.in_proj = nn.Linear(width, self.inner * 2, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.inner, conv_kernel))
        self.conv_bias = nn.Parameter(torch.zeros(self.inner))
        self.x_proj = nn.Linear(self.inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner, bias=True)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        with torch.no_grad():
            initial_dt = torch.exp(
                torch.empty(self.inner).uniform_(math.log(1e-3), math.log(1e-1))
            ).clamp_min(1e-4)
            self.dt_proj.bias.copy_(initial_dt + torch.log(-torch.expm1(-initial_dt)))
        state_values = torch.arange(1, state_size + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(state_values.log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Input is [batch, time, joints, width]. Every joint has its own causal
        # state, while parameters are shared across joints and examples.
        batch, length, joints, _ = value.shape
        if length == 0:
            return value
        sequence = value.permute(0, 2, 1, 3).reshape(batch * joints, length, self.width)
        sequence_valid = valid[:, None, :].expand(batch, joints, length).reshape(batch * joints, length)

        projected = self.in_proj(sequence)
        x, gate = projected.chunk(2, dim=-1)
        conv_state = x.new_zeros(batch * joints, self.inner, max(0, self.conv_kernel - 1))
        ssm_state = x.new_zeros(batch * joints, self.inner, self.state_size)
        A = -torch.exp(self.A_log.to(dtype=x.dtype))
        outputs: list[torch.Tensor] = []

        for index in range(length):
            active = sequence_valid[:, index]
            current_x = x[:, index]
            window = torch.cat((conv_state, current_x.unsqueeze(-1)), dim=-1)
            conv_value = (window * self.conv_weight.to(dtype=x.dtype).unsqueeze(0)).sum(-1)
            conv_value = F.silu(conv_value + self.conv_bias.to(dtype=x.dtype))

            parameters = self.x_proj(conv_value)
            dt_low, B, C = torch.split(
                parameters, (self.dt_rank, self.state_size, self.state_size), dim=-1
            )
            delta = F.softplus(self.dt_proj(dt_low))
            decay = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
            next_ssm = decay * ssm_state + (
                delta.unsqueeze(-1) * B.unsqueeze(1) * conv_value.unsqueeze(-1)
            )
            readout = (next_ssm * C.unsqueeze(1)).sum(-1)
            readout = readout + self.D.to(dtype=x.dtype).unsqueeze(0) * conv_value
            output = self.out_proj(readout * F.silu(gate[:, index]))
            outputs.append(torch.where(active.unsqueeze(-1), output, torch.zeros_like(output)))

            if self.conv_kernel > 1:
                shifted = window[..., 1:]
                conv_state = torch.where(active[:, None, None], shifted, conv_state)
            ssm_state = torch.where(active[:, None, None], next_ssm, ssm_state)

        result = torch.stack(outputs, dim=1).reshape(batch, joints, length, self.width)
        return result.permute(0, 2, 1, 3)


class _MambaResidual(nn.Module):
    def __init__(
        self, width: int, state_size: int, conv_kernel: int, expand: int
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ssm = _SelectiveMamba(width, state_size, conv_kernel, expand)

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        value = value + self.ssm(self.norm(value), valid)
        return value * valid[:, :, None, None].to(dtype=value.dtype)


class _TrajectoryLayer(nn.Module):
    def __init__(self, config: TWMConfig) -> None:
        super().__init__()
        width = config.width
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial = nn.MultiheadAttention(
            width, config.heads, dropout=0.0, batch_first=True
        )
        self.spatial_ff_norm = nn.LayerNorm(width)
        self.spatial_ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.temporal = _MambaResidual(
            width, config.state_size, config.conv_kernel, config.expand
        )
        self.expert_norm = nn.LayerNorm(width)
        self.router = nn.Linear(width, config.experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
            )
            for _ in range(config.experts)
        )

    def _mask_padding(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return value * valid[:, :, None, None].to(dtype=value.dtype)

    def forward(
        self, value: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, joints, width = value.shape
        flat = value.reshape(batch * length, joints, width)
        normalized = self.spatial_norm(flat)
        attended, _ = self.spatial(normalized, normalized, normalized, need_weights=False)
        value = value + attended.reshape(batch, length, joints, width)
        value = self._mask_padding(value, valid)
        value = value + self.spatial_ff(self.spatial_ff_norm(value))
        value = self._mask_padding(value, valid)

        value = self.temporal(value, valid)
        normalized = self.expert_norm(value)
        routing = torch.softmax(self.router(normalized), dim=-1)
        expert_outputs = torch.stack([expert(normalized) for expert in self.experts], dim=-2)
        value = value + (routing.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        value = self._mask_padding(value, valid)

        token_valid = valid[:, :, None, None].to(dtype=routing.dtype)
        route_mean = (routing * token_valid).sum(dim=(0, 1, 2))
        route_mean = route_mean / (token_valid.sum() * joints).clamp_min(1.0)
        balance = routing.shape[-1] * route_mean.square().sum() - 1.0
        return value, balance


class TrajectoryRestorer(nn.Module):
    """Restore a current MANO state and optionally predict whole future modes."""

    def __init__(self, config: TWMConfig) -> None:
        super().__init__()
        if config.heads <= 0 or config.width <= 0 or config.width % config.heads:
            raise ValueError("width must be positive and divisible by heads")
        if config.layers <= 0 or config.state_size <= 0 or config.conv_kernel <= 0:
            raise ValueError("layers, state_size, and conv_kernel must be positive")
        if config.expand <= 0 or config.experts <= 0 or config.modes <= 0:
            raise ValueError("expand, experts, and modes must be positive")
        if config.history_length < 0 or config.future_steps < 0:
            raise ValueError("history_length and future_steps cannot be negative")
        self.config = config

        self.history_slot = _StateSlotEncoder(config.width)
        self.raw_slot = _StateSlotEncoder(config.width)
        self.candidate_slot = _StateSlotEncoder(config.width)
        self.current_fuse = nn.Sequential(
            nn.Linear(config.width * 2, config.width), nn.LayerNorm(config.width)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(3 + 2 * len(TIME_FREQUENCIES), config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.handedness_embedding = nn.Embedding(2, config.width)
        self.joint_embedding = nn.Embedding(JOINT_COUNT, config.width)
        self.parent_embedding = nn.Embedding(JOINT_COUNT, config.width)
        self.finger_embedding = nn.Embedding(6, config.width)
        self.depth_embedding = nn.Embedding(4, config.width)

        parents = torch.tensor((0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14))
        fingers = torch.tensor((0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5))
        depths = torch.tensor((0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3))
        self.register_buffer("joint_parents", parents, persistent=False)
        self.register_buffer("joint_fingers", fingers, persistent=False)
        self.register_buffer("joint_depths", depths, persistent=False)

        self.layers = nn.ModuleList(_TrajectoryLayer(config) for _ in range(config.layers))
        self.delta_head = nn.Linear(config.width, 6)
        self.sigma_head = nn.Linear(config.width, 3)
        self.kappa_head = nn.Linear(config.width, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.sigma_head.bias, math.log(0.1 / 0.9))
        nn.init.constant_(self.kappa_head.bias, math.log(0.4 / 0.6))

        future_width = max(16, min(64, config.width))
        self.mode_embedding = nn.Embedding(config.modes, future_width)
        self.future_context = nn.Linear(config.width, future_width)
        self.future_joint = nn.Linear(config.width, future_width)
        self.future_time = nn.Linear(config.width, future_width)
        self.future_temporal = _MambaResidual(
            future_width, config.state_size, config.conv_kernel, config.expand
        )
        self.future_ff = nn.Sequential(
            nn.LayerNorm(future_width),
            nn.Linear(future_width, future_width * 2),
            nn.GELU(),
            nn.Linear(future_width * 2, future_width),
        )
        self.future_velocity = nn.Linear(future_width, 6)
        self.future_sigma = nn.Linear(future_width, 3)
        self.future_kappa = nn.Linear(future_width, 1)
        self.mode_logits = nn.Linear(config.width, config.modes)
        nn.init.zeros_(self.future_velocity.weight)
        nn.init.zeros_(self.future_velocity.bias)
        nn.init.constant_(self.future_sigma.bias, math.log(0.1 / 0.9))
        nn.init.constant_(self.future_kappa.bias, math.log(0.4 / 0.6))

    def _joint_structure(self, dtype: torch.dtype) -> torch.Tensor:
        indices = torch.arange(JOINT_COUNT, device=self.joint_parents.device)
        return (
            self.joint_embedding(indices)
            + self.parent_embedding(self.joint_parents)
            + self.finger_embedding(self.joint_fingers)
            + self.depth_embedding(self.joint_depths)
        ).to(dtype=dtype)

    def _encode_history(
        self,
        history: torch.Tensor,
        available: torch.Tensor,
        times: torch.Tensor,
        intervals: torch.Tensor,
        handedness: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = history.shape
        if length == 0:
            return history.new_empty(batch, 0, JOINT_COUNT, self.config.width)
        value = self.history_slot(history, available)
        time = self.time_encoder(_time_features(times, intervals))
        hand = self.handedness_embedding(handedness)
        structure = self._joint_structure(value.dtype)
        value = value + time[:, :, None, :] + hand[:, None, None, :] + structure[None, None]
        return value * mask[:, :, None, None].to(dtype=value.dtype)

    def _current_tokens(
        self,
        raw: torch.Tensor,
        candidate: torch.Tensor,
        raw_available: torch.Tensor,
        candidate_available: torch.Tensor,
        dt: torch.Tensor,
        handedness: torch.Tensor,
    ) -> torch.Tensor:
        raw_tokens = self.raw_slot(raw, raw_available)
        candidate_tokens = self.candidate_slot(candidate, candidate_available)
        value = self.current_fuse(torch.cat((raw_tokens, candidate_tokens), dim=-1))
        time = self.time_encoder(_time_features(torch.zeros_like(dt), dt))
        hand = self.handedness_embedding(handedness)
        structure = self._joint_structure(value.dtype)
        return value + time[:, None, :] + hand[:, None, :] + structure[None]

    @staticmethod
    def _bounded_sigma(logits: torch.Tensor) -> torch.Tensor:
        return 0.001 + 0.999 * torch.sigmoid(logits)

    @staticmethod
    def _bounded_kappa(logits: torch.Tensor) -> torch.Tensor:
        low = math.log(0.1)
        span = math.log(10000.0) - low
        return torch.exp(low + span * torch.sigmoid(logits))

    def _future_outputs(
        self,
        current_state: torch.Tensor,
        current_tokens: torch.Tensor,
        future_times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, future_count = future_times.shape
        modes = self.config.modes
        pooled = current_tokens.mean(dim=1)
        logits = self.mode_logits(pooled)

        intervals = torch.cat(
            (future_times[:, :1], future_times[:, 1:] - future_times[:, :-1]), dim=1
        )
        intervals = intervals.clamp_min(0.0)
        query_time = self.future_time(
            self.time_encoder(_time_features(future_times, intervals))
        )
        context = self.future_context(pooled)
        joints = self.future_joint(current_tokens)
        modes_embedding = self.mode_embedding.weight
        future = (
            context[:, None, None, None, :]
            + modes_embedding[None, :, None, None, :]
            + query_time[:, None, :, None, :]
            + joints[:, None, None, :, :]
        )
        future = future.reshape(batch * modes, future_count, JOINT_COUNT, -1)
        future_valid = torch.ones(
            batch * modes,
            future_count,
            dtype=torch.bool,
            device=future_times.device,
        )
        future = self.future_temporal(future, future_valid)
        future = future + self.future_ff(future)
        future = future.reshape(batch, modes, future_count, JOINT_COUNT, -1)

        velocity = self.future_velocity(future)
        position, rotations = unpack_state(current_state)
        position = position[:, None, :].expand(batch, modes, 3)
        rotations = rotations[:, None, :, :, :].expand(batch, modes, JOINT_COUNT, 3, 3)
        state_steps: list[torch.Tensor] = []
        previous_time = future_times.new_zeros(batch)
        for index in range(future_count):
            interval = (future_times[:, index] - previous_time).clamp_min(0.0)
            dt_position = interval[:, None, None]
            dt_rotation = interval[:, None, None, None]
            step_velocity = velocity[:, :, index]
            next_position = position + 0.5 * step_velocity[:, :, 0, :3] * dt_position
            rotation_delta = so3_exp(3.0 * step_velocity[..., 3:6] * dt_rotation)
            next_rotations = rotation_delta @ rotations
            state_steps.append(pack_state(next_position, next_rotations))
            position, rotations = next_position, next_rotations
            previous_time = future_times[:, index]

        state = torch.stack(state_steps, dim=2)
        pooled_future = future.mean(dim=3)
        sigma = self._bounded_sigma(self.future_sigma(pooled_future))
        kappa = self._bounded_kappa(self.future_kappa(future).squeeze(-1))
        return {"state": state, "sigma": sigma, "kappa": kappa, "logits": logits}

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
    ) -> dict[str, Any]:
        if raw.ndim != 2 or raw.shape[-1] != STATE_SIZE:
            raise ValueError("raw must have shape [batch, 99]")
        if candidate.shape != raw.shape:
            raise ValueError("candidate must have the same shape as raw")
        batch = raw.shape[0]
        if history.ndim != 3 or history.shape[0] != batch or history.shape[-1] != STATE_SIZE:
            raise ValueError("history must have shape [batch, time, 99]")
        length = history.shape[1]
        expected_groups = (batch, length, AVAILABILITY_GROUPS)
        if tuple(history_available.shape) != expected_groups:
            raise ValueError("history_available must have shape [batch, time, 17]")
        if tuple(raw_available.shape) != (batch, AVAILABILITY_GROUPS):
            raise ValueError("raw_available must have shape [batch, 17]")
        if tuple(candidate_available.shape) != (batch, AVAILABILITY_GROUPS):
            raise ValueError("candidate_available must have shape [batch, 17]")
        if tuple(history_times.shape) != (batch, length):
            raise ValueError("history_times must have shape [batch, time]")
        if tuple(history_mask.shape) != (batch, length):
            raise ValueError("history_mask must have shape [batch, time]")
        dt = dt.reshape(batch).to(device=raw.device, dtype=raw.dtype)
        handedness = handedness.reshape(batch).to(device=raw.device, dtype=torch.long)
        history_mask = history_mask.to(device=raw.device, dtype=torch.bool)
        history_times = history_times.to(device=raw.device, dtype=raw.dtype)
        history_times = torch.where(
            history_mask, history_times, torch.zeros_like(history_times)
        )
        history = history.to(device=raw.device, dtype=raw.dtype)
        history_available = history_available.to(device=raw.device)
        raw_available = raw_available.to(device=raw.device)
        candidate_available = candidate_available.to(device=raw.device)

        if length > self.config.history_length:
            start = length - self.config.history_length
            history = history[:, start:]
            history_available = history_available[:, start:]
            history_times = history_times[:, start:]
            history_mask = history_mask[:, start:]
            length = self.config.history_length

        if length:
            history_intervals = _history_intervals(history_times, history_mask, dt)
        else:
            history_intervals = history_times.new_empty(batch, 0)
        history_tokens = self._encode_history(
            history,
            history_available,
            history_times,
            history_intervals,
            handedness,
            history_mask,
        )
        current = self._current_tokens(
            raw,
            candidate,
            raw_available,
            candidate_available,
            dt,
            handedness,
        )
        value = torch.cat((history_tokens, current[:, None]), dim=1)
        valid = torch.cat(
            (
                history_mask,
                torch.ones(batch, 1, dtype=torch.bool, device=raw.device),
            ),
            dim=1,
        )
        balance_terms: list[torch.Tensor] = []
        for layer in self.layers:
            value, balance = layer(value, valid)
            balance_terms.append(balance)

        current_tokens = value[:, -1]
        delta_readout = self.delta_head(current_tokens)
        delta = torch.cat(
            (
                0.1 * delta_readout[:, :1, :3],
                0.5 * delta_readout[:, :, 3:6],
            ),
            dim=1,
        ).reshape(batch, 51)
        current_state = apply_delta(candidate, delta)

        pooled = current_tokens.mean(dim=1)
        sigma = self._bounded_sigma(self.sigma_head(pooled))
        kappa = self._bounded_kappa(self.kappa_head(current_tokens).squeeze(-1))
        evidence = raw_available.bool().any(dim=-1) | candidate_available.bool().any(dim=-1)
        if length:
            evidence = evidence | (
                history_available.bool().any(dim=-1) & history_mask
            ).any(dim=-1)
        sigma = torch.where(evidence[:, None], sigma, torch.ones_like(sigma))
        kappa = torch.where(evidence[:, None], kappa, torch.full_like(kappa, 0.1))

        future = None
        if future_times is not None:
            if future_times.ndim != 2 or future_times.shape[0] != batch:
                raise ValueError("future_times must have shape [batch, future_steps]")
            if future_times.shape[1] == 0:
                raise ValueError("future_times cannot be empty when future decoding is requested")
            future_times = future_times.to(device=raw.device, dtype=raw.dtype)
            future = self._future_outputs(current_state, current_tokens, future_times)
            future["sigma"] = torch.where(
                evidence[:, None, None, None],
                future["sigma"],
                torch.ones_like(future["sigma"]),
            )
            future["kappa"] = torch.where(
                evidence[:, None, None, None],
                future["kappa"],
                torch.full_like(future["kappa"], 0.1),
            )

        return {
            "current": {"state": current_state, "sigma": sigma, "kappa": kappa},
            "future": future,
            "balance_loss": torch.stack(balance_terms).mean(),
        }


__all__ = ["TWMConfig", "TrajectoryRestorer"]
