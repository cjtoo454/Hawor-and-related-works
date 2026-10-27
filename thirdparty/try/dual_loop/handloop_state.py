"""Numerical HandLoop state and geometry bridge.

The dynamic state stores the anatomical wrist position followed by the MANO
root and fifteen parent-relative rotations.  Shape coefficients remain an
external geometry parameter.
"""

from __future__ import annotations

from typing import Any

import torch

from .state import pack_state as _pack_legacy_state
from .state import unpack_state as _unpack_legacy_state


NUM_ROTATIONS = 16
ROTATION_DIM = NUM_ROTATIONS * 6
STATE_DIM = 3 + ROTATION_DIM
TANGENT_DIM = 3 + NUM_ROTATIONS * 3
POSITION = slice(0, 3)
ROTATIONS = slice(3, STATE_DIM)


def _check_state(state: torch.Tensor) -> None:
    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}, got {state.shape[-1]}")


def _matrix_to_rotation_6d(rotations: torch.Tensor) -> torch.Tensor:
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [..., 3, 3], got {rotations.shape}")
    return rotations[..., :2, :].reshape(*rotations.shape[:-2], 6)


def _rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Decode Zhou's first-two-rows representation with degenerate fallbacks."""
    first, second = value[..., :3], value[..., 3:]
    eps = 1e-8

    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback_first = torch.zeros_like(first)
    fallback_first[..., 0] = 1
    b1 = torch.where(first_norm > eps, first / first_norm.clamp_min(eps), fallback_first)

    orthogonal = second - (b1 * second).sum(-1, keepdim=True) * b1
    orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    least_aligned = b1.abs().argmin(dim=-1)
    fallback_second = torch.nn.functional.one_hot(least_aligned, num_classes=3).to(value.dtype)
    fallback_second = fallback_second - (
        fallback_second * b1).sum(-1, keepdim=True) * b1
    fallback_second = fallback_second / torch.linalg.vector_norm(
        fallback_second, dim=-1, keepdim=True).clamp_min(eps)
    b2 = torch.where(
        orthogonal_norm > eps,
        orthogonal / orthogonal_norm.clamp_min(eps),
        fallback_second,
    )
    b3 = torch.nn.functional.normalize(torch.cross(b1, b2, dim=-1), dim=-1, eps=eps)
    return torch.stack((b1, b2, b3), dim=-2)


def pack_state(p: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Pack wrist XYZ and 16 rotation matrices into a 99D state."""
    if p.shape[-1] != 3:
        raise ValueError(f"Expected wrist positions [..., 3], got {p.shape}")
    if rotations.shape[-3:] != (NUM_ROTATIONS, 3, 3):
        raise ValueError(
            f"Expected rotations [..., {NUM_ROTATIONS}, 3, 3], got {rotations.shape}"
        )
    return torch.cat((p, _matrix_to_rotation_6d(rotations).flatten(start_dim=-2)), dim=-1)


def unpack_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return wrist XYZ and 16 proper rotation matrices."""
    _check_state(state)
    rotation_6d = state[..., ROTATIONS].reshape(*state.shape[:-1], NUM_ROTATIONS, 6)
    return state[..., POSITION], _rotation_6d_to_matrix(rotation_6d)


def project_state(state: torch.Tensor) -> torch.Tensor:
    """Project all 6D rotation codes onto SO(3), preserving wrist position."""
    p, rotations = unpack_state(state)
    return pack_state(p, rotations)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    """Exponential map for axis-angle vectors, stable at zero."""
    if vector.shape[-1] != 3:
        raise ValueError(f"Expected tangent vectors [..., 3], got {vector.shape}")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + a * skew + b * (skew @ skew)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Principal SO(3) logarithm with stable small-angle and pi branches."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [..., 3, 3], got {rotation.shape}")

    skew_vector = torch.stack((
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ), dim=-1) * 0.5
    sin_sq = skew_vector.square().sum(dim=-1)
    sin_safe = sin_sq.clamp_min(1e-16).sqrt()
    cos_angle = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    angle = torch.atan2(sin_safe, cos_angle)

    # theta/sin(theta) = 1 + sin(theta)^2/6 + 3 sin(theta)^4/40 + ...
    small_scale = 1 + sin_sq / 6 + 3 * sin_sq.square() / 40 + 5 * sin_sq.pow(3) / 112
    general_scale = angle / sin_safe
    scale = torch.where(sin_sq < 1e-6, small_scale, general_scale)
    result = skew_vector * scale[..., None]

    # The skew part vanishes at pi; recover its axis from the most stable row.
    near_pi = cos_angle < -0.9999
    if near_pi.any():
        diagonal = rotation.diagonal(dim1=-2, dim2=-1)
        eps = 1e-8
        roots = ((diagonal + 1) * 0.5).clamp_min(eps * eps).sqrt()
        dominant = roots.argmax(dim=-1)
        x, y, z = roots.unbind(-1)
        axis_x = torch.stack((
            x,
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4 * x.clamp_min(eps)),
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4 * x.clamp_min(eps)),
        ), dim=-1)
        axis_y = torch.stack((
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4 * y.clamp_min(eps)),
            y,
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4 * y.clamp_min(eps)),
        ), dim=-1)
        axis_z = torch.stack((
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4 * z.clamp_min(eps)),
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4 * z.clamp_min(eps)),
            z,
        ), dim=-1)
        axis = torch.where((dominant == 0)[..., None], axis_x, axis_y)
        axis = torch.where((dominant == 2)[..., None], axis_z, axis)
        axis = torch.nn.functional.normalize(axis, dim=-1, eps=eps)
        axis_sign = torch.where(
            (axis * skew_vector).sum(dim=-1, keepdim=True) < 0,
            -torch.ones_like(axis[..., :1]),
            torch.ones_like(axis[..., :1]),
        )
        axis = axis * axis_sign
        result = torch.where(near_pi[..., None], axis * angle[..., None], result)
    return result


def apply_delta(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply positional and left-composed SO(3) tangent residuals."""
    _check_state(state)
    if delta.shape[-1] != TANGENT_DIM:
        raise ValueError(f"Expected tangent dimension {TANGENT_DIM}, got {delta.shape[-1]}")
    p, rotations = unpack_state(state)
    delta_p = delta[..., :3]
    delta_rot = delta[..., 3:].reshape(*delta.shape[:-1], NUM_ROTATIONS, 3)
    updated_rotations = so3_exp(delta_rot) @ rotations
    return pack_state(p + delta_p, updated_rotations)


def make_anchor(
    raw: torch.Tensor,
    history: torch.Tensor,
    raw_available: torch.Tensor,
    history_available: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the newest available completed wrist pose, else the raw pose.

    Availability groups are ordered as wrist position, root rotation, then the
    15 finger rotations. The returned anchor is always detached.
    """
    _check_state(raw)
    _check_state(history)
    if raw.ndim != 2 or history.ndim != 3 or history.shape[0] != raw.shape[0]:
        raise ValueError("Expected raw [B,99] and history [B,T,99]")
    batch, time = history.shape[:2]
    if raw_available.shape != (batch, 17):
        raise ValueError(f"Expected raw_available [{batch},17], got {raw_available.shape}")
    raw_ok = raw_available.bool()[..., 0] & raw_available.bool()[..., 1]

    history_ok: torch.Tensor
    if history_available is None:
        history_ok = torch.ones((batch, time), dtype=torch.bool, device=history.device)
    else:
        if history_available.shape == (batch, time):
            history_ok = history_available.bool()
        elif history_available.shape == (batch, time, 17):
            flags = history_available.bool()
            history_ok = flags[..., 0] & flags[..., 1]
        else:
            raise ValueError(
                f"Expected history_available [{batch},{time}] or [{batch},{time},17], "
                f"got {history_available.shape}"
            )

    if time:
        slots = torch.arange(time, device=history.device).expand(batch, time)
        latest = torch.where(history_ok, slots, -1).amax(dim=1)
        has_history = latest >= 0
        selected = history[torch.arange(batch, device=history.device), latest.clamp_min(0)]
        history_p, history_rotations = unpack_state(selected)
        anchor_p = torch.where(has_history[:, None], history_p, raw[..., :3])
        raw_p, raw_rotations = unpack_state(raw)
        anchor_r = torch.where(
            has_history[:, None, None], history_rotations[:, 0], raw_rotations[:, 0]
        )
    else:
        has_history = torch.zeros(batch, dtype=torch.bool, device=raw.device)
        raw_p, raw_rotations = unpack_state(raw)
        anchor_p, anchor_r = raw_p, raw_rotations[:, 0]

    eye = torch.eye(3, dtype=raw.dtype, device=raw.device).expand(batch, 3, 3)
    anchor_p = torch.where(raw_ok[:, None] | has_history[:, None], anchor_p, torch.zeros_like(anchor_p))
    anchor_r = torch.where(
        raw_ok[:, None, None] | has_history[:, None, None], anchor_r, eye
    )
    return anchor_r.detach(), anchor_p.detach()


def _anchor_for_state(
    state: torch.Tensor, rotation: torch.Tensor, position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotation.shape[-2:] != (3, 3) or position.shape[-1] != 3:
        raise ValueError("Expected anchor rotation [B,3,3] and position [B,3]")
    if state.ndim < 2 or rotation.shape[0] != state.shape[0] or position.shape[0] != state.shape[0]:
        raise ValueError("State and anchor batch dimensions must match")
    extra = state.ndim - 2
    for _ in range(extra):
        rotation = rotation.unsqueeze(1)
        position = position.unsqueeze(1)
    return rotation, position


def to_local(state: torch.Tensor, rotation: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Transform world states to an anchor frame, preserving finger frames."""
    p, rotations = unpack_state(state)
    anchor_r, anchor_p = _anchor_for_state(state, rotation, position)
    local_p = (anchor_r.transpose(-1, -2) @ (p - anchor_p).unsqueeze(-1)).squeeze(-1)
    local_rotations = rotations.clone()
    local_rotations[..., 0, :, :] = anchor_r.transpose(-1, -2) @ rotations[..., 0, :, :]
    return pack_state(local_p, local_rotations)


def to_world(state: torch.Tensor, rotation: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Transform anchor-frame states to world coordinates."""
    p, rotations = unpack_state(state)
    anchor_r, anchor_p = _anchor_for_state(state, rotation, position)
    world_p = (anchor_r @ p.unsqueeze(-1)).squeeze(-1) + anchor_p
    world_rotations = rotations.clone()
    world_rotations[..., 0, :, :] = anchor_r @ rotations[..., 0, :, :]
    return pack_state(world_p, world_rotations)


def legacy_to_state(state109: torch.Tensor, geometry: Any) -> torch.Tensor:
    """Convert legacy MANO parameters using their decoded anatomical wrist."""
    if state109.shape[-1] != 109:
        raise ValueError(f"Expected legacy state dimension 109, got {state109.shape[-1]}")
    leading = state109.shape[:-1]
    flat = state109.reshape(-1, 109)
    wrist = geometry.joints(flat)[..., 0, :]
    _, root, hand, _ = _unpack_legacy_state(flat)
    rotations = torch.cat((root, hand), dim=-3)
    return pack_state(wrist.reshape(*leading, 3), rotations.reshape(*leading, NUM_ROTATIONS, 3, 3))


def _expand_betas(betas: torch.Tensor, leading: torch.Size) -> torch.Tensor:
    if betas.shape[-1] != 10:
        raise ValueError(f"Expected betas with final dimension 10, got {betas.shape}")
    beta_leading = betas.shape[:-1]
    if not beta_leading:
        shape = (1,) * len(leading) + (10,)
    elif len(beta_leading) <= len(leading) and beta_leading[0] == leading[0]:
        shape = (*beta_leading, *((1,) * (len(leading) - len(beta_leading))), 10)
    else:
        return torch.broadcast_to(betas, (*leading, 10))
    return torch.broadcast_to(betas.reshape(shape), (*leading, 10))


def state_to_legacy(state99: torch.Tensor, betas: torch.Tensor, geometry: Any) -> torch.Tensor:
    """Create legacy MANO parameters whose decoded wrist equals state XYZ.

    MANO's legacy translation is not generally its anatomical wrist location.
    The pose- and shape-dependent wrist offset is evaluated with zero legacy
    translation, then subtracted from the requested wrist position. This bridge
    remains differentiable through state and beta inputs.
    """
    _check_state(state99)
    leading = state99.shape[:-1]
    p, rotations = unpack_state(state99)
    beta_full = _expand_betas(betas, leading)
    flat_p = p.reshape(-1, 3)
    flat_rotations = rotations.reshape(-1, NUM_ROTATIONS, 3, 3)
    flat_betas = beta_full.reshape(-1, 10)
    zeros = torch.zeros_like(flat_p)
    base_legacy = _pack_legacy_state(
        zeros, flat_rotations[:, :1], flat_rotations[:, 1:], flat_betas
    )
    wrist_offset = geometry.joints(base_legacy)[..., 0, :]
    legacy_translation = flat_p - wrist_offset
    legacy = _pack_legacy_state(
        legacy_translation,
        flat_rotations[:, :1],
        flat_rotations[:, 1:],
        flat_betas,
    )
    return legacy.reshape(*leading, 109)
