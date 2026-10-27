"""MANO state encoding shared by training, inference, and serialization."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """First two matrix rows, matching PyTorch3D's Zhou 6D convention."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first, second = value[..., :3], value[..., 3:]
    b1 = F.normalize(first, dim=-1)
    b2 = F.normalize(second - (b1 * second).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def axis_angle_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues formula, including zero-angle inputs."""
    x, y, z = value.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*value.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(value, dim=-1)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    return identity + a * skew + b * (skew @ skew)


NUM_ROTATIONS = 16
ROTATION_DIM = NUM_ROTATIONS * 6
SHAPE_DIM = 10
STATE_DIM = 3 + ROTATION_DIM + SHAPE_DIM
TRANSLATION = slice(0, 3)
ROTATIONS = slice(3, 3 + ROTATION_DIM)
BETAS = slice(3 + ROTATION_DIM, STATE_DIM)


def pack_state(
    translation: torch.Tensor,
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """Pack MANO parameters into [translation, 16x rotation-6D, betas]."""
    rotations = torch.cat((global_orient, hand_pose), dim=-3)
    rotation_6d = matrix_to_rotation_6d(rotations).flatten(start_dim=-2)
    return torch.cat((translation, rotation_6d, betas), dim=-1)


def unpack_state(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return translation, root matrix, 15 hand matrices, and betas."""
    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}, got {state.shape[-1]}")
    translation = state[..., TRANSLATION]
    rotation_6d = state[..., ROTATIONS].reshape(*state.shape[:-1], NUM_ROTATIONS, 6)
    rotations = rotation_6d_to_matrix(rotation_6d)
    betas = state[..., BETAS]
    return translation, rotations[..., :1, :, :], rotations[..., 1:, :, :], betas


def project_state(state: torch.Tensor) -> torch.Tensor:
    """Project perturbed 6D rotations back onto SO(3)."""
    translation, root, hand, betas = unpack_state(state)
    return pack_state(translation, root, hand, betas)


def state_delta_scale(
    translation: float = 0.03,
    rotation: float = 0.15,
    shape: float = 0.03,
) -> torch.Tensor:
    return torch.tensor(
        [translation] * 3 + [rotation] * ROTATION_DIM + [shape] * SHAPE_DIM,
        dtype=torch.float32,
    )

