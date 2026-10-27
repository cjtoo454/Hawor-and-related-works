"""HandLoop state-space probability and geometric objectives."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F

from .handloop_state import NUM_ROTATIONS, unpack_state


def matrix_fisher_log_normalizer(kappa: torch.Tensor) -> torch.Tensor:
    """Return log Z(kappa) for isotropic matrix Fisher on SO(3).

    For x=2*kappa, Z=e^-x(I0(x)-I1(x))=i0e(x)-i1e(x). Double precision keeps
    this positive difference resolved throughout the specified range to 1e4.
    The output is cast back to the input dtype so the density composes with the
    rest of a model under mixed precision.
    """
    if torch.any(kappa < 0):
        raise ValueError("kappa must be non-negative")
    original_dtype = kappa.dtype
    x = (2 * kappa.to(torch.float64)).clamp_min(0)
    difference = torch.special.i0e(x) - torch.special.i1e(x)
    tiny = torch.finfo(torch.float64).tiny
    result = difference.clamp_min(tiny).log()
    return result.to(original_dtype)


def log_prob(distribution: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Evaluate position-diagonal Gaussian and 16 isotropic MF densities.

    Leading dimensions follow PyTorch broadcasting; the final result retains
    those leading dimensions and sums the 3 positional and 16 rotational terms.
    """
    center = distribution["state"]
    sigma = distribution["sigma"]
    kappa = distribution["kappa"]
    center_p, center_r = unpack_state(center)
    target_p, target_r = unpack_state(target)
    sigma = sigma.clamp_min(torch.finfo(sigma.dtype).tiny)
    kappa = kappa.clamp_min(0)

    standardized = (target_p - center_p) / sigma
    position_log_prob = -0.5 * standardized.square().sum(dim=-1)
    position_log_prob = position_log_prob - sigma.log().sum(dim=-1)
    position_log_prob = position_log_prob - 1.5 * math.log(2 * math.pi)

    relative = center_r.transpose(-1, -2) @ target_r
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    rotation_cost = (3 - trace).clamp_min(0)
    rotation_log_prob = -kappa * rotation_cost - matrix_fisher_log_normalizer(kappa)
    return position_log_prob + rotation_log_prob.sum(dim=-1)


def geometric_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equation (16): wrist SmoothL1 plus root/finger rotation costs."""
    pred_p, pred_r = unpack_state(pred)
    target_p, target_r = unpack_state(target)
    position = F.smooth_l1_loss(
        (pred_p - target_p) / 0.05,
        torch.zeros_like(pred_p),
        beta=0.5,
        reduction="none",
    ).mean(dim=-1)
    relative = pred_r.transpose(-1, -2) @ target_r
    rotation_cost = (3 - relative.diagonal(dim1=-2, dim2=-1).sum(-1)).clamp_min(0)
    return position + 2 * rotation_cost[..., 0] + rotation_cost[..., 1:].mean(dim=-1)


def mixture_log_prob(
    future: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Score complete trajectories with one shared mixture mode per sequence."""
    if target.ndim != 3 or valid.shape != target.shape[:2]:
        raise ValueError("Expected target [B,F,99] and valid [B,F]")
    state = future["state"]
    if state.ndim != 4 or state.shape[0] != target.shape[0] or state.shape[2] != target.shape[1]:
        raise ValueError("Expected future state [B,M,F,99] aligned with target [B,F,99]")
    valid_mask = valid[:, None, :, None].bool()
    target_by_mode = target[:, None, :, :]
    safe_target = torch.where(valid_mask, target_by_mode, state[:, :1].detach())
    step_log_prob = log_prob(future, safe_target)
    step_log_prob = torch.where(valid[:, None, :].bool(), step_log_prob, torch.zeros_like(step_log_prob))
    trajectory_log_prob = step_log_prob.sum(dim=-1)
    log_weights = torch.log_softmax(future["logits"], dim=-1)
    return torch.logsumexp(log_weights + trajectory_log_prob, dim=-1)


def dispersion(distribution: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return [RMS position sigma, 16 rotational concentrations]."""
    sigma = distribution["sigma"]
    kappa = distribution["kappa"]
    if sigma.shape[-1] != 3 or kappa.shape[-1] != NUM_ROTATIONS:
        raise ValueError("Expected sigma [...,3] and kappa [...,16]")
    rms = sigma.square().mean(dim=-1, keepdim=True).sqrt()
    return torch.cat((rms, kappa), dim=-1)
