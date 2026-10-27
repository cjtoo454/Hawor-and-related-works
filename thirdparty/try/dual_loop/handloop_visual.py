"""Candidate-guided visual evidence for the HandLoop observer."""

from __future__ import annotations

import math

import torch
from torch import nn

from .handloop_state import unpack_state


def _rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Encode rotations using the first two matrix rows."""
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def _state_joint_features(state: torch.Tensor) -> torch.Tensor:
    """Return one 9D (root position + rotation) feature per MANO joint."""
    position, rotations = unpack_state(state)
    rotations = _rotation_6d(rotations)
    root_position = torch.cat((position.unsqueeze(-2), rotations[..., :1, :]), dim=-1)
    other_position = rotations[..., 1:, :].new_zeros(
        *rotations.shape[:-2], 15, 3)
    return torch.cat((root_position, torch.cat((other_position, rotations[..., 1:, :]), -1)), -2)


def _time_features(times: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Encode relative history times with seconds and fixed Fourier bands."""
    if times.shape[1] == 0:
        return times.new_zeros((*times.shape, 14), dtype=dtype)
    tau = torch.nan_to_num(
        times.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
    frequencies = tau.new_tensor((0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
    phase = 2.0 * math.pi * tau.unsqueeze(-1) * frequencies
    return torch.cat(
        (tau.unsqueeze(-1), torch.log1p(tau.abs()).unsqueeze(-1),
         phase.sin(), phase.cos()), dim=-1)


class HandLoopVisual(nn.Module):
    """Read history and IAM patches to predict a geometric residual.

    State inputs use the compact 99D hand representation. The predicted 51D
    residual is ``[camera translation, camera root tangent, 15 local finger
    tangents]``. Geometry is intentionally left unbounded after applying its
    physical scale; the zero-initialized head starts with no visual correction.
    """

    def __init__(
        self,
        width: int = 256,
        heads: int = 4,
        feature_dim: int = 1280,
        blocks: int = 2,
        local_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        if feature_dim <= 0 or blocks <= 0 or local_scale <= 0:
            raise ValueError("feature_dim, blocks, and local_scale must be positive")
        self.width = width
        self.heads = heads
        self.feature_dim = feature_dim
        self.local_scale = float(local_scale)

        # Root features carry wrist position and root rotation; finger features
        # carry their parent-local rotations with a zero position slot.
        self.query_projection = nn.Linear(18 + 3, width)
        self.handedness_projection = nn.Linear(1, width, bias=False)
        self.joint_embedding = nn.Embedding(16, width)
        self.history_projection = nn.Linear(9 + 2 + 2 + 1 + 14, width)
        self.patch_projection = nn.Linear(feature_dim, width)
        self.patch_position = nn.Parameter(torch.zeros(1, 192, width))
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        self.null_history = nn.Parameter(torch.zeros(1, 1, width))
        self.null_patch = nn.Parameter(torch.zeros(1, 1, width))

        self.blocks = nn.ModuleList(
            [_HandLoopReadoutBlock(width, heads) for _ in range(blocks)])
        self.output_norm = nn.LayerNorm(width)
        self.root_delta_head = nn.Linear(width, 6)
        self.finger_delta_head = nn.Linear(width, 3)
        nn.init.zeros_(self.root_delta_head.weight)
        nn.init.zeros_(self.root_delta_head.bias)
        nn.init.zeros_(self.finger_delta_head.weight)
        nn.init.zeros_(self.finger_delta_head.bias)
        delta_scale = torch.tensor([0.1] * 3 + [0.5] * 48)
        self.register_buffer("delta_scale", delta_scale)

        rows = torch.arange(16, dtype=torch.float32)
        columns = torch.arange(12, dtype=torch.float32)
        yy, xx = torch.meshgrid(rows, columns, indexing="ij")
        centers = torch.stack(
            (-1.0 + (xx.flatten() + 0.5) * (2.0 / 12.0),
             -1.0 + (yy.flatten() + 0.5) * (2.0 / 16.0)), dim=-1)
        self.register_buffer("patch_centers", centers, persistent=False)

    def _patch_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim == 4:
            if patches.shape[1] != self.feature_dim or patches.shape[2:] != (16, 12):
                raise ValueError(
                    f"patch maps must have shape [B,{self.feature_dim},16,12]")
            patches = patches.flatten(2).transpose(1, 2)
        elif patches.ndim == 3:
            if patches.shape[1:] != (192, self.feature_dim):
                raise ValueError(
                    f"patch tokens must have shape [B,192,{self.feature_dim}]")
        else:
            raise ValueError("patches must be [B,192,C] or [B,C,16,12]")
        patches = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        return self.patch_projection(patches) + self.patch_position.to(patches.dtype)

    def _history_tokens(
        self,
        history_camera: torch.Tensor,
        history_dispersion: torch.Tensor,
        history_available: torch.Tensor,
        history_mask: torch.Tensor,
        history_times: torch.Tensor,
        dispersion_known: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = history_camera.shape[:2]
        if history_camera.shape[-1] != 99:
            raise ValueError("history_camera must have shape [B,T,99]")
        if history_dispersion.shape != (batch, length, 17):
            raise ValueError("history_dispersion must have shape [B,T,17]")
        if history_available.shape != (batch, length, 17):
            raise ValueError("history_available must have shape [B,T,17]")
        if history_mask.shape != (batch, length):
            raise ValueError("history_mask must have shape [B,T]")
        if history_times.shape != (batch, length):
            raise ValueError("history_times must have shape [B,T]")
        if dispersion_known.shape != (batch, length):
            raise ValueError("dispersion_known must have shape [B,T]")

        if length == 0:
            tokens = history_camera.new_zeros(batch, 0, 16, self.width)
            valid = torch.zeros(batch, 0, dtype=torch.bool, device=history_camera.device)
            return tokens, valid

        clean_history = torch.nan_to_num(
            history_camera, nan=0.0, posinf=0.0, neginf=0.0)
        joint_features = _state_joint_features(clean_history)
        joint_index = torch.arange(16, device=history_camera.device)
        # Group 0 is wrist position, group 1 is root rotation, groups 2:17 are
        # finger rotations. The root token receives both root-group flags.
        group_valid = history_available.bool()
        root_valid = torch.stack((group_valid[..., 0], group_valid[..., 1]), -1)
        finger_valid = group_valid[..., 2:]
        finger_available = torch.stack(
            (torch.zeros_like(finger_valid), finger_valid), dim=-1)
        available_features = torch.cat(
            (root_valid[:, :, None, :], finger_available), dim=2)
        # The explicit construction above yields [B,T,16,2]: root carries the
        # wrist and root flags, fingers carry a single rotation flag.
        dispersion = torch.nan_to_num(
            history_dispersion, nan=0.0, posinf=0.0, neginf=0.0)
        position_rms = dispersion[..., :1].clamp_min(0.0)
        log_kappa = dispersion[..., 1:].clamp(0.1, 10000.0).log()
        dispersion = torch.cat((position_rms, log_kappa), dim=-1)
        dispersion = torch.where(
            dispersion_known.bool()[..., None], dispersion, torch.zeros_like(dispersion))
        root_dispersion = torch.stack((dispersion[..., 0], dispersion[..., 1]), -1)
        finger_dispersion = torch.stack(
            (torch.zeros_like(dispersion[..., 2:]), dispersion[..., 2:]), -1)
        dispersion_features = torch.cat(
            (root_dispersion[:, :, None, :], finger_dispersion), dim=2)
        known = torch.nan_to_num(
            dispersion_known.to(dtype=clean_history.dtype),
            nan=0.0, posinf=0.0, neginf=0.0)
        known_features = known[:, :, None, None].expand(batch, length, 16, 1)
        time = _time_features(history_times, clean_history.dtype)
        time_features = time[:, :, None, :].expand(batch, length, 16, time.shape[-1])
        inputs = torch.cat((joint_features, available_features.to(clean_history.dtype),
                            dispersion_features, known_features, time_features), -1)
        tokens = self.history_projection(inputs)
        tokens = tokens + self.joint_embedding(joint_index)[None, None]
        valid = history_mask.bool() & group_valid.any(dim=-1)
        return torch.nan_to_num(tokens), valid

    def forward(
        self,
        patches: torch.Tensor,
        raw_camera: torch.Tensor,
        candidate_camera: torch.Tensor,
        history_camera: torch.Tensor,
        history_dispersion: torch.Tensor,
        history_available: torch.Tensor,
        history_mask: torch.Tensor,
        history_times: torch.Tensor,
        dispersion_known: torch.Tensor,
        projected_grid: torch.Tensor,
        projection_valid: torch.Tensor,
        handedness: torch.Tensor,
        image_available: torch.Tensor,
    ) -> torch.Tensor:
        batch = raw_camera.shape[0]
        if raw_camera.shape != (batch, 99) or candidate_camera.shape != (batch, 99):
            raise ValueError("raw_camera and candidate_camera must have shape [B,99]")
        if projected_grid.shape != (batch, 16, 2):
            raise ValueError("projected_grid must have shape [B,16,2]")
        if projection_valid.shape != (batch, 16):
            raise ValueError("projection_valid must have shape [B,16]")
        if handedness.shape[0] != batch or image_available.shape[0] != batch:
            raise ValueError("handedness and image_available must have batch size B")
        if patches.shape[0] != batch:
            raise ValueError("patch batch size must match state batch size")

        raw_features = _state_joint_features(torch.nan_to_num(
            raw_camera, nan=0.0, posinf=0.0, neginf=0.0))
        candidate_features = _state_joint_features(torch.nan_to_num(
            candidate_camera, nan=0.0, posinf=0.0, neginf=0.0))
        valid_projection = projection_valid.bool() & torch.isfinite(projected_grid).all(-1)
        safe_grid = torch.where(
            valid_projection[..., None], torch.nan_to_num(
                projected_grid, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(projected_grid))
        query_input = torch.cat(
            (raw_features, candidate_features, safe_grid,
             valid_projection.to(raw_features.dtype).unsqueeze(-1)), dim=-1)
        query = self.query_projection(query_input)
        query = query + self.joint_embedding.weight[None]
        handed = torch.nan_to_num(
            handedness.to(query.dtype).reshape(batch, -1)[:, :1],
            nan=0.0, posinf=0.0, neginf=0.0)
        query = query + self.handedness_projection(handed)[:, None]

        history_tokens, history_valid = self._history_tokens(
            history_camera, history_dispersion, history_available, history_mask,
            history_times, dispersion_known)
        # Null keys keep attention defined for empty or fully unavailable
        # history windows. The same learned token serves every joint query.
        history_null = self.null_history.to(query.dtype).expand(batch, 1, self.width)
        patch_tokens = self._patch_tokens(patches)
        patch_null = self.null_patch.to(query.dtype).expand(batch, 1, self.width)
        patch_keys = torch.cat((patch_tokens, patch_null), dim=1)
        image_flag = image_available.bool().reshape(batch)

        # Additive masks have a finite null key for every query, including
        # absent images. This avoids all-masked softmax rows.
        history_bias = query.new_zeros(batch, 16, history_valid.shape[1] + 1)
        if history_valid.shape[1]:
            history_bias[..., :-1] = torch.where(
                history_valid[:, None, :], history_bias[..., :-1],
                torch.full_like(history_bias[..., :-1], -torch.inf))
        patch_valid_bias = query.new_zeros(batch, 16, 193)
        patch_valid_bias[..., :192] = torch.where(
            image_flag[:, None, None], patch_valid_bias[..., :192],
            torch.full_like(patch_valid_bias[..., :192], -torch.inf))

        centers = self.patch_centers.to(device=query.device, dtype=query.dtype)
        squared_distance = (safe_grid[:, :, None, :] - centers[None, None]).square().sum(-1)
        local_patch_bias = (-squared_distance / (2.0 * self.local_scale ** 2)).clamp_min(-80.0)
        local_bias = query.new_full((batch, 16, 193), -4.0)
        local_bias[..., :192] = local_patch_bias
        local_bias = local_bias + patch_valid_bias
        local_bias = local_bias[:, None].expand(batch, self.heads, 16, 193)
        local_bias = local_bias.reshape(batch * self.heads, 16, 193)
        global_bias = patch_valid_bias.clone()
        global_bias = global_bias[:, None].expand(batch, self.heads, 16, 193)
        global_bias = global_bias.reshape(batch * self.heads, 16, 193)
        history_bias = history_bias[:, None].expand(
            batch, self.heads, 16, history_valid.shape[1] + 1)
        history_bias = history_bias.reshape(
            batch * self.heads, 16, history_valid.shape[1] + 1)

        history_keys = torch.cat(
            (history_tokens, history_null[:, :, None, :].expand(
                batch, 1, 16, self.width)), dim=1)
        # [B,T,16,W] -> [B*16,T+1,W], keeping history aligned by joint.
        history_keys = history_keys.permute(0, 2, 1, 3).reshape(
            batch * 16, history_valid.shape[1] + 1, self.width)
        history_bias_by_joint = history_bias.reshape(
            batch, self.heads, 16, history_valid.shape[1] + 1).permute(
                0, 2, 1, 3).reshape(
                    batch * 16 * self.heads, 1, history_valid.shape[1] + 1)
        query = query.reshape(batch * 16, 1, self.width)
        patch_keys_by_batch = patch_keys
        for block in self.blocks:
            query = block(
                query,
                history_keys,
                patch_keys_by_batch,
                history_bias_by_joint,
                global_bias,
                local_bias,
                valid_projection.reshape(batch * 16),
            )
        query = self.output_norm(query.reshape(batch, 16, self.width))
        root_delta = self.root_delta_head(query[:, 0])
        finger_delta = self.finger_delta_head(query[:, 1:]).reshape(batch, 45)
        delta = torch.cat((root_delta, finger_delta), dim=-1)
        delta = delta * self.delta_scale.to(delta.dtype)
        # The visual path can still use history and patch context internally;
        # a missing image simply contributes no geometric residual.
        return torch.where(image_flag[:, None], delta, torch.zeros_like(delta))


class _HandLoopReadoutBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.history_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.image_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.joint_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.history_norm = nn.LayerNorm(width)
        self.global_norm = nn.LayerNorm(width)
        self.local_norm = nn.LayerNorm(width)
        self.joint_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    @property
    def global_attention(self) -> nn.MultiheadAttention:
        """Compatibility view of the shared global/local image attention."""
        return self.image_attention

    @property
    def local_attention(self) -> nn.MultiheadAttention:
        """Compatibility view of the shared global/local image attention."""
        return self.image_attention

    def forward(
        self,
        query: torch.Tensor,
        history_keys: torch.Tensor,
        patch_keys: torch.Tensor,
        history_bias: torch.Tensor,
        global_bias: torch.Tensor,
        local_bias: torch.Tensor,
        local_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = query.shape[0]
        # History is joint-specific, so this path operates on B*16 queries.
        history = self.history_attention(
            query, history_keys, history_keys, attn_mask=history_bias,
            need_weights=False)[0]
        query = self.history_norm(query + history)

        # Patch evidence is shared across the 16 queries in each batch item.
        batch_size = patch_keys.shape[0]
        joints = query.reshape(batch_size, 16, -1)
        global_update = self.image_attention(
            joints, patch_keys, patch_keys, attn_mask=global_bias,
            need_weights=False)[0]
        joints = self.global_norm(joints + global_update)
        local_update = self.image_attention(
            joints, patch_keys, patch_keys, attn_mask=local_bias,
            need_weights=False)[0]
        local_update = local_update * local_valid.reshape(batch_size, 16, 1).to(
            local_update.dtype)
        joints = self.local_norm(joints + local_update)
        self_update = self.joint_attention(
            joints, joints, joints, need_weights=False)[0]
        joints = self.joint_norm(joints + self_update)
        joints = self.ffn_norm(joints + self.ffn(joints))
        return joints.reshape(batch_size * 16, 1, -1)
