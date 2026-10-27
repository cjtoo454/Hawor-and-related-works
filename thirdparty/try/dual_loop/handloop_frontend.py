"""Differentiable adapter to the real HaWoR IAM and pose path.

No cached or synthetic features are accepted here. HaWoR's bidirectional
16-frame context is retained; HandLoop's numerical feedback is causal.
"""
from __future__ import annotations

import torch
from torch import nn

from .state import pack_state as pack_legacy
from .handloop_state import legacy_to_state, state_to_legacy


class HaWoRFrontend(nn.Module):
    def __init__(self, hawor: nn.Module, geometry: nn.Module):
        super().__init__()
        self.model = hawor
        # Geometry belongs to the system, not this adapter's checkpoint.
        object.__setattr__(self, "geometry", geometry)

    def train(self, mode=True):
        super().train(mode)
        # HaWoR's ViT already supports block checkpointing. Keep the genuine
        # trainable backbone while avoiding storage of every block activation.
        if hasattr(self.model.backbone, "use_checkpoint"):
            self.model.backbone.use_checkpoint = bool(mode)
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, frames = batch["img"].shape[:2]
        if frames != 16:
            raise ValueError("HaWoR IAM requires exactly 16 frames per context")
        captured = []
        hook = self.model.backbone.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            output = self.model(batch)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape[1:] != (1280, 16, 12):
            raise ValueError("Expected real HaWoR IAM [B*16,1280,16,12] patches")
        params = output["pred_mano_params"]
        beta = params["betas"].reshape(-1, 10)
        legacy = pack_legacy(output["out"]["trans_full"].reshape(-1, 3),
                             params["global_orient"].reshape(-1, 1, 3, 3),
                             params["hand_pose"].reshape(-1, 15, 3, 3), beta)
        return {
            "state_camera": legacy_to_state(legacy, self.geometry).reshape(batch_size, frames, 99),
            "betas": beta.reshape(batch_size, frames, 10),
            "patches": captured[0].flatten(2).transpose(1, 2).reshape(batch_size, frames, 192, 1280),
            "pred_cam": output["out"]["pred_cam"].reshape(batch_size, frames, 3),
        }


def project_mano16(state_world, betas, geometry, rotation_w2c,
                   translation_w2c, focal, principal, pixel_to_patch):
    """Project actual-wrist state into the *actual* IAM crop, without clamping.

    ``pixel_to_patch`` maps full-image pixels to normalized 192x256 IAM grid
    coordinates, including HaWoR's 32px horizontal crop and any mirror.
    """
    legacy = state_to_legacy(state_world, betas, geometry)
    joints21 = geometry.joints(legacy)
    # ManoGeometry returns OpenPose order; restore native MANO16 ordering.
    joint_map = geometry.mano.joint_map.tolist()
    indices = [joint_map.index(index) for index in range(16)]
    joints = joints21[:, indices]
    camera = torch.einsum("bij,bnj->bni", rotation_w2c, joints) + translation_w2c[:, None]
    depth = camera[..., 2]
    finite_depth = torch.where(depth.abs() > 1e-8, depth, torch.ones_like(depth))
    uv = camera[..., :2] / finite_depth[..., None] * focal[:, None, None] + principal[:, None]
    pixels = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
    grid = torch.einsum("bij,bnj->bni", pixel_to_patch, pixels)[..., :2]
    valid = torch.isfinite(grid).all(-1) & torch.isfinite(depth) & (depth > 1e-5) & (grid.abs() <= 1).all(-1)
    # Invalid queries are masked by the visual reader; keep finite inputs there.
    return torch.nan_to_num(grid), valid


def load_hawor(checkpoint, device="cpu"):
    # Existing loader applies the local licensed-MANO compatibility conversion.
    from pathlib import Path
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_hawor_frontend.py"
    spec = importlib.util.spec_from_file_location("_handloop_hawor_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, _ = module._load_hawor_model(Path(checkpoint), str(device))
    model.requires_grad_(True)
    model.mano.requires_grad_(False)
    return model
