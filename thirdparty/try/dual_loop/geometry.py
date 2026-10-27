"""MANO geometry and camera/world transformations used by both feedback loops."""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from torch import nn
from smplx.utils import Struct

from lib.models.mano_wrapper import MANO

from .state import pack_state, unpack_state


class ManoGeometry(nn.Module):
    def __init__(self, repository_root: Path) -> None:
        super().__init__()
        asset = repository_root / 'thirdparty/try/assets/mano_right.npz'
        model_kwargs = {}
        if asset.exists():
            with np.load(asset, allow_pickle=False) as archive:
                model_kwargs['data_struct'] = Struct(**{key: archive[key] for key in archive.files})
        self.mano = MANO(
            model_path=str(repository_root / "_DATA" / "data" / "mano"),
            gender="neutral",
            num_hand_joints=15,
            create_body_pose=False,
            **model_kwargs,
        )
        self.mano.requires_grad_(False)

    def forward(self, state: torch.Tensor):
        translation, root, hand, betas = unpack_state(state)
        return self.mano(
            global_orient=root,
            hand_pose=hand,
            betas=betas,
            transl=translation,
            pose2rot=False,
        )

    def joints(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).joints

    def camera_to_world_state(
        self,
        camera_state: torch.Tensor,
        rotation_c2w: torch.Tensor,
        translation_c2w: torch.Tensor,
    ) -> torch.Tensor:
        translation, root, hand, betas = unpack_state(camera_state)
        camera_wrist = self.joints(camera_state)[:, 0]
        offset = translation - camera_wrist
        world_root = torch.einsum(
            "bij,bjk->bik", rotation_c2w, root[:, 0]).unsqueeze(1)
        world_wrist = (
            torch.einsum("bij,bj->bi", rotation_c2w, camera_wrist)
            + translation_c2w)
        world_translation = world_wrist + offset
        return pack_state(world_translation, world_root, hand, betas)

    def world_to_camera_parameters(
        self,
        world_state: torch.Tensor,
        rotation_w2c: torch.Tensor,
        translation_w2c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        translation, root, hand, betas = unpack_state(world_state)
        world_wrist = self.joints(world_state)[:, 0]
        offset = translation - world_wrist
        camera_root = torch.einsum(
            "bij,bjk->bik", rotation_w2c, root[:, 0]).unsqueeze(1)
        camera_wrist = (
            torch.einsum("bij,bj->bi", rotation_w2c, world_wrist)
            + translation_w2c)
        camera_translation = camera_wrist + offset
        return camera_translation, camera_root, hand, betas

    def project_world_joints(
        self,
        world_state: torch.Tensor,
        rotation_w2c: torch.Tensor,
        translation_w2c: torch.Tensor,
        focal_length: torch.Tensor,
        image_size_wh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_joints = self.joints(world_state)
        camera_joints = (
            torch.einsum("bij,bnj->bni", rotation_w2c, world_joints)
            + translation_w2c[:, None])
        depth = camera_joints[..., 2]
        principal = image_size_wh[:, None] / 2.0
        uv = camera_joints[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1)
        uv = uv * focal_length[:, None, None] + principal
        denominator = (image_size_wh - 1.0).clamp_min(1.0)[:, None]
        grid = 2.0 * uv / denominator - 1.0
        visible = (
            (depth > 1e-5)
            & (grid[..., 0].abs() <= 1.0)
            & (grid[..., 1].abs() <= 1.0)
        )
        return grid, visible

