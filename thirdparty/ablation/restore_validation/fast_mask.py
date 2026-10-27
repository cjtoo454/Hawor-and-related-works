"""Validate a rasterizer-only replacement for HaWoR's silhouette shader.

This module deliberately does not edit or monkeypatch ``lib.vis.renderer``.
It provides the same ``self``-style function so callers can opt into the fast
path after the real-mesh equivalence check passes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, TexturesVertex
from pytorch3d.structures import Meshes

from lib.vis.renderer import Renderer, prep_shared_geometry


ROOT = Path(__file__).resolve().parents[3]
CLOSED_MANO_FACES = np.asarray([
    [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
    [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
    [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
    [120, 108, 78], [78, 108, 79],
], dtype=np.int64)


def fast_render_silhouettes_batch(self, verts, faces, cameras, lights):
    """Return binary foreground masks by rasterizing, without running a shader.

    ``verts``, shared ``faces``, and the batch camera are prepared exactly as
    in ``Renderer.render_silhouettes_batch``. With the legacy raster blur
    radius (1e-5) and SoftPhong sigma (1e-4), every rasterized face has a
    strictly positive alpha and pixels without a face have zero alpha. Thus
    the legacy ``alpha > 0`` mask is the fragment-validity mask.
    """
    batch_size = verts.shape[0]
    colors = torch.ones((batch_size, 4), device=verts.device, dtype=verts.dtype)
    verts, faces, colors = prep_shared_geometry(verts, faces, colors)
    mesh = Meshes(
        verts=list(torch.unbind(verts, dim=0)),
        faces=list(torch.unbind(faces, dim=0)),
        textures=TexturesVertex(verts_features=colors),
    )
    batch_cameras = PerspectiveCameras(
        device=self.device,
        focal_length=cameras.focal_length.expand(batch_size, -1),
        principal_point=cameras.principal_point.expand(batch_size, -1),
        R=cameras.R.expand(batch_size, -1, -1),
        T=cameras.T.expand(batch_size, -1),
        image_size=cameras.image_size.expand(batch_size, -1),
        in_ndc=cameras.in_ndc(),
    )
    fragments = self.renderer.rasterizer(mesh, cameras=batch_cameras)
    return (fragments.pix_to_face >= 0).any(dim=-1).detach().cpu().numpy()


def _first_prediction(json_root: Path, side: int) -> Path:
    side_root = json_root / str(side)
    paths = sorted(
        side_root.glob("*.json"),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else path.stem),
    )
    if not paths:
        raise FileNotFoundError(f"No HaWoR JSON predictions under {side_root}")
    return paths[0]


def _load_two_frame_mesh(json_path: Path, side: int, device: torch.device):
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    with json_path.open("r", encoding="utf-8") as handle:
        prediction: dict[str, Any] = json.load(handle)
    root_matrix = torch.as_tensor(
        prediction["init_root_orient"], dtype=torch.float32, device=device)[:, :2]
    hand_matrix = torch.as_tensor(
        prediction["init_hand_pose"], dtype=torch.float32, device=device)[:, :2]
    translation = torch.as_tensor(
        prediction["init_trans"], dtype=torch.float32, device=device)[:, :2]
    betas = torch.as_tensor(
        prediction["init_betas"], dtype=torch.float32, device=device)[:, :2]
    root_axis_angle = rotation_matrix_to_angle_axis(root_matrix)
    hand_axis_angle = rotation_matrix_to_angle_axis(hand_matrix)

    original_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        with torch.inference_mode():
            mano = run_mano_left if side == 0 else run_mano
            output = mano(
                translation,
                root_axis_angle,
                hand_axis_angle,
                betas=betas,
                use_cuda=device.type == "cuda",
            )
        faces_right = get_mano_faces()
    finally:
        os.chdir(original_cwd)

    closed_faces = np.concatenate((faces_right, CLOSED_MANO_FACES), axis=0)
    if side == 0:
        closed_faces = closed_faces[:, [0, 2, 1]]
    faces = torch.as_tensor(closed_faces, dtype=torch.int64, device=device)
    return output["vertices"][0].to(device), faces


def _timed_render(function, renderer, vertices, faces, cameras, lights, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    mask = function(renderer, vertices, faces, cameras, lights)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return mask, time.perf_counter() - started


def validate_real_masks(
    json_root: Path,
    side: int,
    device_name: str,
    width: int = 1408,
    height: int = 1408,
    focal: float = 609.8258666992188,
) -> dict[str, Any]:
    device = torch.device(device_name)
    json_path = _first_prediction(json_root, side)
    vertices, faces = _load_two_frame_mesh(json_path, side, device)
    renderer = Renderer(
        width,
        height,
        focal,
        device,
        bin_size=128,
        max_faces_per_bin=20000,
    )
    camera_rotation = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    camera_translation = torch.zeros((1, 3), dtype=torch.float32, device=device)
    cameras, lights = renderer.create_camera_from_cv(camera_rotation, camera_translation)

    old_mask, old_seconds = _timed_render(
        lambda obj, verts, tris, cams, lamps: obj.render_silhouettes_batch(
            verts, tris, cams, lamps),
        renderer, vertices, faces, cameras, lights, device)
    fast_mask, fast_seconds = _timed_render(
        fast_render_silhouettes_batch,
        renderer, vertices, faces, cameras, lights, device)
    equal = bool(np.array_equal(old_mask, fast_mask))
    differences = int(np.count_nonzero(old_mask != fast_mask))

    raster_settings = renderer.renderer.rasterizer.raster_settings
    shader_params = renderer.renderer.shader.blend_params
    result = {
        "json": str(json_path),
        "side": side,
        "frames": int(vertices.shape[0]),
        "image_size": [height, width],
        "focal_length_px": focal,
        "closed_face_count": int(faces.shape[0]),
        "blur_radius": float(raster_settings.blur_radius),
        "softphong_sigma": float(shader_params.sigma),
        "bitwise_equal": equal,
        "differing_pixels": differences,
        "legacy_alpha_pixels": np.count_nonzero(old_mask, axis=(1, 2)).tolist(),
        "fast_valid_face_pixels": np.count_nonzero(fast_mask, axis=(1, 2)).tolist(),
        "legacy_seconds": old_seconds,
        "fast_seconds": fast_seconds,
        "speedup": old_seconds / fast_seconds if fast_seconds else None,
        "device": str(device),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-root", type=Path,
                        default=ROOT / "eval_log/P0001_a68492d5")
    parser.add_argument("--side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=1408)
    parser.add_argument("--height", type=int, default=1408)
    parser.add_argument("--focal", type=float, default=609.8258666992188)
    args = parser.parse_args()

    result = validate_real_masks(
        args.json_root, args.side, args.device,
        width=args.width, height=args.height, focal=args.focal)
    print(json.dumps(result, indent=2))
    if not result["bitwise_equal"]:
        raise SystemExit("Fast raster masks differ from legacy SoftPhong alpha masks")


if __name__ == "__main__":
    main()
