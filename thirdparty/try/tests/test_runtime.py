from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.io import (  # noqa: E402
    _sha256,
    export_hawor_frontend,
    load_slam_c2w,
    load_timestamps,
)
from dual_loop.model import DualLoopConfig, DualLoopObserver  # noqa: E402
from dual_loop.state import axis_angle_to_matrix, pack_state  # noqa: E402
from prepare_training_data import transform_gt_state_to_slam  # noqa: E402
from run_hot3d import load_checkpoint, load_config, load_slam_cameras  # noqa: E402


def _write_frontend_inputs(
    sequence_root: Path,
    frontend_root: Path,
    *,
    gt_boxes: bool = False,
    manifest: bool = True,
    manifest_oracle_assisted: bool | None = None,
) -> tuple[Path, Path]:
    sequence_root.mkdir(parents=True, exist_ok=True)
    preprocess = sequence_root / "preprocess"
    preprocess.mkdir(parents=True, exist_ok=True)
    image_dir = sequence_root / "extracted_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 2
    for frame_id in range(frame_count):
        assert cv2.imwrite(
            str(image_dir / f"{frame_id:04d}.jpg"),
            np.zeros((16, 16, 3), dtype=np.uint8),
        )
    (sequence_root / "focal.txt").write_text("500", encoding="utf-8")
    track_name = "preprocess_gtdet_tracks.npy" if gt_boxes else "preprocess_tracks.npy"
    track_path = preprocess / track_name
    records = [
        {"frame": frame_id, "det": True,
         "det_box": np.asarray([[1, 1, 12, 12, 1]], dtype=np.float32),
         "det_handedness": np.asarray([1])}
        for frame_id in range(frame_count)
    ]
    np.save(track_path, np.asarray({1: records}, dtype=object))

    prediction_dir = frontend_root / "1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = prediction_dir / "0.json"
    identity = np.tile(np.eye(3, dtype=np.float32), (1, frame_count, 1, 1))
    hand_pose = np.tile(np.eye(3, dtype=np.float32), (1, frame_count, 15, 1, 1))
    payload = {
        "init_trans": np.zeros((1, frame_count, 3), dtype=np.float32).tolist(),
        "init_root_orient": identity.tolist(),
        "init_hand_pose": hand_pose.tolist(),
        "init_betas": np.zeros((1, frame_count, 10), dtype=np.float32).tolist(),
    }
    chunk_path.write_text(json.dumps(payload), encoding="utf-8")
    if manifest:
        declared_oracle = gt_boxes if manifest_oracle_assisted is None else manifest_oracle_assisted
        manifest_payload = {
            "schema_version": 1,
            "frontend": "HaWoR",
            "track_file": f"preprocess/{track_name}",
            "track_sha256": _sha256(track_path),
            "oracle_assisted": declared_oracle,
            "chunk_sha256": {"1/0.json": _sha256(chunk_path)},
        }
        (frontend_root / "frontend_provenance.json").write_text(
            json.dumps(manifest_payload), encoding="utf-8")
    return track_path, chunk_path


def test_frontend_requires_verified_manifest_and_exact_hashes(tmp_path: Path) -> None:
    sequence_root = tmp_path / "sequence"
    frontend_root = tmp_path / "frontend"
    _, chunk_path = _write_frontend_inputs(sequence_root, frontend_root)

    arrays, provenance = export_hawor_frontend(
        sequence_root, frontend_root, return_provenance=True)
    assert arrays["valid"].all()
    assert provenance["oracle_assisted"] is False
    assert provenance["oracle_assisted_status"] == "verified_no_oracle"
    assert set(provenance["chunk_sha256"]) == {"1/0.json"}

    chunk_path.write_text(chunk_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the exact track/chunk inputs"):
        export_hawor_frontend(sequence_root, frontend_root, allow_gt_boxes=True)


def test_unknown_and_legacy_gt_frontends_require_opt_in(tmp_path: Path) -> None:
    sequence_root = tmp_path / "unknown_sequence"
    frontend_root = tmp_path / "unknown_frontend"
    _write_frontend_inputs(sequence_root, frontend_root, manifest=False)
    with pytest.raises(ValueError, match="origin is unknown"):
        export_hawor_frontend(sequence_root, frontend_root)
    _, unknown = export_hawor_frontend(
        sequence_root, frontend_root, allow_gt_boxes=True, return_provenance=True)
    assert unknown["oracle_assisted"] is True
    assert unknown["oracle_assisted_status"] == "unknown"

    gt_sequence = tmp_path / "gt_sequence"
    gt_frontend = tmp_path / "gt_frontend"
    _write_frontend_inputs(gt_sequence, gt_frontend, gt_boxes=True)
    with pytest.raises(FileNotFoundError, match="allow_gt_boxes=True"):
        export_hawor_frontend(gt_sequence, gt_frontend)
    _, gt_provenance = export_hawor_frontend(
        gt_sequence, gt_frontend, allow_gt_boxes=True, return_provenance=True)
    assert gt_provenance["bbox_source"] == "legacy_GT_projection_boxes"
    assert gt_provenance["oracle_assisted"] is True


def test_frontend_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    sequence_root = tmp_path / "sequence"
    frontend_root = tmp_path / "frontend"
    track_path, chunk_path = _write_frontend_inputs(sequence_root, frontend_root)
    manifest_path = frontend_root / "frontend_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["track_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the exact track/chunk inputs"):
        export_hawor_frontend(sequence_root, frontend_root, allow_gt_boxes=True)

    manifest["track_sha256"] = _sha256(track_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    chunk_path.write_text(chunk_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the exact track/chunk inputs"):
        export_hawor_frontend(sequence_root, frontend_root, allow_gt_boxes=True)


def test_timestamp_source_is_integer_nanoseconds(tmp_path: Path) -> None:
    source = tmp_path / "sequence"
    source.mkdir()
    timestamps = [53987674924324, 53987708259126]
    with (source / "tt_list.pkl").open("wb") as handle:
        pickle.dump(timestamps, handle)
    timestamp_ns, timestamp_sec = load_timestamps(source, 2)
    assert timestamp_ns.tolist() == timestamps
    assert np.allclose(timestamp_sec, np.asarray(timestamps, dtype=np.float64) / 1e9)

    with (source / "tt_list.pkl").open("wb") as handle:
        pickle.dump([timestamps[0], timestamps[0]], handle)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_timestamps(source, 2)


def test_short_runtime_prefix_loads_full_chunk_name(tmp_path: Path) -> None:
    sequence_root = tmp_path / "sequence"
    slam_dir = sequence_root / "preprocess" / "SLAM"
    slam_dir.mkdir(parents=True)
    trajectory = np.zeros((1000, 7), dtype=np.float32)
    trajectory[:, 6] = 1.0
    np.savez(
        slam_dir / "hawor_slam_w_scale_0_1000.npz",
        traj=trajectory,
        scale=np.asarray(1.0, dtype=np.float64),
    )
    assert not (slam_dir / "hawor_slam_w_scale_0_32.npz").exists()
    camera = load_slam_cameras(sequence_root, 32, 1000, 1000)
    assert camera[0].shape == (32, 3, 3)
    assert camera[1].shape == (32, 3)
    assert camera[0].dtype == camera[1].dtype == torch.float32


def test_gt_to_slam_transform_preserves_real_mano_joints() -> None:
    mano_path = REPOSITORY_ROOT / "_DATA" / "data" / "mano"
    if not mano_path.is_dir():
        pytest.skip("Local MANO assets are unavailable")
    torch.manual_seed(17)
    geometry = ManoGeometry(REPOSITORY_ROOT).eval()
    count = 3
    root = axis_angle_to_matrix(torch.randn(count, 3) * 0.35).unsqueeze(1)
    hand = axis_angle_to_matrix(torch.randn(count, 15, 3) * 0.25)
    source_state = pack_state(
        torch.randn(count, 3), root, hand, torch.randn(count, 10) * 0.2)
    rotation_gt_w2c = axis_angle_to_matrix(
        torch.tensor([[0.2, -0.4, 0.1]]).repeat(count, 1))
    translation_gt_w2c = torch.randn(count, 3)
    rotation_slam_c2w = axis_angle_to_matrix(
        torch.tensor([[-0.3, 0.15, 0.45]]).repeat(count, 1))
    translation_slam_c2w = torch.randn(count, 3)

    with torch.no_grad():
        transformed = transform_gt_state_to_slam(
            geometry,
            source_state,
            rotation_gt_w2c,
            translation_gt_w2c,
            rotation_slam_c2w,
            translation_slam_c2w,
        )
        source_joints = geometry.joints(source_state)
        transformed_joints = geometry.joints(transformed)
        rotation = rotation_slam_c2w @ rotation_gt_w2c
        translation = (
            torch.einsum("tij,tj->ti", rotation_slam_c2w, translation_gt_w2c)
            + translation_slam_c2w)
        expected_joints = (
            torch.einsum("tij,tnj->tni", rotation, source_joints)
            + translation[:, None])
    assert torch.allclose(transformed_joints, expected_joints, atol=5e-4, rtol=0)


def test_prior_only_checkpoint_requires_full_dual_loop(tmp_path: Path) -> None:
    path = tmp_path / "prior.pt"
    torch.save({"stage": "world_model_pretrain", "model": {}}, path)
    with pytest.raises(ValueError, match="world_model_pretrain.*prior-only"):
        load_checkpoint(path, torch.device("cpu"))


def test_checkpoint_config_is_used_before_ablation_overrides(tmp_path: Path) -> None:
    model_config = DualLoopConfig(hidden_dim=64, feature_dim=16, evidence_dim=32)
    checkpoint_path = tmp_path / "full.pt"
    torch.save({
        "stage": "full",
        "config": {"model": model_config.__dict__, "runtime": {"slam_interval": 500}},
        "model": DualLoopObserver(model_config).state_dict(),
    }, checkpoint_path)
    checkpoint, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    args = SimpleNamespace(
        config=PROJECT_ROOT / "configs" / "default.yaml",
        intra_frame_reobserve=True,
        corrected_state_feedback=None,
    )
    config, raw_config, source = load_config(args, checkpoint)
    assert source == "checkpoint"
    assert config.hidden_dim == 64
    assert config.feature_dim == 16
    assert config.intra_frame_reobserve is True
    assert raw_config["runtime"]["slam_interval"] == 500


def test_slam_npz_scale_stays_float32(tmp_path: Path) -> None:
    path = tmp_path / "slam.npz"
    trajectory = np.zeros((1, 7), dtype=np.float32)
    trajectory[0, 0] = 2.0
    trajectory[0, 6] = 1.0
    np.savez(path, traj=trajectory, scale=np.asarray(0.5, dtype=np.float64))
    rotation, translation = load_slam_c2w(path)
    assert rotation.dtype == translation.dtype == torch.float32
    assert torch.allclose(translation, torch.tensor([[1.0, 0.0, 0.0]]))
