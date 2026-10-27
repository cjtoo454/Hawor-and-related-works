"""Focused checks for raw HOT3D trajectory preparation and window sampling."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


TRY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRY_ROOT / "scripts"))

from prepare_raw_trajectories import (  # noqa: E402
    EmptyHandTrajectory,
    MANO_MEMBER,
    DEFAULT_DATA_ROOT,
    DEFAULT_MANO_ASSET,
    load_mano_pca,
    make_states,
    read_right_hand_trajectory,
    sample_nearest_30hz,
)
from train_world_model import (  # noqa: E402
    RESUME_CRITICAL_KEYS,
    SequenceData,
    _finalize_metrics,
    _validate_resume_metadata,
    collect_windows,
    contiguous_segments,
)
from dual_loop.state import STATE_DIM  # noqa: E402


def _sample(timestamp_ns: int, *, x: float = 0.0) -> dict:
    return {
        "timestamp_ns": timestamp_ns,
        "pose_pca": np.zeros(15, dtype=np.float64),
        "q_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "t_xyz": np.asarray([x, 0.0, 0.0]),
        "betas": np.zeros(10, dtype=np.float64),
    }


def _raw_record(timestamp_ns: int, *, include_right: bool = True) -> dict:
    hand = {
        "pose": [0.0] * 15,
        "wrist_xform": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "t_xyz": [0.0, 0.0, 0.0]},
        "betas": [0.0] * 10,
    }
    return {
        "timestamp_ns": timestamp_ns,
        "hand_poses": ({"1": hand} if include_right else {"0": hand}),
    }


def test_raw_zip_reader_keeps_right_hand_and_first_duplicate_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "sample_hand_data.zip"
    rows = [
        _raw_record(20, include_right=False),
        _raw_record(10),
        _raw_record(10),
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(MANO_MEMBER, "\n".join(json.dumps(row) for row in rows) + "\n")

    samples = read_right_hand_trajectory(path, "P0002_example")

    assert [sample["timestamp_ns"] for sample in samples] == [10]
    assert samples[0]["pose_pca"].shape == (15,)
    assert samples[0]["betas"].shape == (10,)


def test_empty_raw_zip_is_an_explicit_skip() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "empty.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(MANO_MEMBER, "")
        with pytest.raises(EmptyHandTrajectory):
            read_right_hand_trajectory(path, "P0004_empty")


def test_nearest_30hz_resampling_marks_missing_tick_and_checks_dt() -> None:
    period = 1_000_000_000 / 30
    samples = [
        _sample(1_000_000_000),
        _sample(1_000_000_000 + round(period)),
        # The third target tick has no source sample within half a frame.
        _sample(1_000_000_000 + round(3 * period)),
    ]

    sampled = sample_nearest_30hz(samples)

    assert sampled["valid"].tolist() == [True, True, False, True]
    assert sampled["timestamps_ns"][2] == -1
    assert np.isnan(sampled["dt_seconds"][2])
    assert sampled["dt_ok"].tolist() == [False, True, False, False]
    assert contiguous_segments(sampled["valid"], sampled["dt_ok"]) == [(0, 2), (3, 4)]


def test_pca_expansion_and_world_state_pack_are_faithful() -> None:
    components = np.eye(45, dtype=np.float32)[:15]
    mean = np.zeros(45, dtype=np.float32)
    samples = [_sample(100, x=0.25)]
    states = make_states(samples, np.asarray([0]), np.asarray([True]), components, mean)

    assert states.shape == (1, STATE_DIM)
    np.testing.assert_allclose(states[0, :3], [0.25, 0.0, 0.0])
    np.testing.assert_allclose(states[0, 3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(states[0, 99:], np.zeros(10))
    assert np.isnan(make_states(
        samples, np.asarray([0]), np.asarray([False]), components, mean)).all()


def _quaternion_to_axis_angle(q_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(q_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    xyz = quaternion[1:]
    sine_half_angle = np.linalg.norm(xyz)
    if sine_half_angle < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sine_half_angle, quaternion[0])
    return (xyz / sine_half_angle * angle).astype(np.float32)


@pytest.mark.skipif(
    not DEFAULT_MANO_ASSET.is_file() or not DEFAULT_DATA_ROOT.joinpath("train.json").is_file(),
    reason="real HOT3D MANO data and PCA asset are not installed",
)
def test_real_nonzero_mano_pca_expansion_matches_pca_mano_joints() -> None:
    import torch
    import smplx
    from smplx.utils import Struct
    from smplx.vertex_ids import vertex_ids

    from dual_loop.geometry import ManoGeometry

    sequence_ids = json.loads((DEFAULT_DATA_ROOT / "train.json").read_text(encoding="utf-8"))
    real_sample = None
    for sequence in sequence_ids:
        candidates = sorted((DEFAULT_DATA_ROOT / sequence).glob("*_hand_data.zip"))
        if not candidates:
            continue
        try:
            candidates_samples = read_right_hand_trajectory(candidates[0], sequence)
        except EmptyHandTrajectory:
            continue
        real_sample = next(
            (sample for sample in candidates_samples
             if np.linalg.norm(sample["pose_pca"]) > 1e-4),
            None,
        )
        if real_sample is not None:
            break
    assert real_sample is not None, "train.json must include a nonzero right-hand PCA record"

    components, mean = load_mano_pca(DEFAULT_MANO_ASSET)
    states = make_states(
        [real_sample], np.asarray([0]), np.asarray([True]), components, mean)
    state = torch.from_numpy(states)
    repository_root = TRY_ROOT.parents[1]
    geometry = ManoGeometry(repository_root).eval()

    with np.load(DEFAULT_MANO_ASSET, allow_pickle=False) as archive:
        model_data = {key: archive[key] for key in archive.files}
    mano = smplx.MANO(
        model_path=str(DEFAULT_MANO_ASSET),
        data_struct=Struct(**model_data),
        is_rhand=True,
        use_pca=True,
        num_pca_comps=15,
        flat_hand_mean=False,
    ).eval()
    reference = mano(
        betas=torch.from_numpy(real_sample["betas"].astype(np.float32))[None],
        global_orient=torch.from_numpy(_quaternion_to_axis_angle(real_sample["q_wxyz"]))[None],
        hand_pose=torch.from_numpy(real_sample["pose_pca"].astype(np.float32))[None],
        transl=torch.from_numpy(real_sample["t_xyz"].astype(np.float32))[None],
    )
    reference_joints = reference.joints
    if reference_joints.shape[1] == 16:
        fingertip_indices = torch.tensor(
            list(vertex_ids["mano"].values()), dtype=torch.long)
        reference_joints = torch.cat(
            (reference_joints, reference.vertices[:, fingertip_indices]), dim=1)
        joint_order = torch.tensor(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12,
             19, 7, 8, 9, 20], dtype=torch.long)
        reference_joints = reference_joints[:, joint_order]

    with torch.no_grad():
        prepared_joints = geometry.joints(state)
    assert prepared_joints.shape == reference_joints.shape == (1, 21, 3)
    torch.testing.assert_close(prepared_joints, reference_joints, rtol=2e-4, atol=2e-5)


def test_training_windows_cannot_cross_invalid_or_bad_dt_frames() -> None:
    valid = np.asarray([True, True, True, True, False, True, True, True, True])
    dt_ok = np.asarray([False, True, True, True, False, False, True, True, True])
    assert contiguous_segments(valid, dt_ok) == [(0, 4), (5, 9)]
    states = np.zeros((len(valid), STATE_DIM), dtype=np.float32)
    sequence = SequenceData(
        "P0002_example", "P0002", states, valid,
        np.arange(len(valid), dtype=np.int64),
        np.full(len(valid), 1 / 30, dtype=np.float32), dt_ok)

    windows = collect_windows([sequence], {sequence.sequence}, history=2, horizon=2)

    assert [(window.sequence_index, window.start) for window in windows] == [(0, 0), (0, 5)]


def test_empty_fast_motion_subset_is_reported_as_null() -> None:
    metrics = _finalize_metrics({
        "translation_error_sum": 0.0,
        "rotation_error_sum": 0.0,
        "beta_squared_error_sum": 0.0,
        "state_nll_sum": 0.0,
        "count": 0.0,
    })

    assert metrics == {
        "sample_count": 0,
        "translation_error_m": None,
        "rotation_error_deg": None,
        "beta_rmse": None,
        "gaussian_nll": None,
    }


def test_resume_rejects_changed_data_or_training_configuration() -> None:
    saved = {key: key for key in RESUME_CRITICAL_KEYS}
    current = dict(saved)
    current["prepared_data_sha256"] = "different-data"

    with pytest.raises(ValueError, match="prepared_data_sha256"):
        _validate_resume_metadata(saved, current, "test checkpoint")
