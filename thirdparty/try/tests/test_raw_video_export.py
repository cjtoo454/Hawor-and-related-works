"""Focused tests for raw HOT3D timestamp and HaWoR export contracts."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


TRY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRY_ROOT / "scripts"))

from export_raw_training import (  # noqa: E402
    EXPORT_SCHEMA,
    ExportError,
    _read_expected_split,
    _resume_sequence,
    _write_root_outputs,
    build_argument_parser,
    map_device_frames_to_timecode,
    make_right_hand_annotations,
    orient_preview_to_native,
    parse_device_timestamps,
    sha256_file,
    standardize_linear_image,
)


def test_mp4_device_timestamps_must_cover_every_decoded_frame() -> None:
    timestamps = parse_device_timestamps("[10, 20, 30]", 3)
    assert timestamps.tolist() == [10, 20, 30]
    with pytest.raises(ExportError, match="does not match video frames"):
        parse_device_timestamps("[10, 20]", 3)
    with pytest.raises(ExportError, match="not monotonic"):
        parse_device_timestamps("[10, 20, 19]", 3)


def test_device_mapping_uses_nearest_rows_and_drops_repeated_timecodes() -> None:
    mapped = map_device_frames_to_timecode(
        device_timestamps_ns=[102, 102, 199, 299],
        mapping_rows=[(10, 100), (11, 101), (20, 199), (30, 299)],
        tolerance_ns=2,
    )
    assert mapped["source_frame_indices"].tolist() == [0, 2, 3]
    assert mapped["device_timestamps_ns"].tolist() == [102, 199, 299]
    assert mapped["timecode_timestamps_ns"].tolist() == [11, 20, 30]
    assert mapped["device_time_mapping_delta_ns"].tolist() == [-1, 0, 0]
    assert mapped["dropped_duplicate_frames"] == [{
        "source_frame_index": 1,
        "device_timestamp_ns": 102,
        "timecode_timestamp_ns": 11,
    }]


def test_device_mapping_rejects_a_timestamp_outside_strict_tolerance() -> None:
    with pytest.raises(ExportError, match="misses nearest mapping by 3 ns"):
        map_device_frames_to_timecode([103], [(10, 100)], tolerance_ns=2)


def test_preview_rotation_is_reversed_before_native_calibration() -> None:
    native = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    preview_clockwise = np.rot90(native, k=-1)
    np.testing.assert_array_equal(
        orient_preview_to_native(preview_clockwise, "clockwise90"), native)
    np.testing.assert_array_equal(orient_preview_to_native(native, "native"), native)


def test_output_image_matches_existing_clockwise_bgr_convention() -> None:
    linear_rgb = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
    expected = np.rot90(linear_rgb[:, :, ::-1], k=-1)
    np.testing.assert_array_equal(standardize_linear_image(linear_rgb), expected)


def test_preview_orientation_must_be_selected_explicitly() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([])


def test_mano_labels_expand_on_the_video_timeline_and_missing_rows_stay_masked() -> None:
    components = np.eye(45, dtype=np.float32)[:15]
    mean = np.linspace(0.0, 0.44, 45, dtype=np.float32)
    pca = np.linspace(-0.5, 0.5, 15, dtype=np.float32)
    sample = {
        "timestamp_ns": 100,
        "pose_pca": pca,
        "q_wxyz": np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        "t_xyz": np.asarray([0.1, -0.2, 0.3]),
        "betas": np.arange(10, dtype=np.float32) / 10,
    }

    annotation, valid = make_right_hand_annotations(
        np.asarray([100, 200], dtype=np.int64),
        {100: sample},
        components,
        mean,
    )

    assert valid.tolist() == [True, False]
    np.testing.assert_allclose(annotation["pose_r"][0], pca @ components + mean)
    np.testing.assert_allclose(annotation["trans_r"][0], sample["t_xyz"])
    np.testing.assert_allclose(annotation["betas_r"][0], sample["betas"])
    np.testing.assert_allclose(annotation["rot_r"][0], [0.0, 0.0, np.pi / 2], atol=1e-6)
    for key in ("rot_r", "trans_r", "pose_r", "betas_r"):
        np.testing.assert_array_equal(annotation[key][1], 0)
    assert annotation["valid_r"].tolist() == [True, False]


def test_split_manifest_accepts_sequence_list_or_entry_count(tmp_path: Path) -> None:
    sequences = ["P0001_a", "P0002_b"]
    manifest = {
        "selected_train_json_entries": sequences,
        "train_sequences": ["P0001_a"],
        "calibration_sequences": ["P0002_b"],
        "sequence_records": [{"sequence": sequence} for sequence in sequences],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, train, calibration = _read_expected_split(path, sequences)
    assert train == ["P0001_a"]
    assert calibration == ["P0002_b"]

    manifest["selected_train_json_entries"] = len(sequences)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _read_expected_split(path, sequences)

    manifest["selected_train_json_entries"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportError, match="different train.json"):
        _read_expected_split(path, sequences)


def test_resume_checks_manifest_source_config_and_output_hashes(tmp_path: Path) -> None:
    sequence_root = tmp_path / "P0001_example"
    sequence_root.mkdir()
    image = sequence_root / "frame.jpg"
    image.write_bytes(b"jpeg bytes")
    source_files = {"video": {"sha256": "source"}}
    config = {"preview_rotation": "clockwise90"}
    manifest = {
        "schema": EXPORT_SCHEMA,
        "status": "exported",
        "source_files": source_files,
        "config": config,
        "frame_count": 1,
        "outputs": {"frame.jpg": {
            "size_bytes": image.stat().st_size,
            "sha256": sha256_file(image),
        }},
    }
    (sequence_root / "raw_export_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")

    assert _resume_sequence(sequence_root, source_files, config) == manifest
    image.write_bytes(b"changed")
    assert _resume_sequence(sequence_root, source_files, config) is None
    assert _resume_sequence(sequence_root, source_files, {"preview_rotation": "native"}) is None


def test_empty_mano_audit_is_kept_out_of_split_without_invalidating_complete_export(
    tmp_path: Path,
) -> None:
    source_sequences = ["P0001_train", "P0002_cal", "P0003_empty"]
    records = [
        {"sequence": "P0001_train", "status": "exported", "frame_count": 100},
        {"sequence": "P0002_cal", "status": "exported", "frame_count": 80},
        {"sequence": "P0003_empty", "status": "skipped",
         "reason": "no right-hand (hand_poses['1']) records"},
    ]
    args = Namespace(max_frames=None, video=None, preview_rotation="clockwise90")

    aggregate = _write_root_outputs(
        tmp_path / "export",
        source_sequences,
        ["P0001_train"],
        ["P0002_cal"],
        records,
        args,
        tmp_path / "split_manifest.json",
    )

    assert aggregate["formal_training_eligible"] is True
    assert json.loads((tmp_path / "export" / "train.json").read_text()) == [
        "P0001_train", "P0002_cal"
    ]
    split = json.loads((tmp_path / "export" / "training_split.json").read_text())
    assert split["train"] == ["P0001_train"]
    assert split["calibration"] == ["P0002_cal"]
    assert split["skipped_sequences"] == [records[-1]]
