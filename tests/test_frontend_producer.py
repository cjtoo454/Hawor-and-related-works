from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "thirdparty/try/scripts/prepare_hawor_frontend.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_hawor_frontend", SCRIPT)
producer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(producer)


def test_parts_are_bounded_repeatable_intervals() -> None:
    assert producer.part_ranges(2101, 1000) == [(0, 1000), (1000, 2000), (2000, 2101)]
    assert producer.part_ranges(2101, 1000, [2, 0, 2]) == [(2000, 2101), (0, 1000)]
    with pytest.raises(ValueError, match="outside"):
        producer.part_ranges(2101, 1000, [3])


def test_tracks_are_frame_aligned_and_choose_one_box_per_hand() -> None:
    detections = [
        [
            {"class_id": 1, "confidence": 0.6, "xyxy": [1, 2, 8, 9], "track_id": 4},
            {"class_id": 1, "confidence": 0.9, "xyxy": [3, 4, 10, 11], "track_id": 5},
            {"class_id": 0, "confidence": 0.8, "xyxy": [12, 2, 18, 9], "track_id": 7},
        ],
        [],
    ]
    tracks = producer.build_frame_aligned_tracks(detections)

    assert set(tracks) == {0, 1}
    assert all(len(tracks[side]) == 2 for side in tracks)
    assert tracks[1][0]["det"] is True
    assert tracks[1][0]["det_box"][0].tolist()[:4] == [3.0, 4.0, 10.0, 11.0]
    assert tracks[1][0]["det_box"][0, 4] == pytest.approx(0.9)
    assert tracks[0][0]["det"] is True
    assert tracks[0][1]["det"] is False
    assert tracks[0][1]["frame"] == tracks[1][1]["frame"] == 1


def test_hand_segment_interpolates_only_inside_selected_part() -> None:
    tracks = producer.build_frame_aligned_tracks([
        [{"class_id": 1, "confidence": 0.5, "xyxy": [0, 0, 10, 10]}],
        [],
        [{"class_id": 1, "confidence": 0.7, "xyxy": [2, 4, 14, 16]}],
        [],
        [{"class_id": 1, "confidence": 0.9, "xyxy": [4, 8, 18, 22]}],
    ])

    start, boxes = producer._interpolated_hand_segment(tracks[1], 1, 4)
    assert start == 2
    assert boxes.shape == (1, 5)
    assert boxes[0, :4].tolist() == [2.0, 4.0, 14.0, 16.0]
    assert producer._interpolated_hand_segment(tracks[1], 1, 2) is None


def test_manifest_binds_exact_integer_chunks_and_track_bytes(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence"
    frontend = tmp_path / "frontend"
    track = sequence / producer.TRACK_RELATIVE_PATH
    track.parent.mkdir(parents=True)
    frontend_right = frontend / "1"
    frontend_right.mkdir(parents=True)
    np.save(track, np.asarray({0: [], 1: []}, dtype=object))
    chunk = frontend_right / "12.json"
    original_chunk = json.dumps({"init_trans": [[[]]]})
    chunk.write_text(original_chunk, encoding="utf-8")

    manifest = producer.build_frontend_manifest(sequence, frontend)
    assert manifest["track_sha256"] == producer.sha256_file(track)
    assert manifest["chunk_sha256"] == {
        "1/12.json": producer.sha256_file(chunk),
    }
    (frontend / producer.FRONTEND_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8")
    assert producer.verify_frontend_manifest(sequence, frontend) == manifest

    chunk.write_text(chunk.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_sha256"):
        producer.verify_frontend_manifest(sequence, frontend)

    chunk.write_text(original_chunk, encoding="utf-8")
    track.write_bytes(track.read_bytes() + b" ")
    with pytest.raises(ValueError, match="track_sha256"):
        producer.verify_frontend_manifest(sequence, frontend)


def test_manifest_rejects_non_integer_chunk_names(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence"
    frontend = tmp_path / "frontend"
    (sequence / "preprocess").mkdir(parents=True)
    (frontend / "1").mkdir(parents=True)
    np.save(sequence / producer.TRACK_RELATIVE_PATH, np.asarray({0: [], 1: []}, dtype=object))
    (frontend / "1" / "0_1000.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="integer start frame"):
        producer.build_frontend_manifest(sequence, frontend)


def test_mask_provenance_writer_matches_slam_verifier(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence"
    (sequence / "preprocess").mkdir(parents=True)
    masks = np.asarray([{"size": [2, 2], "counts": b"4"}], dtype=object)
    sources = {
        "detected_tracks": "track-source-hash",
        "hawor_checkpoint": "hawor-source-hash",
    }

    producer._write_global_masks(
        sequence,
        masks,
        sources,
        completed_parts={"masks/0_1.npy": "part-mask-hash"},
    )
    path, provenance = producer._verify_mask_provenance(
        sequence,
        frame_count=1,
        track_hash="track-source-hash",
        hawor_checkpoint_hash="hawor-source-hash",
    )

    assert path == sequence / producer.MASKS_RELATIVE_PATH
    assert provenance["track_sha256"] == "track-source-hash"
    assert provenance["hawor_checkpoint_sha256"] == "hawor-source-hash"
