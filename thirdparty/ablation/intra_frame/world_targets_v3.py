"""Label-only canonical raw-world targets for feedback v3.

Camera maps are independent training/calibration exports. They are never
inserted into observer inputs or checkpoints as model features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_world_reference import canonical_raw_world_target_from_cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WorldTargetMap:
    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing 53-sequence camera map: {self.manifest_path}")
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._loaded: dict[str, dict[str, np.ndarray]] = {}
        self._files: dict[str, dict[str, Any]] = {}
        entries = self.manifest.get("sequences")
        if isinstance(entries, dict):
            self._files = entries
        elif isinstance(entries, list):
            self._files = {str(row["sequence"]): row for row in entries}
        if len(self._files) != 53:
            raise ValueError(f"World map must bind 53 train/calibration sequences, found {len(self._files)}")

    def _load(self, sequence: str) -> dict[str, np.ndarray]:
        if sequence in self._loaded:
            return self._loaded[sequence]
        if sequence not in self._files:
            raise ValueError(f"Missing world camera mapping for {sequence}")
        record = self._files[sequence]
        archive_record = record.get("camera_archive", record)
        path_value = archive_record.get("path") or archive_record.get("npz_path") or archive_record.get("camera_map_file")
        expected_hash = archive_record.get("sha256") or archive_record.get("npz_sha256")
        if not path_value or not expected_hash:
            raise ValueError(f"{sequence}: camera map lacks path or SHA256")
        path = Path(path_value)
        if not path.is_absolute():
            root_relative = Path(__file__).resolve().parents[3] / path
            path = root_relative if root_relative.is_file() else self.manifest_path.parent / path
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{sequence}: camera map SHA256 mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "frame_ids", "raw_rotation_w2c", "raw_translation_w2c",
                "gt_rotation_c2w", "gt_translation_c2w", "segment_ids",
                "alignment_segment_ids", "align_rotation_raw_to_gt_by_segment",
                "align_translation_raw_to_gt_by_segment",
            }
            if not required.issubset(archive.files):
                raise ValueError(f"{sequence}: camera map missing {sorted(required - set(archive.files))}")
            data = {key: archive[key].copy() for key in archive.files}
        ids = np.asarray(data["frame_ids"], dtype=np.int64)
        segments = np.asarray(data["segment_ids"], dtype=np.int64)
        if not np.array_equal(ids, np.arange(len(ids))) or not np.array_equal(segments, ids // 1000):
            raise ValueError(f"{sequence}: camera map frame/segment indexing is not official")
        if np.any(~np.isfinite(data["raw_rotation_w2c"])) or np.any(~np.isfinite(data["gt_rotation_c2w"])):
            raise ValueError(f"{sequence}: nonfinite camera rotation")
        self._loaded[sequence] = data
        return data

    def target(self, context: Any) -> torch.Tensor:
        """Build A^-1 H_G for one cached context, preserving its legacy labels."""
        data = self._load(context.sequence)
        frames = np.asarray(context.frame_indices, dtype=np.int64)
        if len(frames) != context.size or np.any(frames < 0) or np.any(frames >= len(data["frame_ids"])):
            raise ValueError(f"{context.item_id}: out-of-range frame IDs")
        if not np.array_equal(data["frame_ids"][frames], frames):
            raise ValueError(f"{context.item_id}: camera map frame IDs mismatch")
        labels = context.labels
        if not np.array_equal(labels["frame_indices"].numpy(), frames):
            raise ValueError(f"{context.item_id}: label frame IDs mismatch")
        segments = data["segment_ids"][frames]
        if not np.array_equal(labels["segment_id"].numpy(), segments) or len(set(segments.tolist())) != 1:
            raise ValueError(f"{context.item_id}: segment mismatch or context spans reset")
        raw_r = data["raw_rotation_w2c"][frames]
        raw_t = data["raw_translation_w2c"][frames]
        input_r = context.inputs["rotation_w2c"].numpy()
        input_t = context.inputs["translation_w2c"].numpy()
        if max(float(np.max(np.abs(input_r - raw_r))), float(np.max(np.abs(input_t - raw_t)))) > 2e-4:
            raise ValueError(f"{context.item_id}: raw camera differs from cached X")
        segment_match = np.flatnonzero(data["alignment_segment_ids"] == segments[0])
        if len(segment_match) != 1:
            raise ValueError(f"{context.item_id}: missing or ambiguous fixed A")
        a = int(segment_match[0])
        target = canonical_raw_world_target_from_cache(
            labels["target_joints"].numpy(), raw_r, raw_t,
            data["gt_rotation_c2w"][frames], data["gt_translation_c2w"][frames],
            data["align_rotation_raw_to_gt_by_segment"][a],
            data["align_translation_raw_to_gt_by_segment"][a],
        )["canonical_raw_world_target"]
        valid = (labels["gt_valid"] & labels["visual_valid"]).numpy()
        if not np.isfinite(target[valid]).all():
            raise ValueError(f"{context.item_id}: nonfinite canonical supervised labels")
        return torch.as_tensor(target, dtype=torch.float32)


DEFAULT_MAP = Path(__file__).resolve().parent / "runs/world_reference_audit_v2/camera_map_train_calibration_53seq.json"
