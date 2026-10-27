"""Versioned numerical/visual archives; never substitutes labels for inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path, split, kind=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "handloop_v2":
        raise ValueError("Expected handloop_v2 manifest; legacy translation states are incompatible")
    groups = {}
    for name in ("train", "calibration", "test"):
        entries = manifest.get(name, [])
        groups[name] = {(e.get("source", "unknown"), e["participant"]) for e in entries}
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Participant leakage: {left}/{right}")
    entries = []
    for entry in manifest.get(split, []):
        if kind and entry.get("kind", kind) != kind:
            continue
        item = dict(entry)
        archive = Path(item["path"])
        item["path"] = archive if archive.is_absolute() else path.parent / archive
        entries.append(item)
    if not entries:
        raise ValueError(f"No {kind or ''} archives in {split}")
    return entries


def crop_affine(center, scale):
    """Match TrackDatasetEval integer crop, resize pixel centers, and [:,32:-32]."""
    from lib.utils.imutils import transform
    ul = np.asarray(transform([1, 1], center, scale, [256, 256], invert=1)) - 1
    br = np.asarray(transform([257, 257], center, scale, [256, 256], invert=1)) - 1
    extent = br - ul
    if (extent <= 0).any():
        raise ValueError("Invalid crop extent")
    gain = 256.0 / extent
    offset = (0.5 - ul) * gain - 0.5 - np.array([32., 0.])
    norm = 2.0 / np.array([191., 255.])
    affine = np.eye(3, dtype=np.float32)
    affine[0, 0], affine[1, 1] = gain * norm
    affine[:2, 2] = offset * norm - 1
    return affine


class VisualDataset(Dataset):
    """Live RGB clips with independent boxes and supervised numerical labels.

    Frames stay in physical order. Optional warmup frames build feedback history
    before losses start. Split/camera boundaries and gaps are never crossed.
    """
    def __init__(self, manifest, split="train", window=32, warmup=0):
        if window < 1 or warmup < 0 or (window + warmup) % 16:
            raise ValueError("window+warmup must be a positive multiple of HaWoR's 16-frame context")
        self.window, self.warmup = window, warmup
        self.archives, self.index = [], []
        for entry in read_manifest(manifest, split, "visual"):
            with np.load(entry["path"], allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            if str(data.get("schema", "")) != "handloop_v2":
                raise ValueError(f"Unversioned visual archive: {entry['path']}")
            if bool(data.get("oracle_assisted", True)) and split == "train":
                raise ValueError("Oracle-assisted visual training is forbidden")
            if np.any(data["handedness"] != 1):
                raise ValueError("This repository MANO adapter supports right hand only; mirror upstream explicitly")
            total = window + warmup
            good = ((np.diff(data["timestamp_sec"]) > 0) & (np.diff(data["timestamp_sec"]) <= .25)
                    & (np.diff(data["segment_id"]) == 0))
            sid = len(self.archives)
            for start in range(0, len(data["timestamp_sec"]) - total + 1, window):
                if good[start:start + total - 1].all():
                    self.index.append((sid, start))
            self.archives.append(data)
        if not self.index:
            raise ValueError("No continuous visual clips; prepare more data or reduce window/warmup")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from lib.datasets.track_dataset import TrackDatasetEval
        sid, start = self.index[index]
        data = self.archives[sid]
        stop = start + self.window + self.warmup
        db = TrackDatasetEval(data["image_paths"][start:stop].tolist(), data["bbox"][start:stop].copy(),
                              img_focal=float(data["focal_length"][start]),
                              img_center=data["image_size"].astype(np.float32) / 2,
                              normalization=True, dilate=1.2)
        items = [db[i] for i in range(len(db))]
        hawor = {k: torch.stack([item[k] for item in items])
                 for k in ("img", "center", "scale", "img_focal", "img_center")}
        result = {key: torch.from_numpy(data[key][start:stop].copy()) for key in (
            "rotation_w2c", "translation_w2c", "timestamp_sec", "segment_id", "handedness",
            "raw_available", "image_available", "target_state", "target_available", "target_joints", "joint_valid")}
        result["hawor"] = hawor
        result["pixel_to_patch"] = torch.from_numpy(np.stack([
            crop_affine(item["center"].numpy(), float(item["scale"])) for item in items]))
        result["supervise"] = torch.arange(stop - start) >= self.warmup
        return result
