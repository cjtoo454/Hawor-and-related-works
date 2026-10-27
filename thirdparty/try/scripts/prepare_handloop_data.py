#!/usr/bin/env python3
"""Convert local HOT3D archives to explicit actual-wrist HandLoop v2 inputs.

External datasets must use the documented 99D schema and their own joint
mapping; this converter never claims to have downloaded/converted them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(ROOT)]

from dual_loop.geometry import ManoGeometry
from dual_loop.handloop_state import legacy_to_state, pack_state
from dual_loop.state import unpack_state as unpack_legacy
from dual_loop.io import export_hawor_frontend


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fill_boxes(boxes, size, segments):
    """Only previous detections may fill crop coordinates, never GT or future."""
    result = boxes.copy()
    previous = np.array([0, 0, size[0], size[1]], dtype=np.float32)
    for i, box in enumerate(result):
        if i and segments[i] != segments[i - 1]:
            previous = np.array([0, 0, size[0], size[1]], dtype=np.float32)
        if np.isfinite(box).all() and (box[2:] > box[:2]).all():
            previous = box
        else:
            result[i] = previous
    return result


def convert_motion(path, output, geometry):
    with np.load(path, allow_pickle=False) as z:
        states = torch.from_numpy(z["states"].copy()).float()
        valid = z["valid"].copy().astype(bool)
        times = z["target_timestamps_ns"].astype(np.float64) / 1e9
    # HOT3D's wrist_xform is passed to MANO transl by its own FK adapter.
    # Decode the actual joint0 instead of trusting the field's name.
    state = np.zeros((len(states), 99), dtype=np.float32)
    ids = np.flatnonzero(valid)
    with torch.no_grad():
        for start in range(0, len(ids), 256):
            selected = ids[start:start + 256]
            state[selected] = legacy_to_state(states[selected], geometry).numpy()
    available = np.repeat(valid[:, None], 17, axis=1)
    gaps = np.r_[False, (np.diff(times) <= 0) | (np.diff(times) > .25)]
    np.savez_compressed(output, schema=np.asarray("handloop_v2"), state=state,
                        available=available, timestamp_sec=times,
                        handedness=np.ones(len(state), np.int64), segment_id=np.cumsum(gaps),
                        source=np.asarray("HOT3D"), position_semantics=np.asarray("anatomical_wrist"),
                        input_sha256=np.asarray(sha256(path)))


def convert_visual(path, output, geometry, video_root, frontend_root):
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    if bool(data["oracle_assisted"]) or str(data["oracle_assisted_status"].item()) != "verified_no_oracle":
        raise ValueError(f"Unverified/oracle training observations: {path}")
    sequence = str(data["sequence"].item())
    frontend, provenance = export_hawor_frontend(video_root / sequence, frontend_root / sequence,
                                                 return_provenance=True)
    if provenance["manifest_status"] != "verified" or provenance["oracle_assisted"]:
        raise ValueError("Fresh verified detector provenance is required")
    n = len(data["timestamp_sec"])
    valid = data["gt_valid"].astype(bool)
    target = np.zeros((n, 99), np.float32)
    joints = np.full((n, 21, 3), np.nan, np.float32)
    ids = np.flatnonzero(valid)
    with torch.no_grad():
        for start in range(0, len(ids), 256):
            chosen = ids[start:start + 256]
            legacy = torch.from_numpy(data["gt_state"][chosen]).float()
            target[chosen] = legacy_to_state(legacy, geometry).numpy()
            joints[chosen] = geometry.joints(legacy).numpy()
    raw_available = np.repeat(frontend["valid"][:, None], 17, axis=1)
    arrays = {key: data[key] for key in ("rotation_w2c", "translation_w2c", "timestamp_sec", "segment_id", "image_paths", "image_size")}
    focal_path = video_root / sequence / "focal.txt"
    arrays.update(schema=np.asarray("handloop_v2"), target_state=target,
                  target_available=np.repeat(valid[:, None], 17, axis=1), target_joints=joints,
                  joint_valid=np.repeat(valid[:, None], 21, axis=1), raw_available=raw_available,
                  image_available=frontend["valid"].astype(bool), handedness=np.ones(n, np.int64),
                  bbox=fill_boxes(frontend["bbox"], data["image_size"], data["segment_id"]),
                  focal_length=np.full(n, float(focal_path.read_text()), np.float32),
                  oracle_assisted=np.asarray(False), input_provenance_json=np.asarray(json.dumps(provenance)),
                  input_sha256=np.asarray(sha256(path)), position_semantics=np.asarray("anatomical_wrist"))
    np.savez_compressed(output, **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("motion", "visual", "both"), default="both")
    p.add_argument("--split_file", type=Path, required=True)
    p.add_argument("--motion_root", type=Path, default=PROJECT / "data/raw_trajectories")
    p.add_argument("--visual_root", type=Path, default=PROJECT / "data/prepared_train100")
    p.add_argument("--video_root", type=Path, default=ROOT / "datasets/hot3d_train100_export")
    p.add_argument("--frontend_root", type=Path, default=ROOT / "eval_log_train100_hawor")
    p.add_argument("--output_root", type=Path, default=PROJECT / "data/handloop_v2")
    args = p.parse_args()
    split = json.loads(args.split_file.read_text())
    heldout = json.loads((ROOT / "datasets/hot3d_valset_export/val.json").read_text())
    participants = lambda seqs: {s.split("_")[0] for s in seqs}
    if participants(split["train"]) & participants(split["calibration"]):
        raise ValueError("Train/calibration participant overlap")
    if participants(split["train"] + split["calibration"]) & participants(heldout):
        raise ValueError("Held-out HOT3D participant leakage")
    args.output_root.mkdir(parents=True, exist_ok=True)
    geometry = ManoGeometry(ROOT).eval()
    for kind in (("motion", "visual") if args.mode == "both" else (args.mode,)):
        directory = args.output_root / kind
        directory.mkdir(exist_ok=True)
        manifest = {"schema": "handloop_v2", "train": [], "calibration": [], "test": [], "skipped": []}
        source_root = args.motion_root if kind == "motion" else args.visual_root
        for group in ("train", "calibration"):
            for sequence in split[group]:
                path = source_root / f"{sequence}.npz"
                if not path.exists():
                    manifest["skipped"].append({"sequence": sequence, "reason": "source archive missing"})
                    continue
                output = directory / path.name
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite prepared archive: {output}")
                if kind == "motion":
                    convert_motion(path, output, geometry)
                else:
                    convert_visual(path, output, geometry, args.video_root, args.frontend_root)
                manifest[group].append({"path": str(output.resolve()), "participant": sequence.split("_")[0],
                                        "source": "HOT3D", "kind": kind, "sha256": sha256(output)})
                print(f"[{kind}/{group}] {sequence}", flush=True)
        if not manifest["train"] or not manifest["calibration"]:
            raise ValueError(f"No usable {kind} train/calibration split")
        (args.output_root / f"{kind}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
