"""Training data, objectives, optimizer schedule, and checkpoints for HandLoop v2."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


HANDLOOP_TRAINING_FORMAT_VERSION = 2
HANDLOOP_MANIFEST_SCHEMA = "handloop_v2"
HANDLOOP_STATE_DIM = 99
HANDLOOP_GROUPS = 17
HANDLOOP_HISTORY = 30
HANDLOOP_MIN_FUTURE = 3
HANDLOOP_MAX_FUTURE = 60
HANDLOOP_MAX_GAP_S = 0.25

MOTION_LOSS_WEIGHTS = {
    "current_nll": 0.05,
    "future_nll": 0.30,
    "best_mode": 0.20,
    "velocity": 0.05,
    "balance": 0.01,
}
CORRUPTION_PROBABILITIES = {
    "clean": 0.20,
    "clean_current_bad_history": 0.20,
    "local_error": 0.35,
    "partial_missing": 0.20,
    "strong_local_failure": 0.05,
}


@dataclass(frozen=True)
class MotionSequence:
    path: Path
    participant: str
    source: str
    states: np.ndarray
    timestamps: np.ndarray
    available: np.ndarray
    handedness: np.ndarray
    segment_id: np.ndarray


@dataclass(frozen=True)
class MotionWindow:
    sequence_index: int
    current_index: int
    future_indices: tuple[int, ...]


def _entry_kind(entry: Mapping[str, Any]) -> str:
    return str(entry.get("kind", "motion")).strip().lower()


def _source_participant(entry: Mapping[str, Any]) -> tuple[str, str]:
    source = str(entry.get("source", "unknown")).strip().casefold()
    participant = str(entry.get("participant", "")).strip().casefold()
    if not participant:
        raise ValueError("Every handloop_v2 manifest entry needs a participant")
    return source, participant


def validate_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    required_kind: str | None = None,
    require_test: bool = False,
) -> tuple[dict[str, Any], Path | None, dict[str, list[dict[str, Any]]]]:
    """Validate split isolation and resolve entry paths against a manifest file."""
    manifest_path: Path | None = None
    if isinstance(manifest_or_path, (str, Path)):
        manifest_path = Path(manifest_or_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"HandLoop manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = dict(manifest_or_path)
    if manifest.get("schema") != HANDLOOP_MANIFEST_SCHEMA:
        raise ValueError(
            f"Expected manifest schema {HANDLOOP_MANIFEST_SCHEMA!r}, "
            f"got {manifest.get('schema')!r}")

    required_splits = ["train", "calibration"]
    if require_test:
        required_splits.append("test")
    for split in required_splits:
        if split not in manifest or not isinstance(manifest[split], list) or not manifest[split]:
            raise ValueError(f"Manifest must contain a non-empty {split!r} split")

    splits: dict[str, list[dict[str, Any]]] = {}
    participant_splits: dict[tuple[str, str], str] = {}
    for split in ("train", "calibration", "test"):
        rows = manifest.get(split, [])
        if not isinstance(rows, list):
            raise ValueError(f"Manifest split {split!r} must be a list")
        normalized: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for original in rows:
            if not isinstance(original, Mapping):
                raise ValueError(f"Entries in {split!r} must be JSON objects")
            entry = dict(original)
            source, participant = _source_participant(entry)
            key = (source, participant)
            prior_split = participant_splits.get(key)
            if prior_split is not None and prior_split != split:
                raise ValueError(
                    f"Participant leakage for source={source!r}, participant={participant!r}: "
                    f"appears in {prior_split!r} and {split!r}")
            participant_splits[key] = split

            entry_path = Path(str(entry.get("path", ""))).expanduser()
            if not str(entry.get("path", "")).strip():
                raise ValueError(f"Manifest entry for {participant!r} has no path")
            if not entry_path.is_absolute() and manifest_path is not None:
                entry_path = manifest_path.parent / entry_path
            entry_path = entry_path.resolve()
            path_key = str(entry_path)
            if path_key in seen_paths:
                raise ValueError(f"Duplicate path in {split!r} split: {path_key}")
            seen_paths.add(path_key)
            entry["path"] = str(entry_path)
            entry["participant"] = str(original["participant"])
            entry["source"] = str(original.get("source", "unknown"))
            if required_kind is None or _entry_kind(entry) == required_kind.casefold():
                normalized.append(entry)
        splits[split] = normalized

    for split in required_splits:
        if not splits[split]:
            kind_label = f" {required_kind}" if required_kind else ""
            raise ValueError(f"Manifest has no{kind_label} entries in {split!r} split")
    return manifest, manifest_path, splits


def validate_cross_manifest_splits(
    *split_maps: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Reject source/participant overlap across motion and visual manifests."""
    assignments: dict[tuple[str, str], str] = {}
    for splits in split_maps:
        for split in ("train", "calibration", "test"):
            for entry in splits.get(split, ()):
                key = _source_participant(entry)
                previous = assignments.get(key)
                if previous is not None and previous != split:
                    raise ValueError(
                        f"Participant leakage across manifests for source={key[0]!r}, "
                        f"participant={key[1]!r}: appears in {previous!r} and {split!r}")
                assignments[key] = split


def _is_identity_rotation(rotation_6d: np.ndarray) -> np.ndarray:
    return np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=rotation_6d.dtype)


def _state_group_finite(state: np.ndarray) -> np.ndarray:
    finite = np.zeros(HANDLOOP_GROUPS, dtype=bool)
    finite[0] = np.isfinite(state[:3]).all()
    for group in range(1, HANDLOOP_GROUPS):
        start = 3 + (group - 1) * 6
        finite[group] = np.isfinite(state[start:start + 6]).all()
    return finite


def _rotation_matrix_from_6d(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64).reshape(2, 3)
    first = rows[0]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = rows[1] - np.dot(first, rows[1]) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(first, fallback)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        second = fallback - np.dot(first, fallback) * first
        second /= max(np.linalg.norm(second), 1e-8)
    else:
        second /= second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=0)


def _axis_angle_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(axis_angle, dtype=np.float64) / angle
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _write_rotation_6d(state: np.ndarray, rotation_index: int, matrix: np.ndarray) -> None:
    start = 3 + rotation_index * 6
    state[start:start + 6] = np.asarray(matrix[:2], dtype=state.dtype).reshape(6)


def _noise_state_group(
    state: np.ndarray,
    group: int,
    rng: np.random.Generator,
    *,
    translation_range: tuple[float, float],
    rotation_range: tuple[float, float],
) -> None:
    if group == 0:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.asarray([1.0, 0.0, 0.0])
        else:
            direction /= norm
        distance = rng.uniform(*translation_range)
        state[:3] += (direction * distance).astype(state.dtype)
        return

    rotation_index = group - 1
    start = 3 + rotation_index * 6
    current = _rotation_matrix_from_6d(state[start:start + 6])
    axis = rng.normal(size=3)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        axis = np.asarray([1.0, 0.0, 0.0])
    else:
        axis /= norm
    angle = rng.uniform(*rotation_range)
    # Left composition is a tangent perturbation in the fixed local anchor.
    _write_rotation_6d(state, rotation_index, _axis_angle_matrix(axis * angle) @ current)


def _fill_missing_history(
    states: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill unavailable groups from past states only, keeping their mask false."""
    values = np.asarray(states, dtype=np.float32).copy()
    masks = np.asarray(available, dtype=bool).copy()
    if values.ndim != 2 or values.shape[-1] != HANDLOOP_STATE_DIM:
        raise ValueError("History states must have shape [time, 99]")
    if masks.shape != (values.shape[0], HANDLOOP_GROUPS):
        raise ValueError("History availability must have shape [time, 17]")

    last_value: list[np.ndarray | None] = [None] * HANDLOOP_GROUPS
    for time_index in range(values.shape[0]):
        finite = _state_group_finite(values[time_index])
        masks[time_index] &= finite
        for group in range(HANDLOOP_GROUPS):
            if masks[time_index, group]:
                last_value[group] = _group_slice(values[time_index], group).copy()
            elif last_value[group] is not None:
                _set_group(values[time_index], group, last_value[group])
            else:
                _set_group(values[time_index], group, _identity_group(group, values.dtype))
    return values, masks


def _group_slice(state: np.ndarray, group: int) -> np.ndarray:
    if group == 0:
        return state[:3]
    start = 3 + (group - 1) * 6
    return state[start:start + 6]


def _set_group(state: np.ndarray, group: int, value: np.ndarray) -> None:
    _group_slice(state, group)[:] = value


def _identity_group(group: int, dtype: np.dtype) -> np.ndarray:
    if group == 0:
        return np.zeros(3, dtype=dtype)
    return _is_identity_rotation(np.empty(6, dtype=dtype))


def _fill_current_from_history(
    current: np.ndarray,
    current_available: np.ndarray,
    history: np.ndarray,
    history_available: np.ndarray,
) -> np.ndarray:
    values = np.asarray(current, dtype=np.float32).copy()
    masks = np.asarray(current_available, dtype=bool)
    for group in range(HANDLOOP_GROUPS):
        if masks[group] and _state_group_finite(values)[group]:
            continue
        found = False
        for index in range(history.shape[0] - 1, -1, -1):
            if history_available[index, group]:
                _set_group(values, group, _group_slice(history[index], group))
                found = True
                break
        if not found:
            _set_group(values, group, _identity_group(group, values.dtype))
    return values


def _encode_handedness(value: Any) -> float:
    text = str(value).strip().casefold()
    if text in {"left", "l", "-1", "0"}:
        return 0.0
    if text in {"right", "r", "1"}:
        return 1.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if numeric <= 0 else 1.0


def _load_motion_sequence(entry: Mapping[str, Any]) -> MotionSequence:
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Motion sequence not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"state", "timestamp_sec", "available", "handedness", "segment_id"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing motion arrays: {missing}")
        states = np.asarray(archive["state"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamp_sec"], dtype=np.float64)
        available = np.asarray(archive["available"], dtype=bool)
        handedness = np.asarray(archive["handedness"])
        segment_id = np.asarray(archive["segment_id"])

    count = states.shape[0] if states.ndim else 0
    if states.shape != (count, HANDLOOP_STATE_DIM):
        raise ValueError(f"{path}: state must have shape [N, 99], got {states.shape}")
    if timestamps.shape != (count,) or available.shape != (count, HANDLOOP_GROUPS):
        raise ValueError(f"{path}: timestamp or availability shape is inconsistent with state")
    if handedness.shape != (count,) or segment_id.shape != (count,):
        raise ValueError(f"{path}: handedness and segment_id must have shape [N]")
    if count and (not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0)):
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing")

    for index in range(count):
        available[index] &= _state_group_finite(states[index])
    return MotionSequence(
        path=path,
        participant=str(entry["participant"]),
        source=str(entry.get("source", "unknown")),
        states=states,
        timestamps=timestamps,
        available=available,
        handedness=handedness,
        segment_id=segment_id,
    )


def _same_segment(sequence: MotionSequence, left: int, right: int) -> bool:
    return bool(sequence.segment_id[left] == sequence.segment_id[right])


def _time_contiguous(sequence: MotionSequence, left: int, right: int, max_gap_s: float) -> bool:
    return (
        _same_segment(sequence, left, right)
        and 0.0 < sequence.timestamps[right] - sequence.timestamps[left] <= max_gap_s
    )


class HandLoopMotionDataset(Dataset[dict[str, torch.Tensor]]):
    """30-frame past windows with current/future labels and online corruption."""

    def __init__(
        self,
        manifest: Mapping[str, Any] | str | Path,
        split: str = "train",
        *,
        history_length: int = HANDLOOP_HISTORY,
        min_future: int = HANDLOOP_MIN_FUTURE,
        max_future: int = HANDLOOP_MAX_FUTURE,
        max_gap_s: float = HANDLOOP_MAX_GAP_S,
        seed: int = 7,
        corruption_probabilities: Mapping[str, float] | None = None,
    ) -> None:
        if history_length != HANDLOOP_HISTORY:
            raise ValueError(f"HandLoop v2 uses a fixed {HANDLOOP_HISTORY}-frame history")
        if min_future < 1 or max_future < min_future or max_future > HANDLOOP_MAX_FUTURE:
            raise ValueError("future horizon must be 1..60 and max_future >= min_future")
        if max_gap_s <= 0.0:
            raise ValueError("max_gap_s must be positive")
        _, self.manifest_path, splits = validate_manifest(manifest, required_kind="motion")
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}; expected train/calibration/test")
        entries = splits[split]
        if not entries:
            raise ValueError(f"No motion entries in {split!r} split")
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.history_length = history_length
        self.min_future = min_future
        self.max_future = max_future
        self.max_gap_s = float(max_gap_s)
        self.corruption_probabilities = dict(
            CORRUPTION_PROBABILITIES if corruption_probabilities is None
            else corruption_probabilities)
        if set(self.corruption_probabilities) != set(CORRUPTION_PROBABILITIES):
            raise ValueError("corruption_probabilities must define all five HandLoop categories")
        if any(value < 0.0 for value in self.corruption_probabilities.values()):
            raise ValueError("corruption probabilities must be non-negative")
        total_probability = sum(self.corruption_probabilities.values())
        if not np.isclose(total_probability, 1.0):
            raise ValueError("the five corruption probabilities must sum to one")

        self.sequences = [_load_motion_sequence(entry) for entry in entries]
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"No valid {split!r} windows: each needs 30 contiguous history frames, "
                f"a complete current target, and at least {min_future} future labels")
        self.source_indices: dict[str, list[int]] = {}
        for window_index, window in enumerate(self.windows):
            source = self.sequences[window.sequence_index].source.casefold()
            self.source_indices.setdefault(source, []).append(window_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_windows(self) -> list[MotionWindow]:
        windows: list[MotionWindow] = []
        for sequence_index, sequence in enumerate(self.sequences):
            count = len(sequence.timestamps)
            complete = sequence.available.all(axis=-1)
            for current_index in range(self.history_length, count):
                history_start = current_index - self.history_length
                if not complete[current_index]:
                    continue
                contiguous = all(
                    _time_contiguous(sequence, index - 1, index, self.max_gap_s)
                    for index in range(history_start + 1, current_index + 1))
                if not contiguous:
                    continue
                future_indices: list[int] = []
                for index in range(current_index + 1, min(count, current_index + 1 + self.max_future)):
                    if not _time_contiguous(sequence, index - 1, index, self.max_gap_s):
                        break
                    future_indices.append(index)
                if sum(bool(complete[index]) for index in future_indices) < self.min_future:
                    continue
                # Trailing unavailable labels are padding; retain intermediate query timestamps.
                last_valid = max(i for i, index in enumerate(future_indices) if complete[index])
                future_indices = future_indices[:last_valid + 1]
                windows.append(MotionWindow(sequence_index, current_index, tuple(future_indices)))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _corrupt(
        self,
        history: np.ndarray,
        history_available: np.ndarray,
        current: np.ndarray,
        current_available: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        probabilities = np.asarray(list(self.corruption_probabilities.values()), dtype=np.float64)
        names = list(self.corruption_probabilities)
        category = str(rng.choice(names, p=probabilities / probabilities.sum()))
        history = history.copy()
        history_available = history_available.copy()
        current = current.copy()
        current_available = current_available.copy()

        def corrupt_history(magnitude: str) -> None:
            for time_index in range(history.shape[0]):
                for group in range(HANDLOOP_GROUPS):
                    if not history_available[time_index, group] or rng.random() >= 0.08:
                        continue
                    if magnitude == "strong":
                        translation_range, rotation_range = (0.03, 0.18), (0.15, 0.8)
                    else:
                        translation_range, rotation_range = (0.005, 0.04), (0.02, 0.2)
                    _noise_state_group(
                        history[time_index], group, rng,
                        translation_range=translation_range, rotation_range=rotation_range)

        if category == "clean_current_bad_history":
            corrupt_history("ordinary")
        elif category == "local_error":
            corrupt_history("ordinary")
            active = np.flatnonzero(current_available)
            if active.size:
                group_count = int(rng.integers(1, min(5, active.size) + 1))
                for group in rng.choice(active, size=group_count, replace=False):
                    _noise_state_group(
                        current, int(group), rng,
                        translation_range=(0.005, 0.06), rotation_range=(0.02, 0.35))
        elif category == "partial_missing":
            corrupt_history("ordinary")
            present = np.flatnonzero(current_available)
            if present.size:
                missing_count = int(rng.integers(1, min(9, present.size) + 1))
                current_available[rng.choice(present, size=missing_count, replace=False)] = False
            # Missing historical groups stay missing; the model receives the explicit mask.
            for time_index in range(history.shape[0]):
                present = np.flatnonzero(history_available[time_index])
                if present.size and rng.random() < 0.2:
                    remove_count = max(1, int(rng.integers(1, min(5, present.size) + 1)))
                    history_available[time_index, rng.choice(present, size=remove_count, replace=False)] = False
            history, history_available = _fill_missing_history(history, history_available)
            current = _fill_current_from_history(
                current, current_available, history, history_available)
        elif category == "strong_local_failure":
            corrupt_history("ordinary")
            active = np.flatnonzero(current_available)
            if active.size:
                group = int(rng.choice(active))
                _noise_state_group(
                    current, group, rng,
                    translation_range=(0.12, 0.45), rotation_range=(0.7, 2.4))
                # A small chance of a neighboring finger-group failure adds local structure.
                if group > 1 and rng.random() < 0.35:
                    neighbor = max(1, group - 1)
                    if current_available[neighbor]:
                        _noise_state_group(
                            current, neighbor, rng,
                            translation_range=(0.12, 0.45), rotation_range=(0.7, 2.4))
        # Corruption type and exact error parameters are intentionally not returned.
        return history, history_available, current, current_available, category

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        sequence = self.sequences[window.sequence_index]
        current_index = window.current_index
        history_start = current_index - self.history_length
        history, history_available = _fill_missing_history(
            sequence.states[history_start:current_index],
            sequence.available[history_start:current_index],
        )
        current_target = sequence.states[current_index].copy()
        current_available = sequence.available[current_index].copy()
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        rng = np.random.default_rng(seed_sequence)
        history, history_available, current_input, current_available, category = self._corrupt(
            history, history_available, current_target, current_available, rng)
        current_input = _fill_current_from_history(
            current_input, current_available, history, history_available)

        future_indices = list(window.future_indices)
        future_states = np.zeros((self.max_future, HANDLOOP_STATE_DIM), dtype=np.float32)
        future_valid = np.zeros(self.max_future, dtype=bool)
        real_future_times = sequence.timestamps[future_indices] - sequence.timestamps[current_index]
        future_states[:len(future_indices)] = sequence.states[future_indices]
        future_valid[:len(future_indices)] = sequence.available[future_indices].all(axis=-1)
        if len(real_future_times):
            future_times = np.empty(self.max_future, dtype=np.float32)
            future_times[:len(real_future_times)] = real_future_times.astype(np.float32)
            last_step = (
                float(real_future_times[-1] - real_future_times[-2])
                if len(real_future_times) > 1
                else float(sequence.timestamps[current_index] - sequence.timestamps[current_index - 1])
            )
            for offset in range(len(real_future_times), self.max_future):
                future_times[offset] = future_times[offset - 1] + max(last_step, 1e-3)
        else:
            raise RuntimeError("A motion window unexpectedly has no future query")

        history_times = (
            sequence.timestamps[history_start:current_index] - sequence.timestamps[current_index]
        ).astype(np.float32)
        dt = float(sequence.timestamps[current_index] - sequence.timestamps[current_index - 1])
        handedness = _encode_handedness(sequence.handedness[current_index])
        return {
            "history": torch.from_numpy(history),
            "history_available": torch.from_numpy(history_available),
            "history_times": torch.from_numpy(history_times),
            "history_mask": torch.ones(self.history_length, dtype=torch.bool),
            "raw": torch.from_numpy(current_input.copy()),
            "candidate": torch.from_numpy(current_input.copy()),
            "raw_available": torch.from_numpy(current_available.copy()),
            "candidate_available": torch.from_numpy(current_available.copy()),
            "dt": torch.tensor(dt, dtype=torch.float32),
            "handedness": torch.tensor(handedness, dtype=torch.float32),
            "current": torch.from_numpy(current_target.copy()),
            "current_valid": torch.tensor(True),
            "future": torch.from_numpy(future_states),
            "future_valid": torch.from_numpy(future_valid),
            "future_times": torch.from_numpy(future_times),
            # This audit label is not consumed by TrajectoryRestorer.
            "corruption_category": torch.tensor(
                list(CORRUPTION_PROBABILITIES).index(category), dtype=torch.int64),
        }


class HandLoopBatchStream:
    """Cycling DataLoader with a checkpointable epoch permutation and cursor."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        seed: int,
        num_workers: int = 0,
        shuffle: bool = True,
        sampling_mode: str = "full_coverage",
    ) -> None:
        if batch_size < 1 or len(dataset) < 1:
            raise ValueError("Batch streams require a non-empty dataset and positive batch size")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if sampling_mode not in {"full_coverage", "source_balanced"}:
            raise ValueError("sampling_mode must be 'full_coverage' or 'source_balanced'")
        if sampling_mode == "source_balanced" and not getattr(dataset, "source_indices", None):
            raise ValueError("source_balanced sampling requires HandLoopMotionDataset sources")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.sampling_mode = sampling_mode
        self.epoch = 0
        self.offset = 0
        self.order = self._make_order()
        self._loader: DataLoader | None = None
        self._iterator: Any = None
        self._rebuild_iterator()

    def _make_order(self) -> list[int]:
        count = len(self.dataset)
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        if self.sampling_mode == "source_balanced":
            sources = sorted(self.dataset.source_indices)
            source_choices = torch.randint(len(sources), (count,), generator=generator).tolist()
            return [
                self.dataset.source_indices[source][int(torch.randint(
                    len(self.dataset.source_indices[source]), (1,), generator=generator).item())]
                for source in (sources[source_index] for source_index in source_choices)
            ]
        if not self.shuffle:
            return list(range(count))
        return torch.randperm(count, generator=generator).tolist()

    def _batch_indices(self) -> list[list[int]]:
        return [self.order[start:start + self.batch_size]
                for start in range(0, len(self.order), self.batch_size)]

    def _rebuild_iterator(self) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + 17)
        self._loader = DataLoader(
            self.dataset,
            batch_sampler=self._batch_indices(),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            generator=generator,
        )
        self._iterator = iter(self._loader)
        for _ in range(self.offset):
            next(self._iterator)

    def __iter__(self) -> "HandLoopBatchStream":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.offset >= len(self._batch_indices()):
            self.epoch += 1
            self.offset = 0
            self.order = self._make_order()
            self._rebuild_iterator()
        assert self._iterator is not None
        batch = next(self._iterator)
        self.offset += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset_length": len(self.dataset),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "sampling_mode": self.sampling_mode,
            "epoch": self.epoch,
            "offset": self.offset,
            "order": list(self.order),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "dataset_length": len(self.dataset),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "sampling_mode": self.sampling_mode,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Batch stream checkpoint mismatch for {key}")
        order = [int(index) for index in state["order"]]
        valid_order = len(order) == len(self.dataset) and all(
            0 <= index < len(self.dataset) for index in order)
        if self.sampling_mode == "full_coverage":
            valid_order = valid_order and sorted(order) == list(range(len(self.dataset)))
        if not valid_order:
            raise ValueError("Batch stream checkpoint has an invalid sample permutation")
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        if not 0 <= self.offset <= math.ceil(len(order) / self.batch_size):
            raise ValueError("Batch stream checkpoint has an invalid batch cursor")
        self.order = order
        self._rebuild_iterator()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise KeyError(f"Expected {name!r} in distribution mapping")
        return value[name]
    if not hasattr(value, name):
        raise AttributeError(f"Expected distribution attribute {name!r}")
    return getattr(value, name)


def _first_present(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise KeyError(f"Expected one of fields {tuple(names)}")


def _motion_probability_api():
    from .handloop_probability import geometric_loss, log_prob, mixture_log_prob
    return log_prob, geometric_loss, mixture_log_prob


def _safe_state_targets(
    targets: torch.Tensor,
    valid: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    expanded = valid
    while expanded.ndim < targets.ndim:
        expanded = expanded.unsqueeze(-1)
    while fallback.ndim < targets.ndim:
        fallback = fallback.unsqueeze(1)
    return torch.where(expanded, targets, fallback.expand_as(targets))


def _mode_log_likelihoods(
    future_distribution: Any,
    safe_target: torch.Tensor,
    valid: torch.Tensor,
    log_prob: Callable[[Any, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return each whole-trajectory mode's unnormalized log likelihood."""
    state = _first_present(future_distribution, ("state", "center", "mean"))
    if state.ndim != 4:
        raise ValueError("future state centers must have shape [batch, modes, time, 99]")
    logits = _first_present(future_distribution, ("logits", "mode_logits"))
    if logits.ndim != 2 or logits.shape[:2] != state.shape[:2]:
        raise ValueError("future logits must have shape [batch, modes]")

    # handloop_probability.log_prob broadcasts over the mode axis and leaves time
    # un-reduced. Keeping this per-query result is required for the detached
    # whole-trajectory posterior in the velocity objective.
    per_step = log_prob(future_distribution, safe_target[:, None, :, :])
    if per_step.shape == state.shape[:2]:
        # A reduced output cannot be masked without risking padded labels. The
        # probability module can opt into the standard unreduced contract by
        # returning [B,M,F].
        if not bool(valid.all()):
            raise ValueError("future log_prob must preserve the time axis for masked targets")
        return per_step
    if per_step.shape != state.shape[:3]:
        raise ValueError(
            "future log_prob must return [batch,modes,time] so invalid labels can be masked")
    return (per_step * valid[:, None, :].to(per_step.dtype)).sum(dim=-1)


def _rotation_state(state: torch.Tensor) -> torch.Tensor:
    from .handloop_state import unpack_state
    _, rotations = unpack_state(state)
    return rotations


def _velocity_errors(
    current_center: torch.Tensor,
    future_centers: torch.Tensor,
    target_current: torch.Tensor,
    target_future: torch.Tensor,
    future_times: torch.Tensor,
    future_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-mode normalized velocity error and the supervised query mask."""
    from .handloop_state import so3_log, unpack_state

    # Shapes: current [B,99], future [B,M,F,99].
    pred_p, pred_r0 = unpack_state(current_center)
    del pred_r0
    true_p, true_r = unpack_state(target_current)
    del true_r
    pred_future_p, pred_future_r = unpack_state(future_centers)
    true_future_p, true_future_r = unpack_state(target_future)
    prev_pred_p = torch.cat((
        pred_p[:, None, None, :].expand(-1, pred_future_p.shape[1], 1, -1),
        pred_future_p[:, :, :-1],
    ), dim=2)
    prev_true_p = torch.cat((true_p[:, None, :], true_future_p[:, :-1]), dim=1)
    # Build prior rotations at t-1 for each query, including the current target.
    pred_current_rot = _rotation_state(current_center)
    target_current_rot = _rotation_state(target_current)
    prev_pred_rot = torch.cat((
        pred_current_rot[:, None, None].expand(
            -1, future_centers.shape[1], 1, -1, -1, -1),
        pred_future_r[:, :, :-1],
    ), dim=2)
    prev_true_rot = torch.cat((target_current_rot[:, None], true_future_r[:, :-1]), dim=1)

    query_times = future_times
    deltas = torch.cat((query_times[:, :1], query_times[:, 1:] - query_times[:, :-1]), dim=1)
    deltas = deltas.clamp_min(1e-4)
    pred_position_velocity = (pred_future_p - prev_pred_p) / deltas[:, None, :, None]
    true_position_velocity = (true_future_p - prev_true_p) / deltas[:, :, None]

    pred_relative_rotation = pred_future_r @ prev_pred_rot.transpose(-1, -2)
    true_relative_rotation = true_future_r @ prev_true_rot.transpose(-1, -2)
    pred_rotation_velocity = so3_log(pred_relative_rotation) / deltas[:, None, :, None, None]
    true_rotation_velocity = so3_log(true_relative_rotation) / deltas[:, :, None, None]

    pred_velocity = torch.cat(
        (pred_position_velocity / 0.5,
         pred_rotation_velocity.flatten(start_dim=-2) / 4.0),
        dim=-1,
    )
    true_velocity = torch.cat(
        (true_position_velocity / 0.5,
         true_rotation_velocity.flatten(start_dim=-2) / 4.0),
        dim=-1,
    )
    elementwise = torch.nn.functional.smooth_l1_loss(
        pred_velocity, true_velocity[:, None].expand_as(pred_velocity),
        beta=0.5, reduction="none")
    # 3 translation plus 48 rotation tangent components, averaged per query.
    per_query = elementwise.mean(dim=-1)
    velocity_valid = future_valid.clone()
    if velocity_valid.shape[1] > 1:
        velocity_valid[:, 1:] &= future_valid[:, :-1]
    weight = velocity_valid[:, None, :].to(per_query.dtype)
    return (per_query * weight).sum(-1) / weight.sum(-1).clamp_min(1.0), velocity_valid


def motion_losses(
    output: Any,
    targets: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute equations (16)–(19), including whole-trajectory mode semantics."""
    log_prob, geometric_loss, mixture_log_prob = _motion_probability_api()
    weights = dict(MOTION_LOSS_WEIGHTS if weights is None else weights)
    current_distribution = _field(output, "current")
    future_distribution = _field(output, "future")
    current_center = _first_present(current_distribution, ("center", "state", "mean"))
    future_centers = _first_present(future_distribution, ("state", "center", "mean"))
    current_target = targets["current"]
    current_valid = targets.get("current_valid")
    if current_valid is None:
        current_valid = torch.ones(current_target.shape[:-1], dtype=torch.bool, device=current_target.device)
    current_valid = current_valid.bool().reshape(current_target.shape[0])
    safe_current_target = _safe_state_targets(current_target, current_valid, current_center)

    current_log_density = log_prob(current_distribution, safe_current_target)
    if current_log_density.shape != current_valid.shape:
        raise ValueError("current log_prob must return one unnormalized density per batch item")
    current_nll = (
        -current_log_density / 51.0 * current_valid.to(current_log_density.dtype)
    ).mean()

    current_geometry = geometric_loss(current_center, safe_current_target)
    if current_geometry.shape != current_valid.shape:
        raise ValueError("geometric_loss must return one value per current state")
    restore = (current_geometry * current_valid.to(current_geometry.dtype)).sum() / current_valid.sum().clamp_min(1)

    future_target = targets["future"]
    future_valid = targets["future_valid"].bool()
    if future_valid.shape != future_target.shape[:-1]:
        raise ValueError("future_valid must match the leading [batch,time] target dimensions")
    safe_future_target = _safe_state_targets(future_target, future_valid, future_centers[:, 0])
    future_mix = mixture_log_prob(future_distribution, safe_future_target, future_valid)
    if future_mix.shape != future_valid.shape[:1]:
        raise ValueError("mixture_log_prob must return one unnormalized whole-trajectory density per item")
    future_denominator = (51.0 * future_valid.sum(dim=-1).clamp_min(1)).to(future_mix.dtype)
    future_nll = (-future_mix / future_denominator).mean()

    mode_geometry = geometric_loss(future_centers, safe_future_target[:, None])
    expected_geometry_shape = (
        future_valid.shape[0], future_centers.shape[1], future_valid.shape[1])
    if mode_geometry.shape != expected_geometry_shape:
        raise ValueError("future geometric_loss must return [batch,modes,time]")
    future_counts = future_valid.sum(dim=-1).clamp_min(1).to(mode_geometry.dtype)
    per_mode_geometry = (
        mode_geometry * future_valid[:, None, :].to(mode_geometry.dtype)
    ).sum(dim=-1) / future_counts[:, None]
    best_mode = per_mode_geometry.min(dim=-1).values.mean()

    mode_log_likelihood = _mode_log_likelihoods(
        future_distribution, safe_future_target, future_valid, log_prob)
    logits = _first_present(future_distribution, ("logits", "mode_logits"))
    posterior = torch.softmax(logits + mode_log_likelihood, dim=-1).detach()
    velocity_per_mode, _ = _velocity_errors(
        current_center, future_centers, safe_current_target, safe_future_target,
        targets["future_times"], future_valid)
    velocity = (posterior * velocity_per_mode).sum(dim=-1).mean()

    balance = _first_present(output, ("balance_loss", "balance"))
    if not torch.is_tensor(balance):
        balance = torch.as_tensor(balance, dtype=current_nll.dtype, device=current_nll.device)
    total = (
        restore
        + weights["current_nll"] * current_nll
        + weights["future_nll"] * future_nll
        + weights["best_mode"] * best_mode
        + weights["velocity"] * velocity
        + weights["balance"] * balance
    )
    return {
        "loss": total,
        "restore": restore,
        "current_nll": current_nll,
        "future_nll": future_nll,
        "best_mode": best_mode,
        "velocity": velocity,
        "balance": balance,
    }


def normalize_motion_batch(
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Use one detached past wrist anchor for all motion inputs and labels."""
    from .handloop_state import make_anchor, to_local

    anchor_rotation, anchor_position = make_anchor(
        batch["raw"], batch["history"], batch["raw_available"],
        batch.get("history_available"),
    )
    inputs = {
        "history": to_local(batch["history"], anchor_rotation, anchor_position),
        "raw": to_local(batch["raw"], anchor_rotation, anchor_position),
        "candidate": to_local(batch["candidate"], anchor_rotation, anchor_position),
        "history_available": batch["history_available"],
        "raw_available": batch["raw_available"],
        "candidate_available": batch["candidate_available"],
        "history_times": batch["history_times"],
        "dt": batch["dt"],
        "handedness": batch["handedness"],
        "history_mask": batch["history_mask"],
        "future_times": batch["future_times"],
    }
    targets = {
        "current": to_local(batch["current"], anchor_rotation, anchor_position),
        "current_valid": batch.get("current_valid"),
        "future": to_local(batch["future"], anchor_rotation, anchor_position),
        "future_valid": batch["future_valid"],
        "future_times": batch["future_times"],
    }
    return inputs, targets


def motion_training_step(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run one text-only motion batch through TrajectoryRestorer."""
    model_inputs, targets = normalize_motion_batch(batch)
    input_keys = (
        "history", "raw", "candidate", "history_available", "raw_available",
        "candidate_available", "history_times", "dt", "handedness", "history_mask",
        "future_times",
    )
    output = model(**{key: model_inputs[key] for key in input_keys})
    return motion_losses(output, targets)


def learning_rate_factor(step: int, total_steps: int, warmup_steps: int = 20,
                         final_ratio: float = 0.1) -> float:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= final_ratio <= 1.0:
        raise ValueError("final_ratio must be between zero and one")
    if warmup_steps > 0 and step < warmup_steps:
        return max(0.0, float(step + 1) / warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(
    named_parameters: Mapping[str, Iterable[nn.Parameter]],
    learning_rates: Mapping[str, float],
    *,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    parameter_values = {name: list(parameters) for name, parameters in named_parameters.items()}
    joint_names = {"hawor", "twm", "visual"}
    if {"hawor", "visual"}.intersection(parameter_values):
        missing = sorted(joint_names - set(parameter_values))
        empty = [name for name in joint_names if name in parameter_values
                 and not any(parameter.requires_grad for parameter in parameter_values[name])]
        if missing or empty:
            raise ValueError(
                f"Joint optimizer requires non-empty HaWoR/TWM/visual groups; "
                f"missing={missing}, empty={sorted(empty)}")
    groups: list[dict[str, Any]] = []
    for name, parameters in parameter_values.items():
        params = [parameter for parameter in parameters if parameter.requires_grad]
        if not params:
            continue
        if name not in learning_rates:
            raise ValueError(f"No learning rate configured for parameter group {name!r}")
        groups.append({
            "params": params,
            "lr": float(learning_rates[name]),
            "initial_lr": float(learning_rates[name]),
            "group_name": name,
        })
    if not groups:
        raise ValueError("optimizer needs at least one trainable parameter")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


class HandLoopTrainer:
    """Shared optimizer-step logic for motion pretraining and 3:1 joint updates."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int = 20,
        final_lr_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        global_step: int = 0,
    ) -> None:
        if total_steps < 1 or max_grad_norm <= 0:
            raise ValueError("total_steps and max_grad_norm must be positive")
        self.model = model
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_lr_ratio = float(final_lr_ratio)
        self.max_grad_norm = float(max_grad_norm)
        self.global_step = int(global_step)

    @staticmethod
    def task_for_step(step: int) -> str:
        return "motion" if int(step) % 4 == 3 else "visual"

    def optimize(self, loss: torch.Tensor) -> dict[str, float]:
        if loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise ValueError("training loss must be a finite scalar")
        factor = learning_rate_factor(
            self.global_step, self.total_steps, self.warmup_steps, self.final_lr_ratio)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        group_norms: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            name = str(group.get("group_name", "parameters"))
            parameters = [p for p in group["params"] if p.grad is not None]
            if parameters:
                norm = clip_grad_norm_(parameters, self.max_grad_norm)
                group_norms[name] = float(norm.detach().cpu())
            base_lr = float(group.get("initial_lr", group["lr"]))
            group["lr"] = base_lr * factor
        self.optimizer.step()
        self.global_step += 1
        return {
            "loss": float(loss.detach().cpu()),
            "lr_factor": factor,
            **{f"grad_norm/{name}": value for name, value in group_norms.items()},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "final_lr_ratio": self.final_lr_ratio,
            "max_grad_norm": self.max_grad_norm,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = ("total_steps", "warmup_steps", "final_lr_ratio", "max_grad_norm")
        for key in expected:
            if key in state and getattr(self, key) != state[key]:
                raise ValueError(f"Trainer checkpoint mismatch for {key}")
        self.global_step = int(state["global_step"])


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    trainer: HandLoopTrainer,
    optimizer: torch.optim.Optimizer,
    stage: str,
    config: Mapping[str, Any],
    split_fingerprint: str,
    base_checkpoint: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "architecture": "handloop_v2",
        "format": "handloop_training",
        "format_version": HANDLOOP_TRAINING_FORMAT_VERSION,
        "stage": stage,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trainer": trainer.state_dict(),
        "config": dict(config),
        "split_fingerprint": split_fingerprint,
        "rng": capture_rng_state(),
        "base_checkpoint": dict(base_checkpoint or {}),
        "extra": dict(extra or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    trainer: HandLoopTrainer,
    optimizer: torch.optim.Optimizer,
    stage: str,
    split_fingerprint: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if (checkpoint.get("architecture") != "handloop_v2"
            or checkpoint.get("format") != "handloop_training"
            or checkpoint.get("format_version") != HANDLOOP_TRAINING_FORMAT_VERSION):
        raise ValueError("Unsupported HandLoop training checkpoint format")
    if checkpoint.get("stage") != stage:
        raise ValueError(f"Cannot resume {checkpoint.get('stage')!r} checkpoint as {stage!r}")
    if checkpoint.get("split_fingerprint") != split_fingerprint:
        raise ValueError("Checkpoint manifest/split differs from the current training data")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.load_state_dict(checkpoint["trainer"])
    restore_rng_state(checkpoint["rng"])
    return checkpoint


def parameter_groups_for_system(system: nn.Module) -> dict[str, Iterable[nn.Parameter]]:
    """Return the three disjoint modules used by joint visual adaptation."""
    if not all(hasattr(system, name) for name in ("hawor", "observer")):
        raise ValueError("joint system must expose .hawor and .observer")
    observer = system.observer
    if not all(hasattr(observer, name) for name in ("twm", "visual")):
        raise ValueError("joint observer must expose .twm and .visual")
    groups = {
        "hawor": list(system.hawor.parameters()),
        "twm": list(observer.twm.parameters()),
        "visual": list(observer.visual.parameters()),
    }
    if any(not parameters for parameters in groups.values()):
        empty = [name for name, parameters in groups.items() if not parameters]
        raise ValueError(f"Joint optimizer has empty required parameter groups: {empty}")
    ids: list[int] = []
    for parameters in groups.values():
        ids.extend(id(parameter) for parameter in parameters)
    if len(ids) != len(set(ids)):
        raise ValueError("HaWoR, TWM, and visual parameter groups must be disjoint")
    return groups
