"""Reusable pinch-event extraction, filtering, and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Event:
    """A confirmed event with inclusive frame indices and sample-time bounds.

    ``onset_sec`` is the timestamp of the first pinched frame. ``offset_sec``
    is the timestamp of the first known open frame after the inclusive
    ``stop`` frame.
    """

    start: int
    stop: int
    onset_sec: float
    offset_sec: float


def _validated_timestamps(timestamps: np.ndarray, length: int) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or len(times) != length:
        raise ValueError("timestamps must be one-dimensional with one value per frame")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return times


def _validated_mask(valid: np.ndarray, length: int) -> np.ndarray:
    mask = np.asarray(valid)
    if mask.ndim != 1 or len(mask) != length:
        raise ValueError("valid must be one-dimensional with one value per frame")
    return mask.astype(bool, copy=False)


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def extract_events(
    distance: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    min_frames: int = 3,
    max_gap_s: float = 0.25,
) -> list[Event]:
    """Extract confirmed below-threshold events from a quality-aware trace.

    An event is a run of finite, valid distances strictly below ``threshold``.
    It is returned only when a valid open sample precedes and follows the run
    within the same continuous time segment. This excludes events censored at
    either edge, beside unknown samples, or across gaps larger than
    ``max_gap_s``. Frame indices in :class:`Event` are inclusive.
    """
    distances = np.asarray(distance, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("distance must be one-dimensional")
    times = _validated_timestamps(timestamps, len(distances))
    mask = _validated_mask(valid, len(distances))
    threshold = _finite_scalar(threshold, "threshold")
    if (isinstance(min_frames, bool) or not isinstance(min_frames, Integral)
            or min_frames < 1):
        raise ValueError("min_frames must be a positive integer")
    min_frames = int(min_frames)
    max_gap_s = _finite_scalar(max_gap_s, "max_gap_s")
    if max_gap_s < 0:
        raise ValueError("max_gap_s must be non-negative")

    known = mask & np.isfinite(distances)
    pinched = distances < threshold
    events: list[Event] = []
    index = 0
    while index < len(distances):
        if not known[index]:
            index += 1
            continue

        segment_start = index
        index += 1
        while (index < len(distances) and known[index]
               and times[index] - times[index - 1] <= max_gap_s):
            index += 1
        segment_stop = index  # exclusive

        cursor = segment_start
        while cursor < segment_stop:
            if not pinched[cursor]:
                cursor += 1
                continue

            run_start = cursor
            while cursor < segment_stop and pinched[cursor]:
                cursor += 1
            run_stop = cursor  # exclusive
            run_length = run_stop - run_start
            has_open_brackets = (run_start > segment_start
                                 and run_stop < segment_stop)
            if run_length >= min_frames and has_open_brackets:
                events.append(Event(
                    start=run_start,
                    stop=run_stop - 1,
                    onset_sec=float(times[run_start]),
                    offset_sec=float(times[run_stop]),
                ))

    return events


def causal_ema(
    joints: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    tau_sec: float = 0.1,
    max_gap_s: float = 0.25,
) -> np.ndarray:
    """Apply a causal time-aware EMA, resetting at every evidence boundary.

    The result has the same shape as ``joints`` and is NaN at every invalid
    input frame. Valid frames retain the input quality mask. The first valid
    sample after an invalid frame, long time gap, or segment change initializes
    the filter directly, so no values are carried across those boundaries.
    """
    values = np.asarray(joints, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (21, 3):
        raise ValueError("joints must have shape [frames, 21, 3]")
    times = _validated_timestamps(timestamps, len(values))
    mask = _validated_mask(valid, len(values))
    segments = np.asarray(segment_id)
    if segments.ndim != 1 or len(segments) != len(values):
        raise ValueError("segment_id must be one-dimensional with one value per frame")
    tau_sec = _finite_scalar(tau_sec, "tau_sec")
    if tau_sec <= 0:
        raise ValueError("tau_sec must be positive")
    max_gap_s = _finite_scalar(max_gap_s, "max_gap_s")
    if max_gap_s < 0:
        raise ValueError("max_gap_s must be non-negative")

    filtered = np.full(values.shape, np.nan, dtype=np.float64)
    previous_valid = False
    for index in range(len(values)):
        if not mask[index]:
            previous_valid = False
            continue

        same_segment = False
        if previous_valid:
            try:
                same_segment = bool(segments[index] == segments[index - 1])
            except (TypeError, ValueError):
                same_segment = False
        delta = times[index] - times[index - 1] if previous_valid else np.inf
        if not previous_valid or not same_segment or delta > max_gap_s:
            filtered[index] = values[index]
        else:
            alpha = -np.expm1(-delta / tau_sec)
            filtered[index] = filtered[index - 1] + alpha * (
                values[index] - filtered[index - 1])
        previous_valid = True

    return filtered


def _event_duration_frames(event: Event) -> int:
    if (isinstance(event.start, bool) or isinstance(event.stop, bool)
            or not isinstance(event.start, Integral)
            or not isinstance(event.stop, Integral)
            or event.start < 0 or event.stop < event.start):
        raise ValueError("event frame indices must be ordered non-negative integers")
    return int(event.stop - event.start + 1)


def _validate_events(events: Sequence[Event], name: str) -> list[Event]:
    result = list(events)
    for event in result:
        if not isinstance(event, Event):
            raise TypeError(f"{name} must contain Event instances")
        _event_duration_frames(event)
        onset = _finite_scalar(event.onset_sec, f"{name} onset_sec")
        offset = _finite_scalar(event.offset_sec, f"{name} offset_sec")
        if offset < onset:
            raise ValueError(f"{name} event offset_sec must not precede onset_sec")
    return result


def match_events(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    onset_tolerance_sec: float = 0.2,
) -> dict[str, object]:
    """Match events one-to-one by overlap and onset tolerance.

    Assignment first maximizes the number of eligible matches, then minimizes
    total absolute onset error. ``onset_error_ms`` and ``onset_bias_ms`` use
    prediction minus reference, so positive values mean late predictions.
    Short-event counts are based on reference runs of 3 through 10 inclusive
    frames; short misses remain included in the overall FN and recall counts.
    Undefined ratios and onset summaries are returned as ``None``.
    """
    refs = _validate_events(reference, "reference")
    preds = _validate_events(predicted, "predicted")
    tolerance = _finite_scalar(onset_tolerance_sec, "onset_tolerance_sec")
    if tolerance < 0:
        raise ValueError("onset_tolerance_sec must be non-negative")

    ref_count = len(refs)
    pred_count = len(preds)
    eligible = np.zeros((ref_count, pred_count), dtype=bool)
    onset_errors = np.zeros((ref_count, pred_count), dtype=np.float64)
    for ref_index, ref_event in enumerate(refs):
        for pred_index, pred_event in enumerate(preds):
            error = pred_event.onset_sec - ref_event.onset_sec
            overlaps = (max(ref_event.onset_sec, pred_event.onset_sec)
                        < min(ref_event.offset_sec, pred_event.offset_sec))
            if overlaps and abs(error) <= tolerance:
                eligible[ref_index, pred_index] = True
                onset_errors[ref_index, pred_index] = abs(error)

    pairs: list[tuple[int, int]] = []
    if ref_count and pred_count and np.any(eligible):
        finite_errors = onset_errors[eligible]
        # Scaling all real edges equally preserves their absolute-error order
        # and keeps the cardinality penalty safely above every possible finite
        # assignment cost, even for unusually large time origins.
        scale = max(1.0, float(np.max(finite_errors)))
        feasible_edge_count = int(np.count_nonzero(eligible))
        unmatched_penalty = float(feasible_edge_count + 1)
        invalid_edge_cost = unmatched_penalty + 1.0
        costs = np.full((ref_count, pred_count + ref_count), unmatched_penalty,
                        dtype=np.float64)
        costs[:, :pred_count] = invalid_edge_cost
        costs[:, :pred_count][eligible] = onset_errors[eligible] / scale

        rows, columns = linear_sum_assignment(costs)
        pairs = [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if column < pred_count and eligible[row, column]
        ]
        pairs.sort()

    matched_ref = {ref_index for ref_index, _ in pairs}
    matched_pred = {pred_index for _, pred_index in pairs}
    unmatched_ref = [index for index in range(ref_count) if index not in matched_ref]
    unmatched_pred = [index for index in range(pred_count) if index not in matched_pred]
    tp = len(pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_ref)

    signed_errors_ms = [
        (preds[pred_index].onset_sec - refs[ref_index].onset_sec) * 1000.0
        for ref_index, pred_index in pairs
    ]
    short_ref_indices = {
        index for index, event in enumerate(refs)
        if 3 <= _event_duration_frames(event) <= 10
    }
    short_matched = sum(ref_index in short_ref_indices for ref_index, _ in pairs)
    short_reference_count = len(short_ref_indices)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "onset_mae_ms": (float(np.mean(np.abs(signed_errors_ms)))
                         if signed_errors_ms else None),
        "onset_bias_ms": (float(np.mean(signed_errors_ms))
                          if signed_errors_ms else None),
        "short_reference_events": short_reference_count,
        "short_matched_events": short_matched,
        "short_event_recall": (short_matched / short_reference_count
                               if short_reference_count else None),
        "matches": [
            {
                "reference_index": ref_index,
                "prediction_index": pred_index,
                "onset_error_ms": signed_errors_ms[match_index],
            }
            for match_index, (ref_index, pred_index) in enumerate(pairs)
        ],
        "unmatched_reference_indices": unmatched_ref,
        "unmatched_prediction_indices": unmatched_pred,
    }
