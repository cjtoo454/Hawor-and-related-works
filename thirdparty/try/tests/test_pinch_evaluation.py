from __future__ import annotations

import numpy as np

from dual_loop.pinch_evaluation import Event, causal_ema, extract_events, match_events


def test_extract_events_excludes_censored_unknown_and_short_runs() -> None:
    distance = np.full(20, 0.5)
    valid = np.ones(20, dtype=bool)
    distance[0:2] = 0.1  # Left-censored at the trace boundary.
    valid[4] = False
    distance[5:8] = 0.1  # Left-censored after unknown evidence.
    distance[9:11] = 0.1  # Closed but shorter than min_frames.
    distance[12:15] = 0.1  # Confirmed event.
    distance[16:19] = 0.1
    valid[19] = False  # Right-censored by unknown evidence.

    events = extract_events(
        distance,
        timestamps=np.arange(20, dtype=np.float64) * 0.1,
        valid=valid,
        threshold=0.2,
        min_frames=3,
    )

    assert len(events) == 1
    assert (events[0].start, events[0].stop) == (12, 14)
    assert np.isclose(events[0].onset_sec, 1.2)
    assert np.isclose(events[0].offset_sec, 1.5)


def test_extract_events_does_not_bridge_large_time_gaps() -> None:
    distance = np.array([0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.5])
    timestamps = np.array([10.0, 10.1, 10.2, 10.3, 10.7, 10.8, 11.2, 11.3, 11.4, 11.5])

    events = extract_events(
        distance, timestamps, np.ones(10, dtype=bool), threshold=0.2,
        min_frames=3, max_gap_s=0.25,
    )

    # Both pinched sections touch a long gap at one edge and are censored.
    assert events == []


def test_causal_ema_step_response_mask_and_segment_resets() -> None:
    joints = np.zeros((5, 21, 3), dtype=np.float64)
    joints[:, 0, 0] = [0.0, 1.0, 1.0, 8.0, 0.0]
    timestamps = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
    valid = np.array([True, True, True, False, True])
    segment_id = np.array([0, 0, 1, 1, 1])

    filtered = causal_ema(joints, timestamps, valid, segment_id, tau_sec=0.1)

    alpha = -np.expm1(-1.0)
    assert np.isclose(filtered[0, 0, 0], 0.0)
    assert np.isclose(filtered[1, 0, 0], alpha)
    assert np.isclose(filtered[2, 0, 0], 1.0)  # Segment change resets to raw input.
    assert np.isnan(filtered[3]).all()  # Same quality mask as the input.
    assert np.isclose(filtered[4, 0, 0], 0.0)  # Invalid frame resets the filter.

    changed_future = joints.copy()
    changed_future[4, 0, 0] = 1000.0
    filtered_changed = causal_ema(
        changed_future, timestamps, valid, segment_id, tau_sec=0.1,
    )
    np.testing.assert_allclose(filtered_changed[:3], filtered[:3])


def test_match_events_is_one_to_one_and_maximizes_cardinality() -> None:
    references = [
        Event(0, 20, 0.0, 2.0),
        Event(30, 32, 0.3, 0.5),
    ]
    predictions = [
        Event(0, 2, 0.1, 0.4),  # Eligible for both references.
        Event(3, 5, -0.2, 0.15),  # Eligible only for the first reference.
        Event(6, 8, 0.24, 0.28),  # A second fragment of the first reference.
    ]

    result = match_events(references, predictions, onset_tolerance_sec=0.25)

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["unmatched_prediction_indices"] == [2]
    assert [(m["reference_index"], m["prediction_index"])
            for m in result["matches"]] == [(0, 1), (1, 0)]


def test_match_events_short_counts_use_inclusive_lengths_and_actual_times() -> None:
    references = [
        Event(2, 4, 50.0, 50.4),  # 3 frames, short.
        Event(10, 19, 51.0, 51.5),  # 10 frames, short.
        Event(30, 40, 52.0, 52.6),  # 11 frames, not short.
    ]
    predictions = [
        Event(7, 9, 50.12, 50.3),
        Event(20, 29, 51.0, 51.4),
    ]

    result = match_events(references, predictions)

    assert result["tp"] == 2
    assert result["fn"] == 1  # The 11-frame miss remains in full recall.
    assert result["short_reference_events"] == 2
    assert result["short_matched_events"] == 2
    assert result["short_event_recall"] == 1.0
    assert np.isclose(result["matches"][0]["onset_error_ms"], 120.0)
    assert np.isclose(result["onset_mae_ms"], 60.0)
    assert np.isclose(result["onset_bias_ms"], 60.0)


def test_match_events_returns_none_for_undefined_empty_metrics() -> None:
    result = match_events([], [])

    assert result["tp"] == result["fp"] == result["fn"] == 0
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["onset_mae_ms"] is None
    assert result["onset_bias_ms"] is None
    assert result["short_event_recall"] is None
