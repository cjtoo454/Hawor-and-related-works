from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

import run_mini_ablation as runner  # noqa: E402
from run_hot3d import load_checkpoint  # noqa: E402


def _minimal_plan() -> dict:
    sequences = (
        ("P0001_9b6feab7", 1050),
        ("P0011_76ea6d47", 2040),
        ("P0014_84ea2dcc", 3060),
    )
    return {
        "frames_per_clip": 900,
        "fixed_event_protocol": "missing_frozen_protocol.json",
        "legacy_checkpoint": "missing_checkpoint.pt",
        "legacy_variants": [
            {"intra_frame_reobserve": False, "corrected_state_feedback": False},
            {"intra_frame_reobserve": True, "corrected_state_feedback": False},
            {"intra_frame_reobserve": True, "corrected_state_feedback": True},
        ],
        "clips": [
            {
                "sequence": sequence,
                "start_frame": start,
                "stop_frame_exclusive": start + 900,
                "warmup_start_frame": start - 32,
            }
            for sequence, start in sequences
        ],
    }


def test_variant_configs_preserve_two_steps_and_only_change_requested_switches(tmp_path: Path) -> None:
    config_path = REPOSITORY_ROOT / "thirdparty" / "try" / "configs" / "default.yaml"
    args = type("Args", (), {"config": config_path})()
    model_config = {
        "state_dim": 109,
        "feature_dim": 64,
        "evidence_dim": 128,
        "hidden_dim": 256,
        "history_length": 16,
        "correction_steps": 2,
        "image_size": 256,
        "intra_frame_reobserve": True,
        "corrected_state_feedback": True,
        "quality_decay_on_missing": 0.92,
        "min_trace_quality": 0.20,
        "world_model_variant": "legacy",
    }
    variants = runner.variant_configs(args, {"config": {"model": model_config}})

    assert [
        (config.intra_frame_reobserve, config.corrected_state_feedback, config.correction_steps)
        for _variant, config in variants
    ] == [(False, False, 2), (True, False, 2), (True, True, 2)]


def test_w_mpjpe_uses_only_contiguous_chunks_at_least_sixteen_frames() -> None:
    rng = np.random.default_rng(12)
    gt = rng.normal(size=(35, 21, 3)).astype(np.float32)
    # A constant translation is removed by first-align W-MPJPE.
    pred = gt + np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
    frame_ids = np.r_[np.arange(20), np.arange(25, 40)]

    mean_mm, scored_frames, by_frame = runner.score_w_mpjpe(
        frame_ids, pred, gt, np.ones(35, dtype=bool)
    )

    assert mean_mm is not None and mean_mm < 1e-3
    assert scored_frames == 20
    assert np.isfinite(by_frame[:20]).all()
    assert np.isnan(by_frame[20:]).all()


def test_w_mpjpe_rejects_nonfinite_joints_on_valid_frames() -> None:
    gt = np.ones((20, 21, 3), dtype=np.float32)
    pred = gt.copy()
    pred[4, 8, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite W-MPJPE joints on valid frames"):
        runner.score_w_mpjpe(np.arange(20), pred, gt, np.ones(20, dtype=bool))


def test_common_mask_must_match_frozen_prediction_valid_exactly() -> None:
    valid = np.asarray([True, False, True, False])

    np.testing.assert_array_equal(
        runner.baseline_mask_from_frames(np.asarray([0, 2]), valid, 4), valid
    )
    with pytest.raises(ValueError, match="do not match frozen baseline"):
        runner.baseline_mask_from_frames(np.asarray([0, 1, 2]), valid, 4)


def test_frontend_cannot_drop_common_frames_even_in_short_geometry_chunks() -> None:
    common = np.ones(14, dtype=bool)
    available = common.copy()
    available[3] = False
    with pytest.raises(ValueError, match="frontend lost frozen common frames"):
        runner.fixed_prediction_mask(common, available, np.arange(14))
    np.testing.assert_array_equal(runner.fixed_prediction_mask(common, common, np.arange(14)), common)


def test_protocol_parameters_are_frozen_to_the_selected_values() -> None:
    valid = {
        "reference_threshold": 0.2,
        "prediction_threshold": 0.275,
        "min_frames": 3,
        "onset_tolerance_sec": 0.2,
    }
    assert runner._protocol_issues(valid) == []
    changed = {**valid, "prediction_threshold": 0.25}
    assert any("prediction_threshold must be 0.275" in x for x in runner._protocol_issues(changed))


def test_missing_dataset_preflight_writes_blocker_without_entering_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_minimal_plan()), encoding="utf-8")
    output_root = tmp_path / "preflight"

    def inference_must_not_run(*_args, **_kwargs):
        raise AssertionError("model inference must not run when required inputs are absent")

    monkeypatch.setattr(runner, "run_experiment", inference_must_not_run)
    result = runner.main([
        "--plan", str(plan_path),
        "--video_root", str(tmp_path / "missing_video_root"),
        "--frontend_log_root", str(tmp_path / "missing_frontend"),
        "--eval_log_root", str(tmp_path / "missing_eval"),
        "--frozen_signal_root", str(tmp_path / "missing_signals"),
        "--output_root", str(output_root),
    ])

    assert result == 2
    report = json.loads((output_root / "missing_inputs.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked_missing_inputs"
    assert report["inference_started"] is False
    assert any("missing_video_root" in item["path"] for item in report["missing_inputs"])
    assert not (output_root / "summary.json").exists()


def test_checkpoint_loader_rejects_world_model_pretrain_stage(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "prior_only.pt"
    torch.save({"stage": "world_model_pretrain", "model": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="prior-only weights"):
        load_checkpoint(checkpoint_path, torch.device("cpu"))
