"""Run the bounded, non-recursive-TWM feedback model with frozen row-1 inputs.

The v1 runtime's data loading and source/frame checks are reused unchanged.
Only this process's model loader and version labels are adapted.
"""
from __future__ import annotations

import math
from pathlib import Path

import runtime
import run_intra

ARCHITECTURE = "handloop_intra_v2"
METHOD = "TWM + bounded Intra-frame Feedback v2; raw HaWoR history"
BOUND_KEYS = ("translation_bound", "root_rotation_bound", "finger_rotation_bound")
_load_original = runtime.load_intra_system
_validate_original = runtime.validate_intra_checkpoint


def validate_feedback_checkpoint(checkpoint):
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Only an explicitly versioned handloop_intra_v2 checkpoint is accepted")
    validated = _validate_original(checkpoint)
    bounds = checkpoint.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(BOUND_KEYS):
        raise ValueError("v2 checkpoint must explicitly record all three visual bounds")
    if any(type(bounds[k]) not in (int, float) or not math.isfinite(bounds[k])
           or bounds[k] <= 0 for k in BOUND_KEYS):
        raise ValueError("Invalid v2 visual residual bounds")
    return validated


def load_feedback_system(**kwargs):
    from feedback_v2 import install_feedback_v2

    configure_runtime()
    checkpoint = runtime.twm_only.load_checkpoint_restricted(kwargs["visual_checkpoint_path"])
    validate_feedback_checkpoint(checkpoint)
    bounds = checkpoint["bounds"]
    system, metadata = _load_original(**kwargs)
    install_feedback_v2(system, **bounds)
    system.eval()
    metadata.update(
        architecture=ARCHITECTURE, method=METHOD, bounds=bounds,
        candidate_rule="accumulated visual updates only; TWM output is reobserved, not accumulated",
        feedback_implementation_sha256=runtime.sha256(Path(__file__).with_name("feedback_v2.py")),
        v2_inference_adapter_sha256=runtime.sha256(Path(__file__)),
        calibration_selection=checkpoint.get("calibration_selection"),
    )
    return system, metadata


def configure_runtime():
    runtime.INTRA_ARCHITECTURE = ARCHITECTURE
    runtime.INTRA_METHOD = METHOD
    runtime.validate_intra_checkpoint = validate_feedback_checkpoint
    run_intra.INTRA_ARCHITECTURE = ARCHITECTURE
    run_intra.validate_intra_checkpoint = validate_feedback_checkpoint
    run_intra.load_intra_system = load_feedback_system


if __name__ == "__main__":
    configure_runtime()
    run_intra.main()
