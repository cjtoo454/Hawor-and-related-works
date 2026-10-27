"""Run centered intra-frame feedback with raw HaWoR history and frozen bases."""
import math
import json
from pathlib import Path

import torch
import runtime
import run_intra

ARCHITECTURE = 'handloop_intra_v4'
METHOD = 'TWM + centered Intra-frame Feedback v4; raw HaWoR history'
BOUND_KEYS = ('translation_bound', 'root_rotation_bound', 'finger_rotation_bound')
PROTOCOL_HASH_KEYS = ('world_reference_manifest_sha256', 'event_manifest_sha256', 'calibration_protocol_sha256', 'cache_manifest_sha256')
_infer_original = runtime.infer_sequence
_load_original = runtime.load_intra_system
_validate_original = runtime.validate_intra_checkpoint


def validate_feedback_checkpoint(checkpoint):
    if checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('protocol_revision') != 4:
        raise ValueError('Only an explicit centered-feedback protocol-4 checkpoint is accepted')
    validated = _validate_original(checkpoint)
    trainer = checkpoint['trainer']
    if trainer.get('protocol_revision') != 4:
        raise ValueError('Training evidence is not protocol 4')
    for key in PROTOCOL_HASH_KEYS:
        value = trainer.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing or invalid source binding: ' + key)
    bounds = checkpoint.get('bounds')
    if not isinstance(bounds, dict) or set(bounds) != set(BOUND_KEYS):
        raise ValueError('All three physical residual bounds must be explicit')
    if any(type(bounds[k]) not in (int, float) or not math.isfinite(bounds[k]) or bounds[k] <= 0 for k in BOUND_KEYS):
        raise ValueError('Invalid physical residual bounds')
    scale = checkpoint['visual'].get('delta_scale')
    expected = torch.tensor([bounds['translation_bound']] * 3 + [bounds['root_rotation_bound']] * 3 + [bounds['finger_rotation_bound']] * 45)
    if not isinstance(scale, torch.Tensor) or scale.shape != (51,) or not torch.allclose(scale.cpu().float(), expected, atol=1e-9, rtol=1e-6):
        raise ValueError('Visual head scale differs from the recorded v4 physical bounds')
    return validated


def load_feedback_system(**kwargs):
    from feedback_v4 import install_feedback_v4
    configure_runtime()
    checkpoint = runtime.twm_only.load_checkpoint_restricted(kwargs['visual_checkpoint_path'])
    validate_feedback_checkpoint(checkpoint)
    system, metadata = _load_original(**kwargs)
    install_feedback_v4(system, **checkpoint['bounds'])
    system.eval()
    trainer = checkpoint['trainer']
    metadata.update(architecture=ARCHITECTURE, method=METHOD, protocol_revision=4,
                    bounds=checkpoint['bounds'], candidate_rule='second-round visual candidate updates the state it observes; centered TWM innovation',
                    distribution_semantics='centered state with approximate frozen-restorer uncertainty; no distribution NLL training',
                    feedback_implementation_sha256=runtime.sha256(Path(__file__).with_name('feedback_v4.py')),
                    v4_inference_adapter_sha256=runtime.sha256(Path(__file__)),
                    **{key: trainer[key] for key in PROTOCOL_HASH_KEYS})
    return system, metadata


def infer_sequence(*args, **kwargs):
    metadata = _infer_original(*args, **kwargs)
    metadata.update(twm_predictions_per_frame=3, twm_calls_per_frame=3,
                    twm_calls_per_observer_batch=3, visual_reobservation_rounds=2,
                    centered_reference_calls_per_frame=1)
    path = Path(kwargs['output_dir']) / 'try_inference.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(path)
    return metadata


def configure_runtime():
    runtime.INTRA_ARCHITECTURE = ARCHITECTURE
    runtime.INTRA_METHOD = METHOD
    runtime.validate_intra_checkpoint = validate_feedback_checkpoint
    run_intra.INTRA_ARCHITECTURE = ARCHITECTURE
    run_intra.validate_intra_checkpoint = validate_feedback_checkpoint
    run_intra.load_intra_system = load_feedback_system
    run_intra.infer_sequence = infer_sequence


if __name__ == '__main__':
    configure_runtime()
    run_intra.main()
