"""Inference adapter for protocol-3 training of the bounded feedback architecture."""
import runtime
import run_intra
import run_feedback_v2 as base

ARCHITECTURE = base.ARCHITECTURE
METHOD = base.METHOD
BOUND_KEYS = base.BOUND_KEYS
PROTOCOL_HASH_KEYS = ('world_reference_manifest_sha256', 'event_manifest_sha256', 'calibration_protocol_sha256', 'cache_manifest_sha256')


def validate_protocol(checkpoint):
    base.validate_feedback_checkpoint(checkpoint)
    trainer = checkpoint.get('trainer', {})
    if checkpoint.get('protocol_revision') != 3 or trainer.get('protocol_revision') != 3:
        raise ValueError('Protocol-3 training evidence is required')
    for key in PROTOCOL_HASH_KEYS:
        value = trainer.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing or invalid protocol binding: ' + key)
    return trainer


def load_feedback_system(**kwargs):
    checkpoint = runtime.twm_only.load_checkpoint_restricted(kwargs['visual_checkpoint_path'])
    trainer = validate_protocol(checkpoint)
    system, metadata = base.load_feedback_system(**kwargs)
    metadata.update(protocol_revision=3, **{key: trainer[key] for key in PROTOCOL_HASH_KEYS})
    return system, metadata


def configure_runtime():
    base.configure_runtime()
    run_intra.load_intra_system = load_feedback_system


if __name__ == '__main__':
    configure_runtime()
    run_intra.main()
