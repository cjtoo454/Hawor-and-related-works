"""Read-only local resource inventory; never unpickles checkpoint contents."""
from pathlib import Path
import hashlib
import json
import pickletools
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

def audit():
    checkpoints = []
    for path in sorted((ROOT / 'thirdparty/try').rglob('*.pt')):
        item = {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size}
        strings = []
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
                if names:
                    strings = [arg for op, arg, pos in pickletools.genops(archive.read(names[0]))
                               if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE')]
        item['v2_architecture_marker_present'] = 'handloop_v2' in strings
        item['stage_markers'] = [v for v in ('motion_pretrain', 'joint_visual') if v in strings]
        checkpoints.append(item)
    protocol = ROOT / 'diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json'
    return {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_inspection': 'Static pickle opcode string inspection, no object deserialization; markers are inventory evidence, not strict compatibility validation.',
        'checkpoints': checkpoints,
        'validation_root_exists': (ROOT / 'datasets/hot3d_valset_export').is_dir(),
        'frozen_protocol': str(protocol.relative_to(ROOT)),
        'frozen_protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'status': 'missing_v2_checkpoint_and_original_validation_inputs',
        'formal_table_row': {'Variant': 'TWM only', 'W-MPJPE_mm': None, 'Onset_MAE_ms': None},
        'note': 'Archived evaluation_details.npz supplies XYZ joints only, not the full MANO rotations needed for v2 TWM input. No formal experiment was run.'
    }

if __name__ == '__main__':
    target = ROOT / 'thirdparty/ablation/records/environment_audit.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit(), indent=2) + '\n')
    print(target)
