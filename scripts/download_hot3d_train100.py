"""Download fixed, participant-disjoint HOT3D RGB and supervision files."""
import concurrent.futures as cf
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/hot3d_train100'
TYPES = ['video_main_rgb', 'ground_truth', 'hand_data', 'mps_slam_trajectories', 'mps_slam_calibration', 'mps_slam_summary']

def main():
    manifest = json.loads((ROOT / 'hot3d/data_downloader/Hot3DAria_download_urls.json').read_text())['sequences']
    tests = json.loads((ROOT / 'datasets/hot3d_valset_export/val.json').read_text())
    excluded = {s.split('_')[0] for s in tests}
    selected = sorted(s for s in manifest if s.split('_')[0] not in excluded)[:100]
    assert len(selected) == 100 and not set(selected) & set(tests)
    OUT.mkdir(exist_ok=True)
    (OUT / 'train.json').write_text(json.dumps(selected, indent=2))
    (OUT / 'selection.json').write_text(json.dumps({'excluded_test_sequences': tests, 'excluded_participants': sorted(excluded), 'data_types': TYPES}, indent=2))
    jobs = [(s, k, manifest[s][k]) for s in selected for k in TYPES]
    total = sum(x[2]['file_size_bytes'] for x in jobs)
    print(f'START {len(selected)} sequences, {len(jobs)} files, {total / 1e9:.2f} GB', flush=True)
    def download(job):
        s, kind, item = job
        folder = OUT / s
        folder.mkdir(exist_ok=True)
        path = folder / item['filename']
        partial = path.with_suffix(path.suffix + '.part')
        def valid():
            if not path.exists() or path.stat().st_size != item['file_size_bytes']:
                return False
            h = hashlib.sha1()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    h.update(chunk)
            return h.hexdigest() == item['sha1sum']
        if valid():
            return s, kind, None
        for attempt in range(3):
            r = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '30', '--max-time', '1800', '--retry', '4', '--retry-delay', '3', '--continue-at', '-', '--output', str(partial), item['download_url']], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if r.returncode == 0:
                partial.replace(path)
                if valid():
                    return s, kind, None
                path.unlink()
            time.sleep(3)
        return s, kind, f'failed curl/checksum (curl={r.returncode})'
    completed = []
    errors = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for future in cf.as_completed([pool.submit(download, job) for job in jobs]):
            s, kind, error = future.result()
            (errors if error else completed).append({'sequence': s, 'type': kind, 'error': error})
            status = {'completed_files': len(completed), 'total_files': len(jobs), 'failed_files': len(errors), 'total_bytes': total, 'updated_at': time.time(), 'errors': errors}
            (OUT / 'download_status.json').write_text(json.dumps(status, indent=2))
            print(f'{len(completed)}/{len(jobs)} OK; {len(errors)} failed; {s} {kind} {error or "OK"}', flush=True)
    print('FINISHED', flush=True)
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
