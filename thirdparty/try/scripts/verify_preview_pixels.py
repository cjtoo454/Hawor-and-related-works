#!/usr/bin/env python3
"""Compare HOT3D preview pixels against timestamp-matched original VRS frames."""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vrs', type=Path, required=True)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 1200, 2500])
    args = parser.parse_args()
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    provider = data_provider.create_vrs_data_provider(str(args.vrs))
    if provider is None:
        raise RuntimeError('Could not open VRS')
    container = av.open(str(args.preview))
    times = json.loads(container.metadata['description'])
    wanted = set(args.frames)
    rows = []
    for index, frame in enumerate(container.decode(video=0)):
        if index not in wanted:
            continue
        device_ns = int(times[index])
        raw, record = provider.get_image_data_by_time_ns(
            StreamId('214-1'), device_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        raw = raw.to_numpy_array()
        preview = frame.to_ndarray(format='rgb24')
        errors = {}
        for quarter_turns in range(4):
            candidate = np.rot90(raw, -quarter_turns)
            if candidate.shape != preview.shape:
                continue
            errors[str(quarter_turns * 90)] = float(np.abs(
                candidate.astype(np.float32) - preview.astype(np.float32)).mean())
        best = min(errors, key=errors.get)
        rows.append({'frame': index, 'preview_device_ns': device_ns,
                     'vrs_device_ns': int(record.capture_timestamp_ns),
                     'raw_to_preview_clockwise_degrees': int(best),
                     'mean_absolute_pixel_errors': errors})
        if len(rows) == len(wanted):
            break
    if len(rows) != len(wanted):
        raise ValueError('Not all requested frames decoded')
    rotations = {row['raw_to_preview_clockwise_degrees'] for row in rows}
    verified = (len(rotations) == 1
                and all(min(row['mean_absolute_pixel_errors'].values()) < 12 for row in rows)
                and all(abs(row['preview_device_ns'] - row['vrs_device_ns']) <= 2 for row in rows))
    result = {'verified_raw_fisheye_preview': verified,
              'preview_sha256': sha256(args.preview),
              'vrs_sha256': sha256(args.vrs),
              'vrs': str(args.vrs.resolve()), 'preview': str(args.preview.resolve()),
              'frames': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not verified:
        raise RuntimeError('Preview pixel transform has not been verified')


if __name__ == '__main__':
    main()
