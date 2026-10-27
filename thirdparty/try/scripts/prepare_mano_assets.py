#!/usr/bin/env python3
"""Convert the locally licensed legacy MANO pickle to arrays; never downloads assets."""
import argparse
import inspect
import pickle
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / '_DATA/data/mano/MANO_RIGHT.pkl')
    parser.add_argument('--output', type=Path, default=root / 'thirdparty/try/assets/mano_right.npz')
    args = parser.parse_args()
    # Chumpy is only needed for decoding this old, trusted local asset.
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'unicode': str, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    with args.source.open('rb') as handle:
        legacy = pickle.load(handle, encoding='latin1')
    arrays = {}
    for key, value in legacy.items():
        if hasattr(value, 'toarray'):
            value = value.toarray()
        elif hasattr(value, 'r'):
            value = value.r
        arrays[key] = np.asarray(value)
        if arrays[key].dtype == object:
            raise ValueError(f'Non-numeric model field: {key}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(args.output)


if __name__ == '__main__':
    main()
