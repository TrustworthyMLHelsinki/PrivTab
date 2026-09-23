"""Create a benchmark cache from numerical classification datasets in NPZ files."""
import argparse
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Each NPZ contains x [rows, features] and y [rows].')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    datasets = []
    for path in sorted(args.input_dir.glob('*.npz')):
        with np.load(path, allow_pickle=False) as arrays:
            x, y = arrays['x'].astype(np.float32), arrays['y']
        if x.ndim != 2 or not 1 <= x.shape[1] <= 120 or y.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError(f'Invalid numerical dataset: {path}')
        classes = np.unique(y)
        if not 2 <= len(classes) <= 10 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError(f'{path}: use public integer class IDs 0 through C-1, with 2–10 classes.')
        datasets.append((path.stem, (x, y.astype(np.int64))))
    if not datasets:
        raise ValueError('No NPZ files found.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(datasets, args.output)
    print(f'Saved {len(datasets)} datasets to {args.output}')


if __name__ == '__main__':
    main()
