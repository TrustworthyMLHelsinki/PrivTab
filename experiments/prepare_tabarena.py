"""Cache the 33 TabArena v0.1 tasks for PrivTab benchmark evaluation."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.multiclass import check_classification_targets


def encode_openml_dataset(dataset):
    """Apply ordinal encoding, numeric imputation, and constant-column filtering."""
    target = dataset.default_target_attribute
    if not isinstance(target, str):
        raise ValueError(f'{dataset.name}: expected one default target column')
    frame, labels, _, _ = dataset.get_data(target=target, dataset_format='dataframe')
    check_classification_targets(labels)
    y = LabelEncoder().fit_transform(labels)
    categorical = make_column_selector(dtype_include=['string', 'object', 'category', 'boolean'])(frame)
    numeric = make_column_selector(dtype_include='number')(frame)
    columns = ColumnTransformer([
        ('categorical', OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value',
                                       unknown_value=-1, encoded_missing_value=-1),
         [frame.columns.get_loc(c) for c in categorical]),
        ('continuous', SimpleImputer(), [frame.columns.get_loc(c) for c in numeric]),
    ])
    x = np.asarray(columns.fit_transform(frame))
    keep = np.array([len(np.unique(x[:, i])) > 1 for i in range(x.shape[1])])
    x = x[:, keep]
    if not 1 <= x.shape[1] <= 120 or not 2 <= len(np.unique(y)) <= 10 or not np.isfinite(x).all():
        raise ValueError(f'{dataset.name}: invalid encoded dimensions or values')
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/tabarena_datasets.pickle'))
    parser.add_argument('--suite', default='tabarena-v0.1')
    parser.add_argument('--expected-datasets', type=int, default=33)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Cache already exists: {args.output}')
    import openml
    suite = openml.study.get_suite(args.suite)
    datasets = []
    for task_id in suite.tasks:
        dataset = openml.tasks.get_task(task_id).get_dataset()
        raw = dataset.get_data()[0]
        target = dataset.default_target_attribute
        class_count = len(raw[target].unique())
        if len(raw) > 100000 or len(raw.columns) > 120 or class_count > 10:
            continue
        x, y = encode_openml_dataset(dataset)
        datasets.append((dataset.name, (x, y)))
        print(f'{dataset.name}: {x.shape[0]} rows, {x.shape[1]} features, {len(np.unique(y))} classes', flush=True)
    if len(datasets) != args.expected_datasets:
        raise ValueError(f'Expected {args.expected_datasets} datasets, got {len(datasets)}; cache was not written.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(datasets, args.output)
    print(f'Saved {len(datasets)} datasets to {args.output}')


if __name__ == '__main__':
    main()
