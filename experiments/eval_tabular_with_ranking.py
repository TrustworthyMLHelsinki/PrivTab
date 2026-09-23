"""ELO rankings for PrivTab, DP MLP, and DP Logistic Regression benchmark results.

Uses TabArena Elo scoring, with per-mu and pooled rankings.
Binary tasks use 1 - AUC; multiclass tasks use log loss.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from experiments.end_to_end_cases_elo import compute_elo, find_mu_key, metric_errors


METHODS = {'PrivTab': ('privtab',),
           'DP-LR': ('dp_logistic_regression',),
           'DP-MLP': ('dp_mlp',)}
DISPLAY_NAMES = {'DP-LR': 'DP Logistic Regression', 'DP-MLP': 'DP MLP'}


def load_results(path):
    path = Path(path)
    if path.suffix != '.json':
        results = joblib.load(path)
        if not isinstance(results, dict):
            raise TypeError(f'Expected nested result dictionary in {path}')
        return results
    rows = json.loads(path.read_text())
    results = {}
    for row in sorted(rows, key=lambda row: (row['dataset'], row['mu'], row['repeat'])):
        if 'binary' not in row:
            raise ValueError(f'{path}: JSON rows must record whether the dataset is binary; rerun experiments.evaluate.')
        dataset = results.setdefault(row['dataset'], {'binary': row['binary'], 'privtab': {}})
        metric = dataset['privtab'].setdefault(float(row['mu']), {'aucs': [], 'losses': [], 'repeat_ids': []})
        metric['aucs'].append(row['auc'])
        metric['losses'].append(row['loss'])
        metric['repeat_ids'].append(row['repeat'])
    return results


def merge_results(paths):
    merged = {}
    for path in paths:
        for dataset, methods in load_results(path).items():
            target = merged.setdefault(dataset, {})
            for code, by_mu in methods.items():
                if code == 'binary':
                    if code in target and target[code] != by_mu:
                        raise ValueError(f'Conflicting class metadata for {dataset}')
                    target[code] = by_mu
                    continue
                destination = target.setdefault(code, {})
                for mu, metrics in by_mu.items():
                    if find_mu_key(destination, mu) is not None:
                        raise ValueError(f'Duplicate result block: {dataset}, {code}, mu={mu}')
                    destination[mu] = metrics
    return merged


def build_battles(results_table, mu_values, datasets_filter=None):
    """Require matched tasks across all methods; datasets_filter excludes names."""
    tables = list(results_table)
    expected_methods = {name for name, _, _ in tables}
    if len(expected_methods) != len(tables) or len(tables) < 2:
        raise ValueError('Ranking needs at least two distinct methods.')
    metadata = {}
    for _, _, results in tables:
        for dataset, methods in results.items():
            if 'binary' in methods:
                if dataset in metadata and metadata[dataset] != bool(methods['binary']):
                    raise ValueError(f'Conflicting class metadata for {dataset}')
                metadata[dataset] = bool(methods['binary'])
    rows = []
    for method, code, results in tables:
        for dataset, methods in results.items():
            if datasets_filter and dataset in datasets_filter:
                continue
            if code not in methods:
                continue
            if dataset not in metadata:
                raise ValueError(f'Missing binary/multiclass metadata for {dataset}')
            for mu in mu_values:
                key = find_mu_key(methods[code], mu)
                if key is None and method == 'Non-DP Logistic Regression':
                    key = 'non_dp' if 'non_dp' in methods[code] else None
                if key is None:
                    raise ValueError(f'Missing mu={mu} for {dataset}, {method}')
                metrics = methods[code][key]
                errors = metric_errors(metrics, metadata[dataset])
                repeats = list(metrics.get('repeat_ids', range(len(errors))))
                if len(repeats) != len(errors) or len(set(repeats)) != len(repeats):
                    raise ValueError(f'Invalid repeat identifiers: {dataset}, {method}')
                for repeat, error in zip(repeats, errors):
                    rows.append(dict(method=method, task=f'{dataset}_repeat{repeat}_mu{mu:g}',
                                     dataset=dataset, repeat=repeat, mu=mu, metric_error=float(error)))
    if not rows:
        raise ValueError('No matching results to rank.')
    battles = pd.DataFrame(rows)
    for task, group in battles.groupby('task'):
        if len(group) != len(expected_methods) or set(group['method']) != expected_methods:
            raise ValueError(f'Missing or duplicate method results for task {task}')
    return battles


def generate_rankings(datasets_filter, mu_values, results_table,
                      calibration_framework='DP-LR', output_dir='runs/rankings',
                      bootstrap_rounds=10000, seed=123):
    if calibration_framework != 'DP-LR':
        raise ValueError('The three-method comparison is calibrated to DP-LR = 1000.')
    battles = build_battles(results_table, mu_values, datasets_filter)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = {}
    for mu in [*mu_values, 'all']:
        subset = battles if mu == 'all' else battles[battles['mu'] == mu]
        elo = compute_elo(subset, bootstrap_rounds, seed)
        output = elo.reset_index().rename(columns={'method': 'method_name', 'Elo': 'elo_ranking',
                                                   '95% CI min': 'ci_min', '95% CI max': 'ci_max'})
        output['method_name'] = output['method_name'].replace(DISPLAY_NAMES)
        output['mu'] = mu
        suffix = '' if mu == 'all' else f'_mu_{mu:g}'
        output.to_csv(output_dir / f'rank_models_dp_elo{suffix}.csv', index=False)
        print(f'ELO, mu={mu}\n{output.to_string(index=False)}')
        rankings[mu] = output
    return rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--privtab-results', required=True)
    parser.add_argument('--baseline-results', nargs='+', required=True)
    parser.add_argument('--reference-results', help='Optional saved non-DP logistic regression results for four-method ELO.')
    parser.add_argument('--privtab-code', help='Explicit PrivTab method key in a saved result pickle.')
    parser.add_argument('--mu-values', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    parser.add_argument('--output-dir', default='runs/rankings')
    parser.add_argument('--expected-datasets', type=int, default=33,
                        help='Require all 33 TabArena datasets; set to another count for a subset.')
    parser.add_argument('--bootstrap-rounds', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    privtab = load_results(args.privtab_results)
    baseline = merge_results(args.baseline_results)
    if len(privtab) != args.expected_datasets or len(baseline) != args.expected_datasets:
        raise ValueError(f'Expected {args.expected_datasets} datasets in each result source; got '
                         f'{len(privtab)} PrivTab and {len(baseline)} baseline datasets.')
    tables = []
    for name, aliases in METHODS.items():
        source = privtab if name == 'PrivTab' else baseline
        if name == 'PrivTab' and args.privtab_code:
            aliases = (args.privtab_code,)
        found = {key for values in source.values() for key in aliases if key in values}
        if len(found) != 1:
            raise ValueError(f'Expected one result key for {name}; found {sorted(found)}')
        tables.append((name, found.pop(), source))
    if args.reference_results:
        reference = load_results(args.reference_results)
        if len(reference) != args.expected_datasets:
            raise ValueError(f'Expected {args.expected_datasets} non-DP reference datasets, got {len(reference)}.')
        found = {code for values in reference.values()
                 for code in ('logistic_regression', 'non_dp_logistic_regression') if code in values}
        if len(found) != 1:
            raise ValueError(f'Expected one non-DP logistic regression result key; found {sorted(found)}')
        tables.append(('Non-DP Logistic Regression', found.pop(), reference))
    generate_rankings(None, args.mu_values, tables, output_dir=args.output_dir,
                      bootstrap_rounds=args.bootstrap_rounds, seed=args.seed)


if __name__ == '__main__':
    main()
