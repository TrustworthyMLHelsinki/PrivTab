import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

from experiments import dp_hpo, evaluate
from experiments.eval_tabular_with_ranking import build_battles, load_results


def test_saved_indices_are_reused(tmp_path):
    data = [('toy', (np.arange(60).reshape(30, 2), np.arange(30) % 2))]
    indices = np.arange(29, -1, -1)
    path = tmp_path / 'toy_repeat_0.pickle'
    joblib.dump(indices, path)
    dp_hpo.ensure_indices(data, str(tmp_path), 1, .8)
    np.testing.assert_array_equal(joblib.load(path), indices)


def test_context_routing_and_saved_result_schema(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    datasets = [('short', (rng.normal(size=(20, 4)).astype('float32'), np.arange(20) % 2)),
                ('long', (rng.normal(size=(40, 4)).astype('float32'), np.arange(40) % 2))]
    cache = tmp_path / 'datasets.pickle'
    joblib.dump(datasets, cache)
    weights_loaded = []
    class FakeModel:
        def __init__(self, path):
            self.path = path
        def summarize(self, xc, yc, mu, d, normalize_perturbed_output):
            return torch.tensor([1.])
        def predict_from_summary(self, summary, xt, d):
            return torch.zeros(1, xt.shape[1], 10)
    def load(path, device):
        weights_loaded.append(path)
        return FakeModel(path)
    monkeypatch.setattr(evaluate, 'load_model', load)
    output = tmp_path / 'privtab.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--datasets-cache', str(cache), '--indices-dir', str(tmp_path / 'indices'),
                                   '--output', str(output), '--repeats', '1', '--context-switch-threshold', '25',
                                   '--mu-values', '.4'])
    evaluate.main()
    rows = json.loads(output.read_text())
    assert len(rows) == 2
    assert {r['dataset']: Path(r['checkpoint']).parent.name for r in rows} == {
        'short': 'stage2_short_contexts', 'long': 'stage2_long_contexts'}
    assert len(weights_loaded) == 2
    parsed = load_results(output)
    assert {k: list(v['privtab']) for k, v in parsed.items()} == {'short': [.4], 'long': [.4]}


def test_saved_result_keys():
    metrics = {'aucs': [.7, .8], 'losses': [.5, .4]}
    methods = [('PrivTab', 'privtab'), ('DP-LR', 'dp_logistic_regression'),
               ('DP-MLP', 'dp_mlp'), ('Non-DP Logistic Regression', 'logistic_regression')]
    tables = []
    for name, key in methods:
        mu_key = 'non_dp' if name.startswith('Non-DP') else .4
        tables.append((name, key, {'toy': {'binary': True, key: {mu_key: metrics}}}))
    battles = build_battles(tables, [.4])
    assert len(battles) == 8
    assert battles.task.nunique() == 2
