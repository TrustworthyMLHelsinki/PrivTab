from dataclasses import replace
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest
import torch

from experiments import dataset_utils, dp_hpo, end_to_end_cases as cases
from experiments import end_to_end_cases_elo as case_elo
from experiments import eval_tabular_with_ranking as ranking
from experiments import run_end_to_end_dp_baselines as baselines
from privtab import PrivTab


@pytest.fixture
def case_cache(tmp_path):
    rng = np.random.default_rng(9)
    case = dict(slug='pima_diabetes', dataset_name='Synthetic case',
                x=rng.uniform(-2, 2, size=(40, 4)).astype(np.float32), y=np.arange(40) % 2,
                feature_mins=np.full(4, -2.), feature_maxs=np.full(4, 2.),
                oracle_feature_mins=np.full(4, -1.9), oracle_feature_maxs=np.full(4, 1.9))
    case['columns'] = [dict(name=f'f{i}', type='numeric') for i in range(4)]
    case['bounds_results'] = {f'f{i}': dict(suggested_lower_clip=-2., suggested_upper_clip=2., confidence='high') for i in range(4)}
    path = tmp_path / 'cases.pickle'
    joblib.dump({'pima_diabetes': case}, path)
    return path, case


def test_preprocessing_budget_and_finite_outputs(case_cache):
    _, case = case_cache
    prep_mu, model_mu = baselines.split_total_mu_for_preprocessing(0.4, 0.6)
    assert prep_mu == pytest.approx(0.24)
    assert model_mu == pytest.approx(0.32)
    values = dataset_utils.pre_process_and_split_data_dp_range_zscore_with_public_ranges(
        0.8, case['x'], case['y'], prep_mu, case['feature_mins'], case['feature_maxs'],
        indices=np.arange(40))
    assert values[0].shape == (32, 120)
    assert values[2].shape == (8, 120)
    assert np.isfinite(values[0]).all() and np.isfinite(values[2]).all()
    assert np.all(values[0][:, 4:] == 0)
    with pytest.raises(ValueError, match='bounds'):
        dataset_utils.pre_process_and_split_data_dp_range_zscore_with_public_ranges(
            0.8, case['x'], case['y'], prep_mu, [2]*4, [-2]*4)


def test_privtab_end_to_end_cli(case_cache, tmp_path, monkeypatch):
    cache, _ = case_cache
    import privtab.checkpoints
    monkeypatch.setattr(privtab.checkpoints, 'load_model', lambda *a, **k: PrivTab().eval())
    monkeypatch.setattr(sys, 'argv', ['end_to_end_cases', '--case-datasets-cache', str(cache),
                                   '--dataset', 'pima_diabetes', '--repeats', '1',
                                   '--indices-dir', str(tmp_path / 'indices'),
                                   '--output', str(tmp_path / 'results.pickle')])
    cases.main_dp_pre_processing_comparison()
    results = joblib.load(tmp_path / 'results.pickle')['Synthetic case']
    assert results['binary']
    assert set(results) == {'binary', 'privtab_exact_zscore', 'privtab_dp_oracle_bounds', 'privtab_dp_llm_bounds'}
    for key, value in results.items():
        if key != 'binary':
            assert len(value[0.4]['losses']) == 1
            assert value[0.4]['repeat_ids'] == [0]
    assert (tmp_path / 'results.csv').is_file()
    assert (tmp_path / 'results.md').is_file()


def test_end_to_end_baseline_cli(case_cache, tmp_path, monkeypatch):
    cache, _ = case_cache
    for code, plan in dp_hpo.DEFAULT_MODEL_PLANS.items():
        monkeypatch.setitem(dp_hpo.DEFAULT_MODEL_PLANS, code,
                            replace(plan, epochs=1, q=1., learning_rates=(0.01, 0.02)))
    monkeypatch.setattr(sys, 'argv', ['run_end_to_end_dp_baselines', '--case-datasets-cache', str(cache),
                                   '--dataset', 'pima_diabetes', '--repeats', '1',
                                   '--indices-dir', str(tmp_path / 'indices'), '--output-dir', str(tmp_path),
                                   '--chunk-size', '8'])
    baselines.main()
    baselines.print_saved_results(tmp_path)
    results = joblib.load(tmp_path / 'end-to-end-dp-baselines-all-results.pickle')['Synthetic case']
    assert results['binary']
    for key in ('dp_mlp_dp_llm_bounds_dp_zscore', 'dp_logistic_regression_dp_llm_bounds_dp_zscore'):
        metrics = results[key][0.4]
        assert len(metrics['losses']) == 1
        assert metrics['repeat_ids'] == [0]
        assert metrics['selected_params'][0]['training_total_mu'] == pytest.approx(0.32)


def sample_results():
    metrics = [dict(aucs=[0.9, 0.6, 0.7, 0.8], losses=[0.2, 0.5, 0.3, 0.4]),
               dict(aucs=[0.7, 0.9, 0.6, 0.8], losses=[0.4, 0.1, 0.5, 0.3]),
               dict(aucs=[0.6, 0.7, 0.9, 0.8], losses=[0.5, 0.4, 0.2, 0.3])]
    private = {'toy': {'binary': True, 'privtab_dp_llm_bounds': {0.4: metrics[0]}}}
    baseline = {'toy': {'dp_logistic_regression_dp_llm_bounds_dp_zscore': {0.4: metrics[1]},
                        'dp_mlp_dp_llm_bounds_dp_zscore': {0.4: metrics[2]}}}
    return private, baseline, metrics


def test_end_to_end_elo(tmp_path):
    private, baseline, _ = sample_results()
    battles = case_elo.build_battles(private, baseline, 0.4)
    assert len(battles) == 12
    assert set(battles['method']) == {'PrivTab', 'DP-LR', 'DP-MLP'}
    elo = case_elo.compute_elo(battles, bootstrap_rounds=10, seed=3)
    assert elo.loc['DP-LR', 'Elo'] == pytest.approx(1000)
    assert np.isfinite(elo.to_numpy()).all()
    baseline['toy']['dp_mlp_dp_llm_bounds_dp_zscore'][0.4]['repeat_ids'] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match='repeat identifiers'):
        case_elo.build_battles(private, baseline, 0.4)


def test_per_mu_and_pooled_rankings(tmp_path):
    _, _, metrics = sample_results()
    tables = [(name, code, {'toy': {'binary': True, code: {0.4: data, 0.8: data}}})
              for name, code, data in zip(('PrivTab', 'DP-LR', 'DP-MLP'),
                                          ('privtab', 'dp_logistic_regression', 'dp_mlp'), metrics)]
    result = ranking.generate_rankings(None, [0.4, 0.8], tables, output_dir=tmp_path,
                                       bootstrap_rounds=10, seed=3)
    assert set(result) == {0.4, 0.8, 'all'}
    assert len(list(tmp_path.glob('*.csv'))) == 3
    tables[2][2]['toy']['dp_mlp'][0.4]['repeat_ids'] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match='Missing or duplicate'):
        ranking.build_battles(tables, [0.4])
