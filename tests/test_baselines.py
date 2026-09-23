import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from dp_accounting.pld.privacy_loss_mechanism import AdjacencyType, GaussianPrivacyLoss

from experiments import dp_hpo as hpo


def test_privacy_composition():
    budget = hpo.allocate_privacy_budget(0.4, 0.5, 4)
    assert budget.selection_train_trial_mu == pytest.approx(0.1)
    assert budget.selection_val_trial_mu == pytest.approx(0.1)
    assert math.hypot(budget.selection_total_mu, budget.final_training_mu) == pytest.approx(0.4)


def test_replace_one_accountant_uses_clip_radius_and_models_twice_the_shift(monkeypatch):
    seen = {}

    def accountant(**kwargs):
        seen.update(kwargs)
        return 7.0, kwargs['target_mu'], 0.0

    monkeypatch.setattr(hpo, 'compute_dpsgd_sigma_for_substitute_dp_fast', accountant)
    assert hpo.exact_noise_std(0.4, 0.1, 0.05, 20) == 7.0
    assert seen['max_grad_norm'] == pytest.approx(0.1)

    # At q=1 Google's REPLACE PLD compares N(-C, sigma^2) with N(+C, sigma^2).
    pld = GaussianPrivacyLoss(standard_deviation=1, sensitivity=1,
                              sampling_prob=1, adjacency_type=AdjacencyType.REPLACE)
    assert pld.get_delta_for_epsilon(0) == pytest.approx(math.erf(1 / math.sqrt(2)))


def test_split_does_not_depend_on_labels():
    x = np.arange(40).reshape(20, 2)
    y = np.arange(20) % 2
    altered = y.copy()
    altered[0] = 1 - altered[0]
    first = hpo.split_train_validation(x, y, 0.16, 12)
    second = hpo.split_train_validation(x, altered, 0.16, 12)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2], second[2])


def test_private_gradient_implementations_agree():
    model = hpo.SimpleMLP(3, 4, 1, 2)
    x, y = torch.randn(5, 3), torch.arange(5) % 2
    components = hpo.build_per_example_grad_components(model)
    slow = hpo.compute_private_gradients_slow(model, x, y, 0.2, 0, 5)
    fast = hpo.compute_private_gradients(model, components, x, y, 0.2, 0, 5, 2)
    for a, b in zip(slow, fast):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('model_code', ['dp_logistic_regression', 'dp_mlp'])
def test_private_selection_and_final_training(model_code):
    plan = replace(hpo.DEFAULT_MODEL_PLANS[model_code], epochs=1, q=1.0, learning_rates=(0.01, 0.02))
    x = np.random.default_rng(4).normal(size=(20, 3)).astype(np.float32)
    y = np.arange(20) % 2
    budget = hpo.allocate_privacy_budget(1.0, 0.5, 2)
    best, trials = hpo.select_learning_rate(x, y, budget, plan, 0.2, 5.0, 4, 4, 'test', 2)
    assert len(trials) == 2
    assert best['dp_val_loss'] == min(t['dp_val_loss'] for t in trials)
    model = hpo.train_private_model(torch.tensor(x), torch.tensor(y), budget.final_training_mu,
                                    plan, best['learning_rate'], 4, 4, 'test-final', 2)
    assert model(torch.tensor(x)).shape == (20, 2)


def test_empty_poisson_batch_still_updates(monkeypatch):
    plan = replace(hpo.DEFAULT_MODEL_PLANS['dp_logistic_regression'], epochs=1, q=1.0)
    model = hpo.SimpleLogisticRegression(3, 2).to(hpo.DEVICE)
    before = [p.detach().clone() for p in model.parameters()]
    monkeypatch.setattr(hpo, 'make_model', lambda *a, **kw: model)
    monkeypatch.setattr(hpo, 'sample_poisson_batch', lambda *a: (None, None))
    hpo.train_private_model(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]),
                             1.0, plan, 0.01, 7, 2, 'empty', 2)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_validation_noise_is_not_determined_by_public_seed():
    model = hpo.SimpleLogisticRegression(3, 2).to(hpo.DEVICE)
    x, y = torch.randn(5, 3), torch.arange(5) % 2
    first = hpo.evaluate_model_dp_noisy_validation_loss(model, x, y, 0.4, 5.0, seed=7)
    second = hpo.evaluate_model_dp_noisy_validation_loss(model, x, y, 0.4, 5.0, seed=7)
    assert first != second
