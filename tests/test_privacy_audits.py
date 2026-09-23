import numpy as np
import torch

from experiments.audit_rowwise import audit_rowwise
from experiments.dp_audit import audit, audited_sigmas, empirical_roc
from privtab.model import PrivTab


def test_rowwise_audit_detects_cross_row_encoding(monkeypatch):
    model = PrivTab()
    assert audit_rowwise(model, context_rows=3, target_rows=2, features=4)["passed"]

    original = model.encoder.encode

    def coupled_encode(x, labels, d):
        encoded = original(x, labels, d)
        return encoded + encoded.mean(dim=1, keepdim=True)

    monkeypatch.setattr(model.encoder, "encode", coupled_encode)
    result = audit_rowwise(model, context_rows=3, target_rows=2, features=4)
    assert not result["passed"]
    assert result["cross_row_gradient_max"] > 0
    assert result["label_cross_row_difference_max"] > 0


def test_noise_audit_uses_model_sigma_and_writes_results(tmp_path):
    torch.set_num_threads(2)
    model = PrivTab()
    sigma = 2 * (128 * 3) ** 0.5 / 0.4
    np.testing.assert_allclose(audited_sigmas(model, 0.4), [sigma] * 3, rtol=1e-6)
    rows = audit(model, mu_values=[0.4], steps=2, samples=8, batch_size=4,
                 output_dir=tmp_path, plot=False)
    assert rows[0]["passed"]
    assert rows[0]["sensitivity_observed"] <= rows[0]["sensitivity_bound"]
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "dp_mhca_audit_mu_0p4.npz").exists()


def test_empirical_roc_handles_ties():
    fpr, tpr = empirical_roc(np.array([1., 1., 0., 0.]), np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(fpr, [0., .5, 1.])
    np.testing.assert_array_equal(tpr, [0., .5, 1.])
