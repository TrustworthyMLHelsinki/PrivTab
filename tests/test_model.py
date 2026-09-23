from pathlib import Path

import pytest
import torch

from privtab import PrivTab, ReleasedSummary
from privtab.checkpoints import load_model
from deployment import export_bundle, export_bundle_from_summary, load_bundle


torch.set_num_threads(2)


def test_original_checkpoint_logits():
    fixtures = torch.load(Path(__file__).parent / 'fixtures/original_logits.pt', weights_only=True)
    for fixture in fixtures:
        path = Path('models') / fixture['stage'] / 'weights.pt'
        assert path.exists(), f'Missing repository checkpoint: {path}'
        model = load_model(path)
        torch.manual_seed(101)
        with torch.no_grad():
            logits = model(fixture['xc'], fixture['yc'], fixture['xt'], fixture['mu'], fixture['d'], fixture['normalize'])
        torch.testing.assert_close(logits, fixture['logits'], rtol=1e-5, atol=1e-5)


def test_summary_roundtrip_and_query_batching(tmp_path):
    model = PrivTab().eval()
    xc, yc, xt = torch.randn(13, 5), torch.randint(0, 3, (13,)), torch.randn(7, 5)
    release = ReleasedSummary.create(model, xc, yc, 0.4, 3)
    summary_path = tmp_path / 'summary.pt'
    release.save(summary_path)
    assert set(torch.load(summary_path, weights_only=True)) == {
        'format_version', 'summary', 'mu', 'num_features', 'num_classes'}
    loaded_release = ReleasedSummary.load(summary_path)
    expected = loaded_release.predict_proba(model, xt)
    torch.testing.assert_close(loaded_release.predict_proba(model, xt, batch_size=2), expected)
    path = tmp_path / 'bundle.pt'
    export_bundle_from_summary(model, loaded_release, path)
    deployed = load_bundle(path)
    rng = torch.get_rng_state()
    actual = deployed.predict_proba(xt)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(deployed.predict_proba(xt, batch_size=2), expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(rng, torch.get_rng_state())
    assert not any('ctoq' in key or 'latents' in key for key in deployed.state_dict())
    bundle = torch.load(path, weights_only=True)
    assert set(bundle) == {'format_version', 'mu', 'num_classes', 'num_features', 'state_dict'}
    assert set(key for key in bundle['state_dict'] if 'summary' in key) == {'summary'}
    with pytest.raises(FileExistsError):
        export_bundle(model, xc, yc, 0.4, 3, path)


def test_private_release_uses_os_rng_not_torch_seed(monkeypatch):
    from privtab import secure_noise

    calls = []
    original = secure_noise.secrets.token_bytes

    def tracked(n):
        calls.append(n)
        return original(n)

    monkeypatch.setattr(secure_noise.secrets, 'token_bytes', tracked)
    model = PrivTab().eval()
    xc, yc = torch.randn(1, 3, 4), torch.zeros(1, 3)
    state = torch.get_rng_state()
    first = model.release_summary(xc, yc, 0.4)
    assert torch.equal(state, torch.get_rng_state())
    second = model.release_summary(xc, yc, 0.4)
    assert len(calls) == 6  # Three fresh Gaussian arrays per release.
    assert all(n == 128 * 256 * 8 for n in calls)
    assert not torch.equal(first, second)
    assert first.grad_fn is None


@pytest.mark.parametrize('mu', [0, -1, float('nan'), float('inf')])
def test_reject_invalid_privacy(mu):
    with pytest.raises(ValueError, match='mu'):
        PrivTab().summarize(torch.randn(1, 3, 4), torch.zeros(1, 3), mu)


def test_new_summary_uses_fresh_noise():
    model = PrivTab().eval()
    xc, yc = torch.randn(1, 3, 4), torch.zeros(1, 3)
    with torch.no_grad():
        first = model.summarize(xc, yc, 0.4)
        second = model.summarize(xc, yc, 0.4)
    assert not torch.equal(first, second)


def test_backward_through_summary():
    model = PrivTab()
    logits = model(torch.randn(1, 5, 4), torch.zeros(1, 5), torch.randn(1, 2, 4), 0.4)
    logits.square().mean().backward()
    assert model.encoder.transformer_encoder.mhca_ctoq_layers[0].attn.to_v.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
