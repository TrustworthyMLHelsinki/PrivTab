"""Empirically audit PrivTab's private attention sensitivity and Gaussian noise.

Optimize a replace-one pair of context-token sets for the actual private
attention layer, then draw noisy outputs through that layer. A known-mean
likelihood-ratio test compares the empirical ROC with the Gaussian prediction
for the found pair and with the claimed per-layer GDP bound. This diagnostic
cannot prove DP; the analytic sensitivity argument remains necessary.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from privtab.checkpoints import load_model
from privtab.model import PrivTab
from privtab.secure_noise import secure_randn_like


def sensitivity_bound(attention, num_latents):
    return 2.0 * math.sqrt(attention.num_heads * num_latents)


def audited_sigmas(model, mu, *, features=8):
    """Capture all three sigma values passed by the real summary path."""
    device = next(model.parameters()).device
    captured = []
    handles = []

    def record(_module, args):
        captured.append(float(args[2][0]))

    for layer in model.encoder.transformer_encoder.mhca_ctoq_layers:
        handles.append(layer.attn.register_forward_pre_hook(record))
    try:
        with torch.inference_mode():
            xc = torch.zeros(1, 2, features, device=device)
            yc = torch.zeros(1, 2, dtype=torch.long, device=device)
            model.summarize(xc, yc, mu, noise_sampler=torch.zeros_like)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 3:
        raise RuntimeError(f"Expected three private Gaussian layers, found {len(captured)}.")
    return captured


def pre_noise(attention, queries, context):
    zero_sigma = torch.zeros(queries.shape[0], device=queries.device)
    return attention(queries, context, zero_sigma, False, torch.zeros_like)


def optimize_adjacent_pair(attention, *, context_rows=4, num_latents=128,
                           steps=200, lr=0.05, seed=123):
    """Maximize pre-noise distance for one replaced context token."""
    if context_rows < 1 or num_latents < 1 or steps < 0 or lr <= 0:
        raise ValueError("Context rows, latents, and learning rate must be positive.")
    device = next(attention.parameters()).device
    torch.manual_seed(seed)  # Synthetic search inputs only.
    width = attention.to_q.in_features
    queries = torch.nn.Parameter(torch.randn(1, num_latents, width, device=device))
    context = torch.nn.Parameter(torch.randn(1, context_rows, width, device=device))
    replacement = torch.nn.Parameter(torch.randn(1, 1, width, device=device))
    optimizer = torch.optim.Adam([queries, context, replacement], lr=lr)
    best = None
    best_distance = -1.0
    attention.eval()
    for step in range(steps + 1):
        adjacent = torch.cat([context[:, :-1], replacement], dim=1)
        first = pre_noise(attention, queries, context)
        second = pre_noise(attention, queries, adjacent)
        distance = torch.linalg.vector_norm(first - second)
        value = float(distance.detach())
        if value > best_distance:
            best_distance = value
            best = (queries.detach().clone(), context.detach().clone(),
                    adjacent.detach().clone(), first.detach().clone(), second.detach().clone())
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        (-distance).backward()
        optimizer.step()
    bound = sensitivity_bound(attention, num_latents)
    if best_distance > bound * (1 + 1e-5):
        raise AssertionError(f"Observed sensitivity {best_distance:g} exceeds bound {bound:g}.")
    return best, best_distance, bound


def sample_known_mean_scores(attention, pair, *, sigma, samples, batch_size,
                             rng="secure"):
    """Sample the real noise-addition path and compute optimal Gaussian scores."""
    if sigma <= 0 or samples < 2 or batch_size < 1:
        raise ValueError("Need positive sigma/batch size and at least two samples per class.")
    if rng not in {"secure", "torch"}:
        raise ValueError("rng must be 'secure' or 'torch'.")
    q, context, adjacent, mean, mean_prime = pair
    noise_sampler = secure_randn_like if rng == "secure" else None
    difference = (mean - mean_prime).flatten().to(torch.float64)
    midpoint = ((mean + mean_prime) / 2).flatten().to(torch.float64)
    scores_first, scores_second = [], []
    noise_sum = 0.0
    noise_square_sum = 0.0
    noise_count = 0
    with torch.inference_mode():
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples - start)
            sigma_batch = torch.full((n,), sigma, device=q.device)
            for context_batch, known_mean, destination in (
                (context, mean, scores_first), (adjacent, mean_prime, scores_second)
            ):
                output = attention(q.expand(n, -1, -1),
                                   context_batch.expand(n, -1, -1),
                                   sigma_batch, False, noise_sampler)
                residual = output.to(torch.float64) - known_mean.to(torch.float64)
                noise_sum += float(residual.sum())
                noise_square_sum += float(residual.square().sum())
                noise_count += residual.numel()
                score = ((output.reshape(n, -1).to(torch.float64) - midpoint)
                         @ difference) / sigma**2
                destination.append(score.cpu().numpy())
    scores = np.concatenate([*scores_first, *scores_second])
    labels = np.concatenate([np.ones(samples), np.zeros(samples)])
    mean_noise = noise_sum / noise_count
    std_noise = math.sqrt(max(0.0, noise_square_sum / noise_count - mean_noise**2))
    return scores, labels, mean_noise, std_noise


def empirical_roc(scores, labels):
    """ROC points including ties, without depending on a classifier library."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    positives = labels[order].astype(bool)
    tp = np.cumsum(positives)
    fp = np.cumsum(~positives)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tpr = np.r_[0.0, tp[ends] / positives.sum()]
    fpr = np.r_[0.0, fp[ends] / (~positives).sum()]
    return fpr, tpr


def plot_roc(path, fpr, tpr, mu_pair, mu_bound):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    alpha = np.linspace(0.001, 0.999, 999)
    figure, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label="Empirical known-mean LR", color="tab:red")
    ax.plot(alpha, norm.cdf(norm.ppf(alpha) + mu_pair),
            label=f"Found pair (mu={mu_pair:.3g})", color="tab:green")
    ax.plot(alpha, norm.cdf(norm.ppf(alpha) + mu_bound), "--",
            label=f"GDP bound (mu={mu_bound:.3g})", color="tab:blue")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="Indistinguishable")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1), title="PrivTab private-attention audit")
    ax.legend()
    ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def audit(model, *, mu_values, layer_index=0, context_rows=4, steps=200,
          samples=512, batch_size=32, seed=123, rng="secure",
          noise_tolerance=0.05, output_dir="runs/dp_audit", plot=True):
    if layer_index not in range(3):
        raise ValueError("layer_index must be 0, 1, or 2.")
    if noise_tolerance <= 0:
        raise ValueError("noise_tolerance must be positive.")
    model.eval()
    attention = model.encoder.transformer_encoder.mhca_ctoq_layers[layer_index].attn
    pair, observed_distance, bound = optimize_adjacent_pair(
        attention, context_rows=context_rows, steps=steps, seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mu in mu_values:
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError("All mu values must be finite and positive.")
        measured_sigmas = audited_sigmas(model, mu)
        sigma = bound * math.sqrt(3) / mu
        if any(abs(actual / sigma - 1) > 1e-5 for actual in measured_sigmas):
            raise AssertionError(f"Summarizer sigma mismatch at mu={mu}: {measured_sigmas}, expected {sigma}.")
        scores, labels, noise_mean, noise_std = sample_known_mean_scores(
            attention, pair, sigma=sigma, samples=samples,
            batch_size=batch_size, rng=rng,
        )
        fpr, tpr = empirical_roc(scores, labels)
        tag = format(mu, ".12g").replace(".", "p")
        np.savez_compressed(output_dir / f"dp_mhca_audit_mu_{tag}.npz",
                            fpr=fpr, tpr=tpr, scores=scores, labels=labels,
                            sigma=sigma, sensitivity=observed_distance)
        if plot:
            plot_roc(output_dir / f"dp_mhca_audit_mu_{tag}.pdf", fpr, tpr,
                     observed_distance / sigma, mu / math.sqrt(3))
        row = {"mu_total": mu, "layer_index": layer_index,
               "sensitivity_observed": observed_distance, "sensitivity_bound": bound,
               "sigma_expected": sigma, "sigmas_in_model": measured_sigmas,
               "noise_mean": noise_mean, "noise_std_observed": noise_std,
               "noise_std_relative_error": abs(noise_std / sigma - 1),
               "mu_found_pair": observed_distance / sigma,
               "mu_layer_bound": mu / math.sqrt(3),
               "rng": rng, "samples_per_class": samples}
        row["passed"] = row["noise_std_relative_error"] <= noise_tolerance
        results.append(row)
        print(json.dumps(row), flush=True)
    (output_dir / "audit.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Optional checkpoint; otherwise audit an initialized layer.")
    parser.add_argument("--mu-values", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--context-rows", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=512, help="Noisy outputs per adjacent input.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rng", choices=["secure", "torch"], default="secure")
    parser.add_argument("--seed", type=int, default=123, help="Synthetic search seed only.")
    parser.add_argument("--noise-tolerance", type=float, default=0.05)
    parser.add_argument("--output-dir", default="runs/dp_audit")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_model(args.weights, args.device) if args.weights else PrivTab().to(args.device)
    results = audit(model, mu_values=args.mu_values, layer_index=args.layer_index,
                    context_rows=args.context_rows, steps=args.steps,
                    samples=args.samples, batch_size=args.batch_size,
                    seed=args.seed, rng=args.rng,
                    noise_tolerance=args.noise_tolerance,
                    output_dir=args.output_dir, plot=not args.no_plot)
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
