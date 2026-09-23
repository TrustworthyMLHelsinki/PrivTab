"""Audit the actual PrivTab path before its first private attention layer.

Checks whether one context row can affect another row's encoded token, whether
target inputs affect context tokens, and whether the initial private queries
depend on either input. This is an implementation check, not a DP proof.
"""

import argparse
import json
from pathlib import Path

import torch

from privtab.checkpoints import load_model
from privtab.model import PrivTab


class _StopAtBoundary(Exception):
    pass


def capture_first_private_inputs(model, xc, yc, xt):
    """Run ``forward`` and capture inputs before the first private layer."""
    first = model.encoder.transformer_encoder.mhca_ctoq_layers[0]
    captured = {}

    def capture(_module, args):
        captured["queries"], captured["context"] = args[:2]
        raise _StopAtBoundary

    handle = first.register_forward_pre_hook(capture)
    try:
        try:
            model(xc, yc, xt, mu=0.4)
        except _StopAtBoundary:
            pass
        else:
            raise RuntimeError("The first private-attention hook did not run.")
    finally:
        handle.remove()
    return captured["queries"], captured["context"]


def max_abs(tensor):
    return 0.0 if tensor is None else float(tensor.detach().abs().max())


def audit_rowwise(model, *, context_rows=5, target_rows=4, features=8,
                  batch_size=2, tolerance=1e-6, seed=123, device="cpu"):
    """Return measured cross-row and target dependencies at the DP boundary."""
    if context_rows < 2 or target_rows < 1 or not 1 <= features <= 120 or batch_size < 1:
        raise ValueError("Need at least two context rows, one target row, and 1–120 features.")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")
    torch.manual_seed(seed)  # Synthetic audit inputs only; no privacy release.
    model = model.to(device).eval()
    xc = torch.randn(batch_size, context_rows, features, device=device, requires_grad=True)
    xt = torch.randn(batch_size, target_rows, features, device=device, requires_grad=True)
    yc = torch.randint(0, 10, (batch_size, context_rows), device=device)
    queries, context = capture_first_private_inputs(model, xc, yc, xt)

    cross_row = 0.0
    target_to_context = 0.0
    for batch in range(batch_size):
        for row in range(context_rows):
            probe = torch.randn_like(context[batch, row])
            grad_xc, grad_xt = torch.autograd.grad(
                (context[batch, row] * probe).sum(), (xc, xt),
                retain_graph=True, allow_unused=True,
            )
            if grad_xc is not None:
                other = grad_xc.clone()
                other[batch, row] = 0
                cross_row = max(cross_row, max_abs(other))
            target_to_context = max(target_to_context, max_abs(grad_xt))

    grad_qc, grad_qt = torch.autograd.grad(
        queries.square().sum(), (xc, xt), allow_unused=True
    )
    query_input = max(max_abs(grad_qc), max_abs(grad_qt))
    # Integer labels have no gradients: replace one label and compare every
    # other context token at the same pre-noise boundary.
    altered_yc = yc.clone()
    altered_yc[0, 0] = (altered_yc[0, 0] + 1) % 10
    altered_queries, altered_context = capture_first_private_inputs(model, xc, altered_yc, xt)
    unchanged = (altered_context - context).detach().abs()
    unchanged[0, 0] = 0
    label_cross_row = float(unchanged.max())
    query_label = float((altered_queries - queries).detach().abs().max())
    result = {"cross_row_gradient_max": cross_row,
              "target_to_context_gradient_max": target_to_context,
              "query_input_gradient_max": query_input,
              "label_cross_row_difference_max": label_cross_row,
              "query_label_difference_max": query_label,
              "tolerance": tolerance}
    result["passed"] = all(value <= tolerance for value in
                           (cross_row, target_to_context, query_input,
                            label_cross_row, query_label))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Optional released checkpoint; otherwise audit the architecture.")
    parser.add_argument("--context-rows", type=int, default=5)
    parser.add_argument("--target-rows", type=int, default=4)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_model(args.weights, args.device) if args.weights else PrivTab()
    result = audit_rowwise(model, context_rows=args.context_rows,
                           target_rows=args.target_rows, features=args.features,
                           batch_size=args.batch_size, tolerance=args.tolerance,
                           seed=args.seed, device=args.device)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
