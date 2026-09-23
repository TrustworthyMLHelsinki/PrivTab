"""Release one PrivTab summary for scikit-learn's public 0-vs-1 digits data.

The 8x8 pixel schema has public bounds [0, 16]. Clipping and rescaling use
only those bounds, never statistics fitted on context rows. For a real private
dataset, use a public schema and account for any other private-data release.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits

from privtab import ReleasedSummary
from privtab.checkpoints import load_model


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models/stage2_short_contexts/weights.pt"
PIXEL_MIN = 0.0
PIXEL_MAX = 16.0


def public_pixel_transform(rows):
    """Map the documented pixel interval [0, 16] to [-1, 1]."""
    clipped = np.clip(rows, PIXEL_MIN, PIXEL_MAX)
    return (2.0 * clipped / (PIXEL_MAX - PIXEL_MIN) - 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--mu", type=float, default=0.4)
    parser.add_argument("--context-rows", type=int, default=256)
    parser.add_argument("--query-rows", type=int, default=8)
    parser.add_argument("--summary", type=Path,
                        help="Optionally save the one released summary; refuses overwrite.")
    args = parser.parse_args()
    if not np.isfinite(args.mu) or args.mu <= 0:
        parser.error("--mu must be finite and positive")
    if args.context_rows < 1 or args.query_rows < 1:
        parser.error("row counts must be positive")
    if args.summary is not None and args.summary.exists():
        parser.error(f"summary already exists: {args.summary}")

    # This bundled dataset is public; treating its context as private illustrates
    # the release API. The index permutation is fixed independently of the data.
    digits = load_digits(n_class=2)
    if args.context_rows + args.query_rows > len(digits.data):
        parser.error("context rows plus query rows exceed the available digits")
    order = np.random.default_rng(2026).permutation(len(digits.data))
    context_indices = order[:args.context_rows]
    query_indices = order[args.context_rows:args.context_rows + args.query_rows]
    xc = torch.from_numpy(public_pixel_transform(digits.data[context_indices]))
    yc = torch.as_tensor(digits.target[context_indices], dtype=torch.long)
    xt = torch.from_numpy(public_pixel_transform(digits.data[query_indices]))

    model = load_model(args.weights)
    released = ReleasedSummary.create(model, xc, yc, mu=args.mu, num_classes=2)
    if args.summary is not None:
        released.save(args.summary)
        released = ReleasedSummary.load(args.summary)

    probabilities = released.predict_proba(model, xt).numpy()
    for index, ((p0, p1), expected) in enumerate(
        zip(probabilities, digits.target[query_indices])
    ):
        print(f"query {index}: expected={int(expected)}  P(0)={p0:.3f}  P(1)={p1:.3f}")
    print(f"One summary released at mu={args.mu:g}; {len(probabilities)} predictions reuse it.")
    if args.summary is not None:
        print(f"Saved and reloaded summary: {args.summary}")


if __name__ == "__main__":
    main()
