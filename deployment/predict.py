"""Predict using a deployment bundle, without context data or W&B access."""
import argparse

import numpy as np
import torch

from .bundle import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--features", required=True, help="NPY array using the same public/DP transform as the context.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_bundle(args.bundle, args.device)
    x = torch.as_tensor(np.load(args.features, allow_pickle=False), dtype=torch.float32)
    probabilities = model.predict_proba(x, args.batch_size)
    np.save(args.output, probabilities.numpy())


if __name__ == "__main__":
    main()
