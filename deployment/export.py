"""Export a private summary and predictor from prepared context arrays."""
import argparse

import numpy as np
import torch

from privtab.checkpoints import load_model
from .bundle import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--context", required=True, help="NPZ with preprocessed x and integer y arrays.")
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--num-classes", type=int, required=True, help="Public class vocabulary size (2–10).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = np.load(args.context, allow_pickle=False)
    export_bundle(load_model(args.weights, args.device), torch.as_tensor(data["x"], dtype=torch.float32),
                  torch.as_tensor(data["y"]), args.mu, args.num_classes, args.output)
    print(f"Saved one mu={args.mu:g} private release to {args.output}")


if __name__ == "__main__":
    main()
