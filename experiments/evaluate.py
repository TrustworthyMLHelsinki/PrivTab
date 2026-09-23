"""Evaluate PrivTab's paper checkpoints on cached public benchmark datasets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from privtab.checkpoints import load_model
from experiments.dp_hpo import (
    compute_auc, ensure_indices, load_datasets, load_repeat_indices,
    preprocess_and_split_data, select_datasets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', help='Use one checkpoint for every dataset instead of context routing.')
    parser.add_argument('--short-weights', default='models/stage2_short_contexts/weights.pt')
    parser.add_argument('--long-weights', default='models/stage2_long_contexts/weights.pt')
    parser.add_argument('--context-switch-threshold', type=int, default=4096)
    parser.add_argument('--datasets-cache', required=True)
    parser.add_argument('--indices-dir', default='runs/indices')
    parser.add_argument('--dataset')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0,
                        help='Noise seed for reproducible experiments on public benchmark data only.')
    parser.add_argument('--mu-values', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    datasets = select_datasets(load_datasets(args.datasets_cache), dataset_name=args.dataset)
    ensure_indices(datasets, args.indices_dir, args.repeats, 0.8)
    if args.context_switch_threshold <= 0:
        parser.error('--context-switch-threshold must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    models = {}
    rows = []
    with torch.inference_mode():
        for name, (x, y) in datasets:
            for repeat in range(args.repeats):
                indices = load_repeat_indices(args.indices_dir, name, repeat)
                xc, yc, xt, yt, d = preprocess_and_split_data(0.8, x, y, indices)
                context_rows = len(xc)
                weights = (args.weights or (args.short_weights if context_rows < args.context_switch_threshold
                                            else args.long_weights))
                if weights not in models:
                    models[weights] = load_model(weights, args.device)
                model = models[weights]
                xc = torch.tensor(xc, dtype=torch.float32, device=args.device).unsqueeze(0)
                yc = torch.tensor(yc, dtype=torch.long, device=args.device).unsqueeze(0)
                xt = torch.tensor(xt, dtype=torch.float32, device=args.device).unsqueeze(0)
                classes = int(np.max(y)) + 1  # Public benchmark vocabulary.
                for mu in args.mu_values:
                    summary = model.summarize(xc, yc, mu, d=d, normalize_perturbed_output=True)
                    logits = torch.cat([model.predict_from_summary(summary, part, d=d)
                                        for part in xt.split(args.batch_size, dim=1)], dim=1)[0, :, :classes]
                    probabilities = logits.softmax(-1).cpu().numpy()
                    row = dict(dataset=name, repeat=repeat, mu=mu, binary=classes == 2,
                               checkpoint=weights, context_rows=context_rows,
                               loss=float(torch.nn.functional.cross_entropy(logits, torch.tensor(yt, device=args.device))),
                               accuracy=float(np.mean(probabilities.argmax(-1) == yt)),
                               auc=float(compute_auc(yt, probabilities)))
                    rows.append(row)
                    print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
