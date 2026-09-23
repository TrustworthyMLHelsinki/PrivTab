import argparse
import csv
import math
import os
import random
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch import nn
from privtab.privacy import compute_dpsgd_sigma_for_substitute_dp_fast


BENCHMARK_MAX_FEATURES = 120
STANDARDIZATION_EPS = 1e-6
STANDARDIZATION_CLIP = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_TORCH_FUNC = hasattr(torch, "func") and all(
    hasattr(torch.func, attr) for attr in ("functional_call", "grad", "vmap")
)


@dataclass(frozen=True)
class ModelPlan:
    method_code: str
    display_name: str
    learning_rates: tuple[float, ...]
    max_grad_norm: float
    q: float
    epochs: int
    nn_width: int | None = None
    nn_layers: int | None = None


@dataclass(frozen=True)
class PrivacyBudgetSplit:
    total_mu: float
    selection_total_mu: float
    selection_train_trial_mu: float
    selection_val_trial_mu: float
    final_training_mu: float


DEFAULT_MODEL_PLANS = {
    "dp_logistic_regression": ModelPlan(
        method_code="dp_logistic_regression",
        display_name="DP Logistic Regression (Private LR Selection)",
        learning_rates=(0.005, 0.02, 0.05, 0.1),
        max_grad_norm=0.1,
        q=0.05,
        epochs=1000,
    ),
    "dp_mlp": ModelPlan(
        method_code="dp_mlp",
        display_name="DP MLP (Private LR Selection)",
        learning_rates=(3e-4, 1e-3, 3e-3, 1e-2),
        max_grad_norm=5.0,
        q=0.05,
        epochs=250,
        nn_width=64,
        nn_layers=1,
    ),
}


def _diagnostic_enabled():
    value = os.environ.get("PRIVTAB_DIAGNOSTIC", "")
    return value.lower() not in ("", "0", "false", "no")


def _diagnostic_every():
    return max(1, int(os.environ.get("PRIVTAB_DIAGNOSTIC_EVERY", "1")))


def _empty_cache_every():
    return max(0, int(os.environ.get("PRIVTAB_EMPTY_CACHE_EVERY", "0")))


def _cuda_memory_summary():
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    device = torch.cuda.current_device()
    return (
        f"alloc={torch.cuda.memory_allocated(device) / 2**20:.1f}MiB "
        f"reserved={torch.cuda.memory_reserved(device) / 2**20:.1f}MiB "
        f"max_alloc={torch.cuda.max_memory_allocated(device) / 2**20:.1f}MiB "
        f"max_reserved={torch.cuda.max_memory_reserved(device) / 2**20:.1f}MiB"
    )


def _log_diagnostic(message: str):
    if _diagnostic_enabled():
        print(f"[dp-lr-grid] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TabArena DP baselines with fixed configs and LR-only selection."
    )
    parser.add_argument("--run-name", default="tabarena-dp-lr-grid")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SCRATCH_PATH", "experiments"),
    )
    parser.add_argument(
        "--datasets-cache",
        default="experiments/datasets/tabarena_datasets.pickle",
        help="Path to a cached `load_tab_arena_datasets()` pickle.",
    )
    parser.add_argument(
        "--indices-dir",
        default=os.path.join(os.environ.get("SCRATCH_PATH", "experiments"), "indices"),
    )
    parser.add_argument("--initialize-indices", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mu-values",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DEFAULT_MODEL_PLANS),
        default=("dp_logistic_regression", "dp_mlp"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.16)
    parser.add_argument(
        "--selection-mu-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of the total mu budget reserved for the LR-selection phase in "
            "GDP root-sum-of-squares composition. The remaining GDP budget is used "
            "for the final training run."
        ),
    )
    parser.add_argument(
        "--validation-loss-clip",
        type=float,
        default=5.0,
        help="Clipping threshold for per-example validation cross-entropy losses in the DP Gaussian mechanism.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Override q for all models. Kept configurable because q was not strongly supported by the pooled HPO results.",
    )
    parser.add_argument(
        "--leaderboard-metric",
        choices=("auc", "loss"),
        default="auc",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for batched per-example gradient computation. Defaults to an adaptive GPU/CPU-aware choice.",
    )
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(cache_path: str):
    cache = Path(cache_path)
    if not cache.is_file():
        raise FileNotFoundError(
            f"Missing TabArena cache at {cache}. Create it with python -m experiments.prepare_data --input-dir DATA --output CACHE."
        )
    datasets = joblib.load(cache)
    if not isinstance(datasets, list):
        raise TypeError(f"Expected a list of datasets in {cache}, got {type(datasets).__name__}.")
    return datasets


def select_datasets(datasets, dataset_name=None, dataset_index=None):
    if dataset_name is not None and dataset_index is not None:
        raise ValueError("Use only one of --dataset and --dataset-index.")
    if dataset_name is not None:
        selected = [item for item in datasets if item[0] == dataset_name]
        if not selected:
            known = ", ".join(name for name, _ in datasets)
            raise ValueError(f"Unknown dataset {dataset_name!r}. Known datasets: {known}")
        return selected
    if dataset_index is not None:
        if dataset_index < 0 or dataset_index >= len(datasets):
            raise IndexError(f"--dataset-index must be in [0, {len(datasets) - 1}]")
        return [datasets[dataset_index]]
    return datasets


def split_indices_with_class_coverage(y, train_fraction, seed=0):
    y = np.asarray(y)
    n = len(y)
    indices = np.arange(n)
    train_size = int(round(n * train_fraction))
    classes = np.unique(y)

    if train_size < len(classes):
        raise ValueError(
            f"Train size {train_size} is smaller than number of classes {len(classes)}."
        )

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for cls in classes:
        cls_idx = indices[y == cls].copy()
        rng.shuffle(cls_idx)
        cls_train_size = int(round(len(cls_idx) * train_fraction))
        if cls_train_size <= 0:
            cls_train_size = 1
        if cls_train_size >= len(cls_idx):
            cls_train_size = len(cls_idx) - 1
        train_parts.append(cls_idx[:cls_train_size])
        test_parts.append(cls_idx[cls_train_size:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    while len(train_idx) > train_size:
        moved = train_idx[-1:]
        train_idx = train_idx[:-1]
        test_idx = np.concatenate([test_idx, moved])
    while len(train_idx) < train_size and len(test_idx) > 0:
        moved = test_idx[-1:]
        test_idx = test_idx[:-1]
        train_idx = np.concatenate([train_idx, moved])

    return np.sort(train_idx), np.sort(test_idx)


def generate_indices(y, dataset_name, path, repeats=10, train_fraction=0.8):
    os.makedirs(path, exist_ok=True)
    for repeat in range(repeats):
        train_idx, test_idx = split_indices_with_class_coverage(
            y,
            train_fraction,
            seed=repeat,
        )
        indices = np.concatenate([train_idx, test_idx])
        joblib.dump(indices, filename=os.path.join(path, f"{dataset_name}_repeat_{repeat}.pickle"))


def ensure_indices(datasets, indices_dir: str, repeats: int, train_fraction: float):
    os.makedirs(indices_dir, exist_ok=True)
    for dataset_name, (_, y) in datasets:
        for repeat in range(repeats):
            path = os.path.join(indices_dir, f"{dataset_name}_repeat_{repeat}.pickle")
            if os.path.isfile(path):
                indices = np.asarray(joblib.load(path))
                if indices.shape != (len(y),) or not np.array_equal(np.sort(indices), np.arange(len(y))):
                    raise ValueError(f"Invalid saved split indices: {path}")
                continue
            train_idx, test_idx = split_indices_with_class_coverage(y, train_fraction, seed=repeat)
            joblib.dump(np.concatenate([train_idx, test_idx]), filename=path)


def load_repeat_indices(indices_dir: str, dataset_name: str, repeat: int):
    path = os.path.join(indices_dir, f"{dataset_name}_repeat_{repeat}.pickle")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing split indices file: {path}")
    return joblib.load(path)


def preprocess_and_split_data(train_fraction, x, y, indices=None):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if indices is None:
        train_idx, test_idx = split_indices_with_class_coverage(y, train_fraction, seed=0)
        train_x = x[train_idx]
        train_y = y[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
    else:
        shuffled_x = x[indices]
        shuffled_y = y[indices]
        train_len = int(round(x.shape[0] * train_fraction))
        train_x = shuffled_x[:train_len]
        train_y = shuffled_y[:train_len]
        test_x = shuffled_x[train_len:]
        test_y = shuffled_y[train_len:]

    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0)
    safe_std = train_std.copy()
    constant_mask = safe_std < STANDARDIZATION_EPS
    safe_std[constant_mask] = 1.0

    train_x = (train_x - train_mean) / safe_std
    test_x = (test_x - train_mean) / safe_std
    train_x[:, constant_mask] = 0.0
    test_x[:, constant_mask] = 0.0
    train_x = np.clip(train_x, -STANDARDIZATION_CLIP, STANDARDIZATION_CLIP)
    test_x = np.clip(test_x, -STANDARDIZATION_CLIP, STANDARDIZATION_CLIP)

    train_features_count = train_x.shape[1]
    if train_features_count > BENCHMARK_MAX_FEATURES:
        raise ValueError(
            f"Dataset has {train_features_count} features, exceeding BENCHMARK_MAX_FEATURES={BENCHMARK_MAX_FEATURES}."
        )

    pad_width = BENCHMARK_MAX_FEATURES - train_features_count
    train_x = np.concatenate([train_x, np.zeros((train_x.shape[0], pad_width), dtype=np.float32)], axis=-1)
    test_x = np.concatenate([test_x, np.zeros((test_x.shape[0], pad_width), dtype=np.float32)], axis=-1)
    return train_x, train_y, test_x, test_y, train_features_count


def split_train_validation(train_x, train_y, val_fraction: float, seed: int):
    # Fixed-size, label-independent partition: changing one record must not
    # move other records between the two parallel-composition branches.
    if not 0 < val_fraction < 1 or len(train_y) < 2:
        raise ValueError("Validation needs at least two rows and a fraction in (0, 1).")
    indices = np.random.default_rng(seed).permutation(len(train_y))
    val_size = min(len(train_y) - 1, max(1, round(len(train_y) * val_fraction)))
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return (
        train_x[train_idx],
        train_y[train_idx],
        train_x[val_idx],
        train_y[val_idx],
    )


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, n_layers: int, num_classes: int):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def make_model(model_plan: ModelPlan, input_dim: int, num_classes: int):
    if model_plan.method_code == "dp_logistic_regression":
        return SimpleLogisticRegression(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    if model_plan.method_code == "dp_mlp":
        return SimpleMLP(
            input_dim=input_dim,
            width=int(model_plan.nn_width),
            n_layers=int(model_plan.nn_layers),
            num_classes=num_classes,
        ).to(DEVICE)
    raise ValueError(f"Unsupported model code: {model_plan.method_code}")


def resolve_poisson_schedule(q: float):
    if not 0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}.")
    if q == 1.0:
        return 1.0, 1
    steps_per_epoch = max(1, int(round(1.0 / q)))
    effective_q = 1.0 / steps_per_epoch
    return effective_q, steps_per_epoch


def resolve_chunk_size(x: torch.Tensor, chunk_size: int | None):
    if chunk_size is not None:
        return max(1, min(int(chunk_size), x.shape[0]))

    env_chunk_size = os.environ.get("PRIVTAB_VMAP_CHUNK_SIZE")
    if env_chunk_size is not None:
        return max(1, min(int(env_chunk_size), x.shape[0]))

    if x.is_cuda:
        return min(128, x.shape[0])
    return min(256, x.shape[0])


def exact_noise_std(target_mu: float, max_grad_norm: float, q: float, compositions: int):
    if not math.isfinite(target_mu) or target_mu <= 0 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("mu and gradient clipping norm must be finite and positive.")
    if not 0 < q <= 1 or compositions < 1:
        raise ValueError("Expected a sampling probability in (0, 1] and positive composition count.")
    # Google's REPLACE_ONE PLD models opposing contributions at +/- sensitivity,
    # so passing the clipping radius already accounts for the 2C difference.
    sigma, _, _ = compute_dpsgd_sigma_for_substitute_dp_fast(
        target_mu=target_mu,
        max_grad_norm=max_grad_norm,
        q=q,
        compositions=compositions,
    )
    return sigma


def gaussian_mu_noise_std(sensitivity: float, target_mu: float):
    if not math.isfinite(target_mu) or target_mu <= 0.0:
        raise ValueError(f"target_mu must be positive, got {target_mu}.")
    if not math.isfinite(sensitivity) or sensitivity <= 0.0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}.")
    return sensitivity / target_mu


def allocate_privacy_budget(total_mu: float, selection_mu_fraction: float, num_trials: int) -> PrivacyBudgetSplit:
    if not math.isfinite(total_mu) or total_mu <= 0.0:
        raise ValueError(f"total_mu must be positive, got {total_mu}.")
    if not 0.0 < selection_mu_fraction < 1.0:
        raise ValueError(
            f"selection_mu_fraction must lie strictly in (0, 1), got {selection_mu_fraction}."
        )
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}.")

    selection_total_mu = total_mu * selection_mu_fraction
    final_training_mu_sq = max(0.0, total_mu ** 2 - selection_total_mu ** 2)
    final_training_mu = math.sqrt(final_training_mu_sq)
    selection_trial_mu = selection_total_mu / math.sqrt(num_trials)
    return PrivacyBudgetSplit(
        total_mu=total_mu,
        selection_total_mu=selection_total_mu,
        selection_train_trial_mu=selection_trial_mu,
        selection_val_trial_mu=selection_trial_mu,
        final_training_mu=final_training_mu,
    )


def build_per_example_grad_components(model: nn.Module):
    cached = getattr(model, "_per_example_grad_components", None)
    if cached is not None:
        return cached

    named_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    if not named_params:
        raise ValueError("Model has no trainable parameters.")
    buffers = dict(model.named_buffers())
    param_names = tuple(named_params.keys())
    param_list = [named_params[name] for name in param_names]

    def single_example_loss(params, model_buffers, sample, target):
        logits = torch.func.functional_call(
            model,
            (params, model_buffers),
            (sample.unsqueeze(0),),
        )
        return F.cross_entropy(logits, target.unsqueeze(0), reduction="sum")

    per_example_grad_fn = torch.func.vmap(
        torch.func.grad(single_example_loss),
        in_dims=(None, None, 0, 0),
    )
    cached = (named_params, buffers, param_names, param_list, per_example_grad_fn)
    model._per_example_grad_components = cached
    return cached


def compute_private_gradients_slow(
    model,
    x,
    y,
    clip_norm: float,
    noise_std: float,
    normalization: float,
):
    params = [param for param in model.parameters() if param.requires_grad]
    clipped_sums = [torch.zeros_like(param) for param in params]

    for idx in range(x.shape[0]):
        x_i = x[idx: idx + 1]
        y_i = y[idx: idx + 1]
        logits = model(x_i)
        loss = F.cross_entropy(logits, y_i, reduction="sum")
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=False)
        grad_norm_sq = torch.zeros((), device=x.device, dtype=x.dtype)
        for grad in grads:
            grad_norm_sq = grad_norm_sq + grad.detach().pow(2).sum()
        grad_norm = torch.sqrt(torch.clamp(grad_norm_sq, min=1e-12))
        clip_factor = min(1.0, float(clip_norm / grad_norm))
        for acc, grad in zip(clipped_sums, grads):
            acc.add_(grad.detach(), alpha=clip_factor)

    normalization = max(1.0, float(normalization))
    noisy_grads = []
    for clipped_sum in clipped_sums:
        noisy_sum = clipped_sum + noise_std * torch.randn_like(clipped_sum)
        noisy_grads.append(noisy_sum / normalization)
    return noisy_grads


def compute_private_gradients_batched(
    model,
    components,
    x,
    y,
    clip_norm: float,
    noise_std: float,
    normalization: float,
    chunk_size: int,
):
    named_params, buffers, param_names, param_list, per_example_grad_fn = components
    clipped_sums = [torch.zeros_like(param) for param in param_list]

    for start_idx in range(0, x.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, x.shape[0])
        x_chunk = x[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        per_example_grads = per_example_grad_fn(
            named_params,
            buffers,
            x_chunk,
            y_chunk,
        )

        grad_norm_sq = torch.zeros(x_chunk.shape[0], device=x.device, dtype=x.dtype)
        for name in param_names:
            grad = per_example_grads[name].detach()
            grad_norm_sq.add_(grad.reshape(grad.shape[0], -1).pow(2).sum(dim=1))
        grad_norm = torch.sqrt(torch.clamp(grad_norm_sq, min=1e-12))
        clip_factors = torch.clamp(clip_norm / grad_norm, max=1.0)

        for acc, name in zip(clipped_sums, param_names):
            grad = per_example_grads[name].detach()
            reshape_dims = (grad.shape[0],) + (1,) * (grad.ndim - 1)
            acc.add_((grad * clip_factors.view(reshape_dims)).sum(dim=0))
        del per_example_grads, grad_norm_sq, grad_norm, clip_factors, x_chunk, y_chunk

    normalization = max(1.0, float(normalization))
    noisy_grads = []
    for clipped_sum in clipped_sums:
        noisy_sum = clipped_sum + noise_std * torch.randn_like(clipped_sum)
        noisy_grads.append(noisy_sum / normalization)
    return noisy_grads


def compute_private_gradients_batched_resilient(
    model,
    components,
    x,
    y,
    clip_norm: float,
    noise_std: float,
    normalization: float,
    chunk_size: int | None,
):
    chunk_size = resolve_chunk_size(x, chunk_size)

    while True:
        try:
            return compute_private_gradients_batched(
                model=model,
                components=components,
                x=x,
                y=y,
                clip_norm=clip_norm,
                noise_std=noise_std,
                normalization=normalization,
                chunk_size=chunk_size,
            )
        except torch.OutOfMemoryError:
            if not x.is_cuda:
                raise
            if chunk_size <= 1:
                _log_diagnostic(
                    "CUDA OOM in batched per-example gradients even at chunk_size=1; "
                    f"falling back to slow path. {_cuda_memory_summary()}"
                )
                torch.cuda.empty_cache()
                return compute_private_gradients_slow(
                    model=model,
                    x=x,
                    y=y,
                    clip_norm=clip_norm,
                    noise_std=noise_std,
                    normalization=normalization,
                )
            next_chunk_size = max(1, chunk_size // 2)
            _log_diagnostic(
                f"CUDA OOM in batched per-example gradients at chunk_size={chunk_size}; "
                f"retrying with chunk_size={next_chunk_size}. {_cuda_memory_summary()}"
            )
            torch.cuda.empty_cache()
            chunk_size = next_chunk_size


def compute_private_gradients(model, components, x, y, clip_norm, noise_std, normalization, chunk_size):
    if HAS_TORCH_FUNC:
        return compute_private_gradients_batched_resilient(
            model=model,
            components=components,
            x=x,
            y=y,
            clip_norm=clip_norm,
            noise_std=noise_std,
            normalization=normalization,
            chunk_size=chunk_size,
        )
    return compute_private_gradients_slow(
        model=model,
        x=x,
        y=y,
        clip_norm=clip_norm,
        noise_std=noise_std,
        normalization=normalization,
    )


def sample_poisson_batch(x, y, q):
    if q >= 1.0:
        return x, y
    mask = torch.rand(x.shape[0], device=x.device) < q
    if not torch.any(mask):
        return None, None
    return x[mask], y[mask]


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    prob_chunks = []

    with torch.no_grad():
        batch_size = min(1024, x.shape[0])
        for start_idx in range(0, x.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, x.shape[0])
            x_batch = x[start_idx:end_idx].to(DEVICE)
            y_batch = y[start_idx:end_idx].to(DEVICE)
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch, reduction="mean")
            total_loss += loss.item() * x_batch.shape[0]
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            total_correct += (preds == y_batch).sum().item()
            total_examples += x_batch.shape[0]
            prob_chunks.append(probs.detach().cpu())

    avg_loss = total_loss / max(1, total_examples)
    acc = total_correct / max(1, total_examples)
    probs = torch.cat(prob_chunks, dim=0).numpy()
    return avg_loss, acc, probs


def compute_auc(y_true: np.ndarray, probs: np.ndarray):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    num_classes = probs.shape[1]
    try:
        if num_classes == 2:
            if np.unique(y_true).size < 2:
                return float("nan")
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probs,
                multi_class="ovr",
                average="macro",
                labels=np.arange(num_classes),
            )
        )
    except ValueError:
        return float("nan")


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        raise ValueError("Binary AUC is undefined when only one class is present.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = np.sum(ranks[y_true == 1])
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def roc_auc_multiclass_ovr_macro(y_true: np.ndarray, probs: np.ndarray):
    classes = np.unique(y_true)
    aucs = []
    for cls in classes:
        binary_true = (y_true == cls).astype(int)
        aucs.append(roc_auc_binary(binary_true, probs[:, int(cls)]))
    return float(np.mean(aucs))


def train_private_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mu: float,
    model_plan: ModelPlan,
    learning_rate: float,
    seed: int,
    chunk_size: int,
    context_label: str,
    num_classes: int,
):
    # Public benchmark seeds control splits, never the DP mechanism randomness.
    set_global_seed(secrets.randbits(32))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    input_dim = train_x.shape[-1]
    if num_classes < 2 or (train_y < 0).any() or (train_y >= num_classes).any():
        raise ValueError("Labels must belong to the supplied public class vocabulary.")
    model = make_model(model_plan, input_dim=input_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    effective_q, steps_per_epoch = resolve_poisson_schedule(model_plan.q)
    compositions = model_plan.epochs * steps_per_epoch
    expected_batch_size = max(1, int(train_x.shape[0] * effective_q))
    noise_std = exact_noise_std(
        target_mu=mu,
        max_grad_norm=model_plan.max_grad_norm,
        q=effective_q,
        compositions=compositions,
    )
    components = build_per_example_grad_components(model) if HAS_TORCH_FUNC else None

    step = 0
    diagnostic_every = _diagnostic_every()
    empty_cache_every = _empty_cache_every()
    _log_diagnostic(
        f"start model={model_plan.method_code} mu={mu} lr={learning_rate:.6g} "
        f"train_shape={tuple(train_x.shape)} "
        f"chunk_size={chunk_size} mem={_cuda_memory_summary()}"
    )
    print(
        f"[train] {context_label}: model={model_plan.method_code} "
        f"lr={learning_rate:.6g} mu={mu} epochs={model_plan.epochs} "
        f"effective_q={effective_q:.6f} steps_per_epoch={steps_per_epoch} "
        f"compositions={compositions} expected_batch_size={expected_batch_size} "
        f"sigma={noise_std:.6f}",
        flush=True,
    )

    for epoch in range(model_plan.epochs):
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == model_plan.epochs:
            print(
                f"[train] {context_label}: epoch {epoch + 1}/{model_plan.epochs}",
                flush=True,
            )
        for _ in range(steps_per_epoch):
            step += 1
            x_batch, y_batch = sample_poisson_batch(train_x, train_y, effective_q)
            if x_batch is None or x_batch.shape[0] == 0:
                # Empty Poisson samples still receive a Gaussian update.
                optimizer.zero_grad(set_to_none=True)
                for parameter in model.parameters():
                    parameter.grad = torch.randn_like(parameter) * noise_std / expected_batch_size
                optimizer.step()
                continue
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            if step == 1 or step % diagnostic_every == 0:
                _log_diagnostic(
                    f"step={step} epoch={epoch + 1} batch_shape={tuple(x_batch.shape)} "
                    f"before_grad mem={_cuda_memory_summary()}"
                )
            grads = compute_private_gradients(
                model=model,
                components=components,
                x=x_batch,
                y=y_batch,
                clip_norm=model_plan.max_grad_norm,
                noise_std=noise_std,
                normalization=expected_batch_size,
                chunk_size=chunk_size,
            )
            if step == 1 or step % diagnostic_every == 0:
                _log_diagnostic(
                    f"step={step} epoch={epoch + 1} after_grad mem={_cuda_memory_summary()}"
                )
            optimizer.zero_grad(set_to_none=True)
            grad_iter = iter(grads)
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                param.grad = next(grad_iter)
            optimizer.step()
            del grads, grad_iter, x_batch, y_batch
            if torch.cuda.is_available() and empty_cache_every > 0 and step % empty_cache_every == 0:
                torch.cuda.empty_cache()

    return model


def fit_private_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    mu: float,
    model_plan: ModelPlan,
    learning_rate: float,
    seed: int,
    chunk_size: int,
    context_label: str,
    num_classes: int,
):
    model = train_private_model(
        train_x=train_x,
        train_y=train_y,
        mu=mu,
        model_plan=model_plan,
        learning_rate=learning_rate,
        seed=seed,
        chunk_size=chunk_size,
        context_label=context_label,
        num_classes=num_classes,
    )
    _log_diagnostic(f"before_evaluate mem={_cuda_memory_summary()}")
    result = evaluate_model(model, eval_x, eval_y)
    _log_diagnostic(f"after_evaluate mem={_cuda_memory_summary()}")
    print(
        f"[train] {context_label}: done eval_loss={result[0]:.4f} eval_acc={result[1]:.4f}",
        flush=True,
    )
    return result


def evaluate_model_dp_noisy_validation_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    target_mu: float,
    loss_clip: float,
    seed: int,
):
    if not math.isfinite(loss_clip) or loss_clip <= 0.0:
        raise ValueError(f"loss_clip must be positive, got {loss_clip}.")

    model.eval()
    clipped_loss_sum = 0.0
    total_examples = int(x.shape[0])
    if total_examples == 0:
        raise ValueError("Validation set must not be empty.")

    with torch.no_grad():
        batch_size = min(1024, total_examples)
        for start_idx in range(0, total_examples, batch_size):
            end_idx = min(start_idx + batch_size, total_examples)
            x_batch = x[start_idx:end_idx].to(DEVICE)
            y_batch = y[start_idx:end_idx].to(DEVICE)
            logits = model(x_batch)
            losses = F.cross_entropy(logits, y_batch, reduction="none")
            clipped_loss_sum += float(torch.clamp(losses, max=loss_clip).sum().item())

    noise_std = gaussian_mu_noise_std(loss_clip, target_mu)
    rng = np.random.default_rng()
    noisy_loss_sum = clipped_loss_sum + float(rng.normal(loc=0.0, scale=noise_std))
    return noisy_loss_sum / max(1, total_examples)


def select_learning_rate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    privacy_budget: PrivacyBudgetSplit,
    model_plan: ModelPlan,
    val_fraction: float,
    validation_loss_clip: float,
    seed: int,
    chunk_size: int,
    context_label: str,
    num_classes: int,
):
    train_part_x, train_part_y, val_x, val_y = split_train_validation(
        train_x,
        train_y,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_part_x_tensor = torch.tensor(train_part_x, dtype=torch.float32)
    train_part_y_tensor = torch.tensor(train_part_y, dtype=torch.long)
    val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long)

    trial_rows = []
    print(
        f"[select] {context_label}: trying {len(model_plan.learning_rates)} learning rates "
        f"{list(model_plan.learning_rates)} "
        f"with selection_total_mu={privacy_budget.selection_total_mu:.6g} "
        f"selection_train_trial_mu={privacy_budget.selection_train_trial_mu:.6g} "
        f"selection_val_trial_mu={privacy_budget.selection_val_trial_mu:.6g}",
        flush=True,
    )
    for lr_idx, learning_rate in enumerate(model_plan.learning_rates, start=1):
        print(
            f"[select] {context_label}: trial {lr_idx}/{len(model_plan.learning_rates)} "
            f"lr={learning_rate:.6g}",
            flush=True,
        )
        model = train_private_model(
            train_x=train_part_x_tensor,
            train_y=train_part_y_tensor,
            mu=privacy_budget.selection_train_trial_mu,
            model_plan=model_plan,
            learning_rate=learning_rate,
            seed=seed + lr_idx,
            chunk_size=chunk_size,
            context_label=f"{context_label} [selection lr={learning_rate:.6g}]",
            num_classes=num_classes,
        )
        dp_val_loss = evaluate_model_dp_noisy_validation_loss(
            model=model,
            x=val_x_tensor,
            y=val_y_tensor,
            target_mu=privacy_budget.selection_val_trial_mu,
            loss_clip=validation_loss_clip,
            seed=seed + 100_000 + lr_idx,
        )
        trial_rows.append(
            {
                "learning_rate": learning_rate,
                "dp_val_loss": dp_val_loss,
                "selection_train_mu": privacy_budget.selection_train_trial_mu,
                "selection_val_mu": privacy_budget.selection_val_trial_mu,
            }
        )
        print(
            f"[select] {context_label}: lr={learning_rate:.6g} "
            f"dp_val_loss={dp_val_loss:.4f}",
            flush=True,
        )

    best_trial = min(trial_rows, key=lambda row: row["dp_val_loss"])
    print(
        f"[select] {context_label}: chose lr={best_trial['learning_rate']:.6g} "
        f"with dp_val_loss={best_trial['dp_val_loss']:.4f}",
        flush=True,
    )
    return best_trial, trial_rows


def empty_metrics_dict():
    return {
        "accs": [],
        "probs": [],
        "aucs": [],
        "losses": [],
        "selected_params": [],
    }


def append_result(metrics_dict, loss, acc, auc, probs, selected_params):
    metrics_dict["accs"].append(lossless_float(acc))
    metrics_dict["probs"].append(probs)
    metrics_dict["aucs"].append(lossless_float(auc))
    metrics_dict["losses"].append(lossless_float(loss))
    metrics_dict["selected_params"].append(selected_params)


def lossless_float(value):
    return float(value) if np.isfinite(value) else float("nan")


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0


def format_metric(values, scale=1.0, precision=2):
    mean_value, std_value = mean_std(values)
    if not np.isfinite(mean_value):
        return "N/A"
    mean_value *= scale
    std_value *= scale
    return f"${mean_value:.{precision}f} \\pm {std_value:.{precision}f}$"


def build_latex_table(results_dict, dataset_name, mu_values_ordered, method_codes, method_name_map):
    metric_specs = [
        ("ACC", "accs", 100.0, 2),
        ("AUC", "aucs", 100.0, 2),
        ("LogLoss", "losses", 1.0, 4),
    ]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{l|" + "c" * len(mu_values_ordered) + "}")
    lines.append("        \\hline")
    header = "        Method " + "".join([f"& $\\mu = {mu}$ " for mu in mu_values_ordered]) + "\\\\"
    lines.append(header)
    lines.append("        \\hline")

    for method_code in method_codes:
        metrics_by_mu = results_dict[dataset_name].get(method_code, {})
        method_name = method_name_map[method_code]
        for metric_label, metric_key, scale, precision in metric_specs:
            row = f"        {method_name} ({metric_label}) "
            for mu in mu_values_ordered:
                metric_values = metrics_by_mu.get(mu, {}).get(metric_key, [])
                row += "& " + format_metric(metric_values, scale=scale, precision=precision) + " "
            row += "\\\\"
            lines.append(row)
        lines.append("        \\hline")

    lines.append("    \\end{tabular}")
    lines.append(f"    \\caption{{TabArena DP baselines on {dataset_name}.}}")
    lines.append(f"    \\label{{tab:{dataset_name.replace(' ', '_').replace('-', '').lower()}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(raw_rows, group_keys):
    grouped = {}
    for row in raw_rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, rows in grouped.items():
        summary = {key_name: key[idx] for idx, key_name in enumerate(group_keys)}
        summary["n"] = len(rows)
        for metric in ("metric_error", "loss", "acc", "auc"):
            values = [row[metric] for row in rows]
            mean_value, std_value = mean_std(values)
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_std"] = std_value
        summary_rows.append(summary)

    return sorted(summary_rows, key=lambda row: tuple(row[key] for key in group_keys))


def main():
    args = parse_args()
    set_global_seed(args.seed)

    datasets = load_datasets(args.datasets_cache)

    if args.list_datasets:
        for idx, (dataset_name, (x, y)) in enumerate(datasets):
            print(
                f"[{idx}] {dataset_name}: samples={x.shape[0]} features={x.shape[1]} classes={len(np.unique(y))}"
            )
        return

    selected_datasets = select_datasets(
        datasets,
        dataset_name=args.dataset,
        dataset_index=args.dataset_index,
    )

    if args.initialize_indices:
        ensure_indices(
            selected_datasets,
            args.indices_dir,
            repeats=args.repeats,
            train_fraction=args.train_fraction,
        )
        print(f"Saved split indices to {args.indices_dir}")
        return

    ensure_indices(
        selected_datasets,
        args.indices_dir,
        repeats=args.repeats,
        train_fraction=args.train_fraction,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.run_name}-all-results.pickle"
    raw_csv_path = output_dir / f"{args.run_name}-raw.csv"
    summary_csv_path = output_dir / f"{args.run_name}-summary.csv"
    dataset_summary_csv_path = output_dir / f"{args.run_name}-dataset-summary.csv"
    latex_path = output_dir / f"{args.run_name}-latex-tables.tex"

    repeats_to_run = [args.repeat] if args.repeat is not None else list(range(args.repeats))
    method_name_map = {
        model_code: DEFAULT_MODEL_PLANS[model_code].display_name
        for model_code in args.models
    }

    results_dict = {}
    raw_rows = []
    total_datasets = len(selected_datasets)
    total_repeats = len(repeats_to_run)
    total_mus = len(args.mu_values)
    total_models = len(args.models)

    for dataset_idx, (dataset_name, (x, y)) in enumerate(selected_datasets, start=1):
        results_dict.setdefault(dataset_name, {"binary": len(np.unique(y)) == 2})
        print(
            f"[run] dataset {dataset_idx}/{total_datasets}: {dataset_name} "
            f"samples={x.shape[0]} features={x.shape[1]} classes={len(np.unique(y))}",
            flush=True,
        )

        for model_code in args.models:
            results_dict[dataset_name].setdefault(model_code, {})

        for repeat_idx, repeat in enumerate(repeats_to_run, start=1):
            print(
                f"[run] dataset={dataset_name}: repeat {repeat_idx}/{total_repeats} "
                f"(repeat_id={repeat})",
                flush=True,
            )
            indices = load_repeat_indices(args.indices_dir, dataset_name, repeat)
            train_x, train_y, test_x, test_y, _ = preprocess_and_split_data(
                args.train_fraction,
                x,
                y,
                indices=indices,
            )

            train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
            train_y_tensor = torch.tensor(train_y, dtype=torch.long)
            test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
            test_y_tensor = torch.tensor(test_y, dtype=torch.long)

            for mu_idx, mu in enumerate(args.mu_values, start=1):
                print(
                    f"[run] dataset={dataset_name} repeat={repeat}: mu {mu_idx}/{total_mus} "
                    f"(mu={mu})",
                    flush=True,
                )
                for model_idx, model_code in enumerate(args.models, start=1):
                    base_plan = DEFAULT_MODEL_PLANS[model_code]
                    plan = base_plan if args.q is None else ModelPlan(
                        method_code=base_plan.method_code,
                        display_name=base_plan.display_name,
                        learning_rates=base_plan.learning_rates,
                        max_grad_norm=base_plan.max_grad_norm,
                        q=args.q,
                        epochs=base_plan.epochs,
                        nn_width=base_plan.nn_width,
                        nn_layers=base_plan.nn_layers,
                    )
                    privacy_budget = allocate_privacy_budget(
                        total_mu=mu,
                        selection_mu_fraction=args.selection_mu_fraction,
                        num_trials=len(plan.learning_rates),
                    )

                    context_label = (
                        f"dataset={dataset_name} repeat={repeat} mu={mu} "
                        f"model={model_code} ({model_idx}/{total_models})"
                    )
                    print(
                        f"[privacy] {context_label}: total_mu={privacy_budget.total_mu:.6g} "
                        f"selection_total_mu={privacy_budget.selection_total_mu:.6g} "
                        f"selection_train_trial_mu={privacy_budget.selection_train_trial_mu:.6g} "
                        f"selection_val_trial_mu={privacy_budget.selection_val_trial_mu:.6g} "
                        f"final_training_mu={privacy_budget.final_training_mu:.6g}",
                        flush=True,
                    )
                    print(f"[run] {context_label}: starting LR selection", flush=True)
                    best_trial, selection_trials = select_learning_rate(
                        train_x=train_x,
                        train_y=train_y,
                        privacy_budget=privacy_budget,
                        model_plan=plan,
                        val_fraction=args.val_fraction,
                        validation_loss_clip=args.validation_loss_clip,
                        seed=args.seed + repeat,
                        chunk_size=args.chunk_size,
                        context_label=context_label,
                        num_classes=int(np.max(y)) + 1,
                    )

                    print(
                        f"[run] {context_label}: retraining on full train split with "
                        f"lr={best_trial['learning_rate']:.6g}",
                        flush=True,
                    )
                    test_loss, test_acc, test_probs = fit_private_model(
                        train_x=train_x_tensor,
                        train_y=train_y_tensor,
                        eval_x=test_x_tensor,
                        eval_y=test_y_tensor,
                        mu=privacy_budget.final_training_mu,
                        model_plan=plan,
                        learning_rate=best_trial["learning_rate"],
                        seed=args.seed + 10_000 + repeat,
                        chunk_size=args.chunk_size,
                        context_label=f"{context_label} [final]",
                        num_classes=int(np.max(y)) + 1,
                    )
                    test_auc = compute_auc(test_y, test_probs)

                    metrics_dict = results_dict[dataset_name][model_code].setdefault(mu, empty_metrics_dict())
                    selected_params = {
                        "learning_rate": best_trial["learning_rate"],
                        "max_grad_norm": plan.max_grad_norm,
                        "q": plan.q,
                        "epochs": plan.epochs,
                        "nn_width": plan.nn_width,
                        "nn_layers": plan.nn_layers,
                        "selection_metric": "dp_val_loss",
                        "total_mu": privacy_budget.total_mu,
                        "selection_total_mu": privacy_budget.selection_total_mu,
                        "selection_train_trial_mu": privacy_budget.selection_train_trial_mu,
                        "selection_val_trial_mu": privacy_budget.selection_val_trial_mu,
                        "final_training_mu": privacy_budget.final_training_mu,
                        "validation_loss_clip": args.validation_loss_clip,
                        "selection_trials": selection_trials,
                    }
                    append_result(
                        metrics_dict,
                        loss=test_loss,
                        acc=test_acc,
                        auc=test_auc,
                        probs=test_probs,
                        selected_params=selected_params,
                    )

                    metrics_dict.setdefault("repeat_ids", []).append(repeat)
                    metric_error = 1.0 - test_auc if args.leaderboard_metric == "auc" and np.isfinite(test_auc) else test_loss
                    raw_rows.append(
                        {
                            "dataset": dataset_name,
                            "repeat": repeat,
                            "mu": mu,
                            "method": plan.display_name,
                            "method_code": model_code,
                            "task": f"{dataset_name}_repeat{repeat}_mu{mu}",
                            "metric_error": metric_error,
                            "loss": lossless_float(test_loss),
                            "acc": lossless_float(test_acc),
                            "auc": lossless_float(test_auc),
                            "learning_rate": best_trial["learning_rate"],
                            "max_grad_norm": plan.max_grad_norm,
                            "q": plan.q,
                            "epochs": plan.epochs,
                            "nn_width": plan.nn_width,
                            "nn_layers": plan.nn_layers,
                        }
                    )

                    print(
                        f"[done] {context_label}: lr={best_trial['learning_rate']:.6g} "
                        f"acc={test_acc:.4f} auc={test_auc:.4f} loss={test_loss:.4f}",
                        flush=True,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    with results_path.open("wb") as f:
        joblib.dump(results_dict, f)

    raw_fieldnames = [
        "dataset",
        "repeat",
        "mu",
        "method",
        "method_code",
        "task",
        "metric_error",
        "loss",
        "acc",
        "auc",
        "learning_rate",
        "max_grad_norm",
        "q",
        "epochs",
        "nn_width",
        "nn_layers",
    ]
    write_csv(raw_csv_path, raw_rows, raw_fieldnames)

    summary_rows = build_summary_rows(raw_rows, group_keys=("method", "method_code", "mu"))
    summary_fieldnames = [
        "method",
        "method_code",
        "mu",
        "n",
        "metric_error_mean",
        "metric_error_std",
        "loss_mean",
        "loss_std",
        "acc_mean",
        "acc_std",
        "auc_mean",
        "auc_std",
    ]
    write_csv(summary_csv_path, summary_rows, summary_fieldnames)

    dataset_summary_rows = build_summary_rows(raw_rows, group_keys=("dataset", "method", "method_code", "mu"))
    dataset_summary_fieldnames = [
        "dataset",
        "method",
        "method_code",
        "mu",
        "n",
        "metric_error_mean",
        "metric_error_std",
        "loss_mean",
        "loss_std",
        "acc_mean",
        "acc_std",
        "auc_mean",
        "auc_std",
    ]
    write_csv(dataset_summary_csv_path, dataset_summary_rows, dataset_summary_fieldnames)

    with latex_path.open("w") as f:
        for dataset_name in results_dict:
            f.write(
                build_latex_table(
                    results_dict=results_dict,
                    dataset_name=dataset_name,
                    mu_values_ordered=args.mu_values,
                    method_codes=args.models,
                    method_name_map=method_name_map,
                )
            )

    print(f"Saved results pickle to {results_path}")
    print(f"Saved raw rows CSV to {raw_csv_path}")
    print(f"Saved overall summary CSV to {summary_csv_path}")
    print(f"Saved per-dataset summary CSV to {dataset_summary_csv_path}")
    print(f"Saved LaTeX tables to {latex_path}")


if __name__ == "__main__":
    main()
