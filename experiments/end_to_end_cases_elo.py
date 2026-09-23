"""Rank the end-to-end DP methods with Elo scores.

Binary datasets use ``1 - AUC`` as their error and multiclass datasets use
cross-entropy (log loss).  Thus, lower values always win an Elo battle.
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from bencheval.tabarena import TabArena
except ModuleNotFoundError:
    from experiments.bencheval.tabarena import TabArena


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVTAB_RESULTS = REPO_ROOT / "runs/end_to_end_cases_privtab.pickle"
DEFAULT_BASELINES_DIR = REPO_ROOT / "runs/end_to_end_dp_baselines"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/end_to_end_cases_elo.csv"

METHOD_CODES = {
    "PrivTab": "privtab_dp_llm_bounds",
    "DP-LR": "dp_logistic_regression_dp_llm_bounds_dp_zscore",
    "DP-MLP": "dp_mlp_dp_llm_bounds_dp_zscore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Elo scores for PrivTab, DP-LR, and DP-MLP on the "
            "end-to-end case-study datasets."
        )
    )
    parser.add_argument("--privtab-results", type=Path, default=DEFAULT_PRIVTAB_RESULTS)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mu",
        type=float,
        default=0.4,
        help="GDP mu to rank (default: 0.4).",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=10_000,
        help="Number of task-bootstrap rounds used for the 95%% CI.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def find_mu_key(results_by_mu: dict[Any, Any], mu: float) -> Any | None:
    """Return the stored key corresponding to ``mu``, allowing float roundoff."""
    for candidate in results_by_mu:
        try:
            if np.isclose(float(candidate), mu, rtol=0.0, atol=1e-8):
                return candidate
        except (TypeError, ValueError):
            continue
    return None


def load_baseline_results(baselines_dir: Path) -> dict[str, dict[str, Any]]:
    pattern = str(baselines_dir / "*-all-results.pickle")
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No baseline result pickles matched {pattern}")

    merged: dict[str, dict[str, Any]] = {}
    wanted_codes = set(METHOD_CODES.values()) - {METHOD_CODES["PrivTab"]}
    for path in paths:
        loaded = joblib.load(path)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected a results dict in {path}, got {type(loaded).__name__}")
        for dataset, dataset_results in loaded.items():
            if not isinstance(dataset_results, dict):
                continue
            for method_code in wanted_codes.intersection(dataset_results):
                target = merged.setdefault(dataset, {})
                if method_code in target:
                    raise ValueError(
                        f"Duplicate results for {dataset!r}, {method_code!r}; "
                        f"the second occurrence is in {path}"
                    )
                target[method_code] = dataset_results[method_code]
    return merged


def metric_errors(metrics: dict[str, Any], is_binary: bool) -> np.ndarray:
    if is_binary:
        values = 1.0 - np.asarray(metrics["aucs"], dtype=float)
    else:
        values = np.asarray(metrics["losses"], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty one-dimensional metric sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("Metric sequence contains NaN or infinity")
    return values


def build_battles(
    privtab_results: dict[str, dict[str, Any]],
    baseline_results: dict[str, dict[str, Any]],
    mu: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_methods = set(METHOD_CODES)

    for dataset, privtab_dataset_results in privtab_results.items():
        if "binary" not in privtab_dataset_results:
            raise KeyError(f"Dataset {dataset!r} has no 'binary' flag")
        is_binary = bool(privtab_dataset_results["binary"])
        source_by_method = {
            "PrivTab": privtab_dataset_results,
            "DP-LR": baseline_results.get(dataset, {}),
            "DP-MLP": baseline_results.get(dataset, {}),
        }

        errors_by_method: dict[str, np.ndarray] = {}
        repeats_by_method = {}
        for method_name, method_code in METHOD_CODES.items():
            source = source_by_method[method_name]
            if method_code not in source:
                raise KeyError(
                    f"Missing {method_name} ({method_code}) results for dataset {dataset!r}"
                )
            results_by_mu = source[method_code]
            mu_key = find_mu_key(results_by_mu, mu)
            if mu_key is None:
                raise KeyError(f"Missing mu={mu:g} for {method_name} on dataset {dataset!r}")
            metrics = results_by_mu[mu_key]
            errors_by_method[method_name] = metric_errors(metrics, is_binary)
            repeats_by_method[method_name] = list(metrics.get('repeat_ids', range(len(errors_by_method[method_name]))))
            if len(repeats_by_method[method_name]) != len(errors_by_method[method_name]) or len(set(repeats_by_method[method_name])) != len(repeats_by_method[method_name]):
                raise ValueError(f"Invalid repeat identifiers for {dataset}, {method_name}")

        repeat_counts = {name: len(values) for name, values in errors_by_method.items()}
        if len(set(repeat_counts.values())) != 1:
            raise ValueError(f"Mismatched repeat counts for {dataset!r}: {repeat_counts}")

        if any(ids != repeats_by_method['PrivTab'] for ids in repeats_by_method.values()):
            raise ValueError(f"Mismatched repeat identifiers for {dataset}")
        metric_name = "auc" if is_binary else "log_loss"
        for method_name, errors in errors_by_method.items():
            for repeat, error in zip(repeats_by_method[method_name], errors):
                rows.append(
                    {
                        "method": method_name,
                        "task": f"{dataset}_repeat{repeat}_mu{mu:g}",
                        "metric_error": float(error),
                        "dataset": dataset,
                        "repeat": repeat,
                        "metric": metric_name,
                    }
                )

    if not rows or set(pd.unique(pd.DataFrame(rows)["method"])) != expected_methods:
        raise ValueError("Not all requested methods were added to the Elo battles")
    return pd.DataFrame(rows)


def compute_elo(
    battles: pd.DataFrame,
    bootstrap_rounds: int,
    seed: int,
) -> pd.DataFrame:
    if bootstrap_rounds < 2:
        raise ValueError("--bootstrap-rounds must be at least 2 for a confidence interval")
    np.random.seed(seed)
    random.seed(seed)

    arena = TabArena(method_col="method", task_col="task", error_col="metric_error")
    elo = arena.compute_elo(
        results_per_task=battles[["method", "task", "metric_error"]],
        calibration_framework="DP-LR",
        calibration_elo=1000,
        include_quantiles=True,
        post_calibrate=True,
        BOOTSTRAP_ROUNDS=bootstrap_rounds,
    )
    elo["95% CI min"] = elo["elo"] - elo["elo-"]
    elo["95% CI max"] = elo["elo"] + elo["elo+"]
    return elo.drop(columns=["elo-", "elo+"]).rename(columns={"elo": "Elo"})


def main() -> None:
    args = parse_args()
    privtab_results = joblib.load(args.privtab_results)
    if not isinstance(privtab_results, dict):
        raise TypeError(
            f"Expected a results dict in {args.privtab_results}, "
            f"got {type(privtab_results).__name__}"
        )
    baseline_results = load_baseline_results(args.baselines_dir)
    battles = build_battles(privtab_results, baseline_results, args.mu)
    elo = compute_elo(battles, args.bootstrap_rounds, args.seed)

    print(f"Elo ranking at mu={args.mu:g} ({battles['task'].nunique()} tasks):")
    print(elo.to_string(float_format=lambda value: f"{value:.1f}"))

    output = elo.reset_index().rename(
        columns={
            "method": "method_name",
            "Elo": "elo_score",
            "95% CI min": "ci_lower",
            "95% CI max": "ci_upper",
        }
    )
    output.insert(1, "mu", args.mu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"\nWrote Elo scores to {args.output}")


if __name__ == "__main__":
    main()
