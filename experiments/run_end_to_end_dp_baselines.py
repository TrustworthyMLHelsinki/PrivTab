import argparse
import math
import os
import random
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import joblib
import numpy as np
import torch

from experiments import end_to_end_cases as cases
from experiments.dataset_utils import (
    BENCHMARK_MAX_FEATURES,
    STANDARDIZATION_EPS,
    pre_process_and_split_data,
    pre_process_and_split_data_dp_range_zscore_with_public_ranges,
)
from experiments.dp_hpo import (
    DEFAULT_MODEL_PLANS,
    ModelPlan,
    allocate_privacy_budget,
    build_summary_rows,
    compute_auc,
    empty_metrics_dict,
    fit_private_model,
    lossless_float,
    select_learning_rate,
    write_csv,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DP Logistic Regression/MLP baselines on the end-to-end case-study datasets."
    )
    parser.add_argument("--run-name", default="end-to-end-dp-baselines")
    parser.add_argument(
        "--output-dir",
        default="runs/end_to_end_dp_baselines",
    )
    parser.add_argument(
        "--indices-dir",
        default=os.path.join(os.environ.get("SCRATCH_PATH", "experiments"), "end_to_end_indices"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mu-values",
        type=float,
        nargs="+",
        default=(0.4,),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DEFAULT_MODEL_PLANS),
        default=("dp_logistic_regression", "dp_mlp"),
    )
    parser.add_argument(
        "--preprocessing",
        choices=("exact_zscore", "dp_llm_bounds", "dp_oracle_bounds"),
        default="dp_llm_bounds",
        help="Feature preprocessing to use before DP model training.",
    )
    parser.add_argument(
        "--llm-bounds-scaling",
        choices=("public_minmax01", "dp_zscore"),
        default="dp_zscore",
        help=(
            "For --preprocessing dp_llm_bounds: either scale to [0, 1] with cached "
            "public/LLM bounds without spending privacy budget, or run the existing "
            "DP z-score preprocessing that spends --preprocessing-mu-fraction."
        ),
    )
    parser.add_argument(
        "--preprocessing-mu-fraction",
        type=float,
        default=0.6,
        help="GDP budget fraction spent on preprocessing for dp_zscore preprocessing modes.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.16)
    parser.add_argument("--selection-mu-fraction", type=float, default=0.5)
    parser.add_argument("--validation-loss-clip", type=float, default=5.0)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--leaderboard-metric", choices=("auc", "loss"), default="auc")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--case-datasets-cache",
        default=os.environ.get("END_TO_END_CASE_DATASETS_CACHE", ""),
        help=(
            "Optional joblib cache for already-loaded case-study datasets. "
            "Use --initialize-case-datasets-cache to create it before batch runs."
        ),
    )
    parser.add_argument(
        "--initialize-case-datasets-cache",
        action="store_true",
        help="Load selected case-study datasets, write --case-datasets-cache, and exit.",
    )
    parser.add_argument(
        "--allow-llm-bounds-refresh",
        action="store_true",
        help="Allow missing/incomplete cached bounds to trigger LLM inference.",
    )
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_case_studies(specs, dataset_slug=None, dataset_index=None):
    if dataset_slug is not None and dataset_index is not None:
        raise ValueError("Use only one of --dataset and --dataset-index.")
    if dataset_slug is not None:
        selected = [spec for spec in specs if spec["slug"] == dataset_slug]
        if not selected:
            known = ", ".join(spec["slug"] for spec in specs)
            raise ValueError(f"Unknown dataset slug {dataset_slug!r}. Known datasets: {known}")
        return selected
    if dataset_index is not None:
        if dataset_index < 0 or dataset_index >= len(specs):
            raise IndexError(f"--dataset-index must be in [0, {len(specs) - 1}]")
        return [specs[dataset_index]]
    return specs


def split_indices_with_class_coverage(y, train_fraction, seed=0):
    y = np.asarray(y)
    n = len(y)
    indices = np.arange(n)
    train_size = int(n * train_fraction)
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
        cls_train_size = max(1, min(cls_train_size, len(cls_idx) - 1))
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

    return np.concatenate([np.sort(train_idx), np.sort(test_idx)])


def ensure_indices(y, dataset_slug: str, indices_dir: str, repeats: int, train_fraction: float):
    os.makedirs(indices_dir, exist_ok=True)
    for repeat in range(repeats):
        path = os.path.join(indices_dir, f"{dataset_slug}_repeat_{repeat}.pickle")
        if not os.path.isfile(path):
            indices = split_indices_with_class_coverage(
                y,
                train_fraction=train_fraction,
                seed=repeat,
            )
            joblib.dump(indices, filename=path)


def load_repeat_indices(indices_dir: str, dataset_slug: str, repeat: int):
    path = os.path.join(indices_dir, f"{dataset_slug}_repeat_{repeat}.pickle")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing split indices file: {path}")
    return joblib.load(path)


def pad_to_benchmark_features(train_x: np.ndarray, test_x: np.ndarray):
    train_features_count = train_x.shape[1]
    test_features_count = test_x.shape[1]
    if train_features_count != test_features_count:
        raise ValueError(
            f"train_features_count ({train_features_count}) != test_features_count ({test_features_count})"
        )
    if train_features_count > BENCHMARK_MAX_FEATURES:
        raise ValueError(
            f"Dataset has {train_features_count} features, exceeding BENCHMARK_MAX_FEATURES={BENCHMARK_MAX_FEATURES}."
        )
    pad_width = BENCHMARK_MAX_FEATURES - train_features_count
    train_x = np.concatenate(
        [train_x, np.zeros((train_x.shape[0], pad_width), dtype=np.float32)],
        axis=-1,
    )
    test_x = np.concatenate(
        [test_x, np.zeros((test_x.shape[0], pad_width), dtype=np.float32)],
        axis=-1,
    )
    return train_x, test_x, train_features_count


def pre_process_and_split_data_public_minmax01(
    train_fraction,
    x,
    y,
    feature_mins,
    feature_maxs,
    indices=None,
):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if indices is None:
        indices = split_indices_with_class_coverage(y, train_fraction, seed=0)

    shuffled_x = x[indices]
    shuffled_y = y[indices]
    train_len = int(x.shape[0] * train_fraction)
    train_x = shuffled_x[:train_len]
    train_y = shuffled_y[:train_len]
    test_x = shuffled_x[train_len:]
    test_y = shuffled_y[train_len:]

    feature_mins = np.asarray(feature_mins, dtype=np.float32)
    feature_maxs = np.asarray(feature_maxs, dtype=np.float32)
    feature_ranges = feature_maxs - feature_mins
    constant_mask = feature_ranges < STANDARDIZATION_EPS
    safe_ranges = feature_ranges.copy()
    safe_ranges[constant_mask] = 1.0

    train_x = (train_x - feature_mins) / safe_ranges
    test_x = (test_x - feature_mins) / safe_ranges
    train_x[:, constant_mask] = 0.0
    test_x[:, constant_mask] = 0.0
    train_x = np.clip(train_x, 0.0, 1.0).astype(np.float32)
    test_x = np.clip(test_x, 0.0, 1.0).astype(np.float32)
    train_x, test_x, features_count = pad_to_benchmark_features(train_x, test_x)
    return train_x, train_y, test_x, test_y, features_count


def load_cached_or_inferred_bounds(spec, dataset_name, columns, allow_llm_bounds_refresh: bool):
    if allow_llm_bounds_refresh:
        return cases.infer_public_bounds_for_dataset(
            dataset_slug=spec["slug"],
            dataset_name=dataset_name,
            columns=columns,
            dataset_context=spec["context"],
        )

    direct_results = {
        column["name"]: cases.direct_bound_result(column)
        for column in columns
        if not cases.needs_llm_bounds(column)
    }
    numeric_columns = [
        column
        for column in columns
        if cases.needs_llm_bounds(column)
    ]
    if not numeric_columns:
        return direct_results

    numeric_feature_names = [column["name"] for column in numeric_columns]
    cache_path = cases.bounds_cache_path(spec["slug"])
    cached_numeric_results = cases.load_bounds_cache(
        path=cache_path,
        required_features=numeric_feature_names,
    )
    if cached_numeric_results is None:
        raise FileNotFoundError(
            f"Missing or incomplete cached bounds for {spec['slug']} at {cache_path}. "
            "Create the cache with end_to_end_cases.py or rerun this script with "
            "--allow-llm-bounds-refresh."
        )
    numeric_results = {
        feature: cached_numeric_results[feature]
        for feature in numeric_feature_names
    }
    return {**direct_results, **numeric_results}


def load_case_study(spec, allow_llm_bounds_refresh: bool):
    dataset_name, (x, y), columns = spec["loader"]()
    bounds_results = load_cached_or_inferred_bounds(
        spec=spec,
        dataset_name=dataset_name,
        columns=columns,
        allow_llm_bounds_refresh=allow_llm_bounds_refresh,
    )
    feature_mins, feature_maxs = cases.clipping_bounds_to_arrays(
        columns=columns,
        results=bounds_results,
    )
    oracle_feature_mins, oracle_feature_maxs = cases.compute_oracle_feature_bounds(x)
    return {
        "slug": spec["slug"],
        "dataset_name": dataset_name,
        "x": x,
        "y": y,
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs,
        "oracle_feature_mins": oracle_feature_mins,
        "oracle_feature_maxs": oracle_feature_maxs,
        "columns": columns,
        "bounds_results": bounds_results,
    }


def load_case_studies_from_cache(cache_path: str, selected_specs):
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.is_file():
        return None

    cached = joblib.load(path)
    if isinstance(cached, dict):
        cached_by_slug = cached
    elif isinstance(cached, list):
        cached_by_slug = {item["slug"]: item for item in cached}
    else:
        raise TypeError(
            f"Expected case dataset cache to contain a dict or list, got {type(cached).__name__}."
        )

    selected_slugs = [spec["slug"] for spec in selected_specs]
    missing = [slug for slug in selected_slugs if slug not in cached_by_slug]
    if missing:
        raise KeyError(
            f"Case dataset cache {path} is missing selected datasets: {missing}."
        )
    print(f"Loaded case-study datasets from {path}", flush=True)
    return [cached_by_slug[slug] for slug in selected_slugs]


def write_case_studies_cache(cache_path: str, case_studies):
    if not cache_path:
        raise ValueError("--case-datasets-cache is required with --initialize-case-datasets-cache.")
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({case["slug"]: case for case in case_studies}, filename=path)
    print(f"Saved case-study dataset cache to {path}", flush=True)


def split_total_mu_for_preprocessing(total_mu: float, preprocessing_mu_fraction: float):
    if not 0.0 < preprocessing_mu_fraction < 1.0:
        raise ValueError(
            f"--preprocessing-mu-fraction must lie strictly in (0, 1), got {preprocessing_mu_fraction}."
        )
    if not math.isfinite(total_mu) or total_mu <= 0:
        raise ValueError("Total mu must be finite and positive.")
    preprocessing_mu = total_mu * preprocessing_mu_fraction
    training_mu = math.sqrt(max(0.0, total_mu ** 2 - preprocessing_mu ** 2))
    return preprocessing_mu, training_mu


def preprocess_for_mode(args, case, indices, mu):
    if args.preprocessing == "exact_zscore":
        return pre_process_and_split_data(
            args.train_fraction,
            case["x"],
            case["y"],
            indices=indices,
        ), mu, {"preprocessing_mu": 0.0, "training_mu": mu}

    if args.preprocessing == "dp_oracle_bounds":
        preprocessing_mu, training_mu = split_total_mu_for_preprocessing(
            mu,
            args.preprocessing_mu_fraction,
        )
        split = pre_process_and_split_data_dp_range_zscore_with_public_ranges(
            args.train_fraction,
            case["x"],
            case["y"],
            preprocessing_mu,
            feature_mins=case["oracle_feature_mins"],
            feature_maxs=case["oracle_feature_maxs"],
            indices=indices,
        )
        return split, training_mu, {
            "preprocessing_mu": preprocessing_mu,
            "training_mu": training_mu,
        }

    if args.preprocessing != "dp_llm_bounds":
        raise ValueError(f"Unsupported preprocessing mode: {args.preprocessing}")

    if args.llm_bounds_scaling == "public_minmax01":
        split = pre_process_and_split_data_public_minmax01(
            args.train_fraction,
            case["x"],
            case["y"],
            feature_mins=case["feature_mins"],
            feature_maxs=case["feature_maxs"],
            indices=indices,
        )
        return split, mu, {"preprocessing_mu": 0.0, "training_mu": mu}

    preprocessing_mu, training_mu = split_total_mu_for_preprocessing(
        mu,
        args.preprocessing_mu_fraction,
    )
    split = pre_process_and_split_data_dp_range_zscore_with_public_ranges(
        args.train_fraction,
        case["x"],
        case["y"],
        preprocessing_mu,
        feature_mins=case["feature_mins"],
        feature_maxs=case["feature_maxs"],
        indices=indices,
    )
    return split, training_mu, {
        "preprocessing_mu": preprocessing_mu,
        "training_mu": training_mu,
    }


def method_code_for(model_code: str, preprocessing: str, llm_bounds_scaling: str):
    if preprocessing == "dp_llm_bounds":
        return f"{model_code}_dp_llm_bounds_{llm_bounds_scaling}"
    return f"{model_code}_{preprocessing}"


def append_result(metrics_dict, loss, acc, auc, probs, selected_params):
    metrics_dict["accs"].append(lossless_float(acc))
    metrics_dict["probs"].append(probs)
    metrics_dict["aucs"].append(lossless_float(auc))
    metrics_dict["losses"].append(lossless_float(loss))
    metrics_dict.setdefault("selected_params", []).append(selected_params)


def _method_display_name(method_code: str):
    for base_model_code, plan in DEFAULT_MODEL_PLANS.items():
        if method_code.startswith(base_model_code):
            suffix = method_code[len(base_model_code):].lstrip("_")
            if suffix:
                suffix = suffix.replace("_", " ")
                return f"{plan.display_name} ({suffix})"
            return plan.display_name
    return method_code.replace("_", " ")


def bootstrap_mean_ci(
        values,
        confidence: float = 0.95,
        rounds: int = 10000,
        seed: int = 0,
):
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        return np.nan, np.nan, np.nan
    if values_array.size == 1:
        value = float(values_array[0])
        return value, value, value

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values_array.size,
        size=(rounds, values_array.size),
    )
    bootstrap_means = values_array[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(values_array.mean()), float(lower), float(upper)


def format_bootstrap_ci(values, scale: float = 1.0) -> str:
    mean, lower, upper = bootstrap_mean_ci(values)
    return f"{mean * scale:.2f} [{lower * scale:.2f}, {upper * scale:.2f}]"


def markdown_table(headers, rows) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _build_end_to_end_markdown_table(results_dict, dataset_name, mu_values_ordered, method_codes):
    headers = ["Method", "mu", "ACC mean [95% CI]", "AUC mean [95% CI]"]
    rows = []
    for method_code in method_codes:
        metrics_by_mu = results_dict[dataset_name].get(method_code, {})
        method_name = _method_display_name(method_code)
        for mu in mu_values_ordered:
            if mu not in metrics_by_mu:
                continue
            metrics = metrics_by_mu[mu]
            rows.append(
                [
                    method_name,
                    str(mu),
                    format_bootstrap_ci(metrics["accs"], scale=100.0),
                    format_bootstrap_ci(metrics["aucs"], scale=100.0),
                ]
            )
    return "\n".join([f"## {dataset_name}", "", markdown_table(headers, rows), ""])


def print_saved_results(results_dir="runs/end_to_end_dp_baselines"):
    results_dir = Path(results_dir)
    result_paths = sorted(results_dir.glob("*-all-results.pickle"))
    if not result_paths:
        raise FileNotFoundError(f"No saved result pickles found in {results_dir}")

    merged_results = {}
    for result_path in result_paths:
        shard_results = joblib.load(result_path)
        for dataset_name, methods in shard_results.items():
            dataset_results = merged_results.setdefault(dataset_name, {})
            for method_code, metrics_by_mu in methods.items():
                if method_code == "binary":
                    continue
                method_results = dataset_results.setdefault(method_code, {})
                for mu, metrics in metrics_by_mu.items():
                    merged_metrics = method_results.setdefault(mu, empty_metrics_dict())
                    for metric_key in ("accs", "probs", "aucs", "losses"):
                        merged_metrics.setdefault(metric_key, []).extend(metrics.get(metric_key, []))
                    merged_metrics.setdefault("selected_params", []).extend(
                        metrics.get("selected_params", [])
                    )

    for dataset_name in sorted(merged_results):
        dataset_results = merged_results[dataset_name]
        mu_values = sorted(
            {mu for metrics_by_mu in dataset_results.values() for mu in metrics_by_mu}
        )
        preferred_method_order = [
            method_code
            for base_model_code in DEFAULT_MODEL_PLANS
            for method_code in dataset_results
            if method_code.startswith(base_model_code)
        ]
        extra_methods = sorted(
            method_code
            for method_code in dataset_results
            if method_code not in preferred_method_order
        )
        method_codes = preferred_method_order + extra_methods
        print(
            _build_end_to_end_markdown_table(
                merged_results,
                dataset_name,
                mu_values,
                method_codes,
            ),
            flush=True,
        )


def main():
    args = parse_args()
    set_global_seed(args.seed)

    specs = select_case_studies(
        cases.case_study_specs(),
        dataset_slug=args.dataset,
        dataset_index=args.dataset_index,
    )
    case_studies = load_case_studies_from_cache(args.case_datasets_cache, specs)
    if case_studies is None:
        case_studies = [
            load_case_study(
                spec,
                allow_llm_bounds_refresh=args.allow_llm_bounds_refresh,
            )
            for spec in specs
        ]

    if args.initialize_case_datasets_cache:
        write_case_studies_cache(args.case_datasets_cache, case_studies)
        return

    for case in case_studies:
        ensure_indices(
            y=case["y"],
            dataset_slug=case["slug"],
            indices_dir=args.indices_dir,
            repeats=args.repeats,
            train_fraction=args.train_fraction,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.run_name}-all-results.pickle"
    raw_csv_path = output_dir / f"{args.run_name}-raw.csv"
    summary_csv_path = output_dir / f"{args.run_name}-summary.csv"
    dataset_summary_csv_path = output_dir / f"{args.run_name}-dataset-summary.csv"

    repeats_to_run = [args.repeat] if args.repeat is not None else list(range(args.repeats))
    results_dict = {}
    raw_rows = []

    total_datasets = len(case_studies)
    total_repeats = len(repeats_to_run)
    total_mus = len(args.mu_values)
    total_models = len(args.models)

    for dataset_idx, case in enumerate(case_studies, start=1):
        dataset_name = case["dataset_name"]
        results_dict.setdefault(dataset_name, {"binary": len(np.unique(case["y"])) == 2})
        print(
            f"[run] dataset {dataset_idx}/{total_datasets}: {dataset_name} "
            f"slug={case['slug']} samples={case['x'].shape[0]} features={case['x'].shape[1]} "
            f"classes={len(np.unique(case['y']))}",
            flush=True,
        )

        for model_code in args.models:
            output_method_code = method_code_for(
                model_code,
                args.preprocessing,
                args.llm_bounds_scaling,
            )
            results_dict[dataset_name].setdefault(output_method_code, {})

        for repeat_idx, repeat in enumerate(repeats_to_run, start=1):
            indices = load_repeat_indices(args.indices_dir, case["slug"], repeat)
            print(
                f"[run] dataset={dataset_name}: repeat {repeat_idx}/{total_repeats} "
                f"(repeat_id={repeat})",
                flush=True,
            )

            for mu_idx, mu in enumerate(args.mu_values, start=1):
                split, training_total_mu, preprocessing_budget = preprocess_for_mode(
                    args=args,
                    case=case,
                    indices=indices,
                    mu=mu,
                )
                train_x, train_y, test_x, test_y, features_count = split
                train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
                train_y_tensor = torch.tensor(train_y, dtype=torch.long)
                test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
                test_y_tensor = torch.tensor(test_y, dtype=torch.long)

                print(
                    f"[run] dataset={dataset_name} repeat={repeat}: mu {mu_idx}/{total_mus} "
                    f"(total_mu={mu}, training_total_mu={training_total_mu:.6g}, "
                    f"preprocessing={args.preprocessing}, scaling={args.llm_bounds_scaling}, "
                    f"features_count={features_count})",
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
                        total_mu=training_total_mu,
                        selection_mu_fraction=args.selection_mu_fraction,
                        num_trials=len(plan.learning_rates),
                    )
                    output_method_code = method_code_for(
                        model_code,
                        args.preprocessing,
                        args.llm_bounds_scaling,
                    )
                    context_label = (
                        f"dataset={dataset_name} repeat={repeat} mu={mu} "
                        f"model={model_code} ({model_idx}/{total_models}) "
                        f"preprocessing={args.preprocessing}/{args.llm_bounds_scaling}"
                    )
                    print(
                        f"[privacy] {context_label}: total_mu={mu:.6g} "
                        f"preprocessing_mu={preprocessing_budget['preprocessing_mu']:.6g} "
                        f"training_total_mu={training_total_mu:.6g} "
                        f"selection_total_mu={privacy_budget.selection_total_mu:.6g} "
                        f"final_training_mu={privacy_budget.final_training_mu:.6g}",
                        flush=True,
                    )
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
                        num_classes=int(np.max(case["y"])) + 1,
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
                        num_classes=int(np.max(case["y"])) + 1,
                    )
                    test_auc = compute_auc(test_y, test_probs)

                    metrics_dict = results_dict[dataset_name][output_method_code].setdefault(
                        mu,
                        empty_metrics_dict(),
                    )
                    selected_params = {
                        "learning_rate": best_trial["learning_rate"],
                        "max_grad_norm": plan.max_grad_norm,
                        "q": plan.q,
                        "epochs": plan.epochs,
                        "nn_width": plan.nn_width,
                        "nn_layers": plan.nn_layers,
                        "selection_metric": "dp_val_loss",
                        "preprocessing": args.preprocessing,
                        "llm_bounds_scaling": args.llm_bounds_scaling,
                        "total_mu": mu,
                        "preprocessing_mu": preprocessing_budget["preprocessing_mu"],
                        "training_total_mu": training_total_mu,
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
                    metric_error = (
                        1.0 - test_auc
                        if args.leaderboard_metric == "auc" and np.isfinite(test_auc)
                        else test_loss
                    )
                    raw_rows.append(
                        {
                            "dataset": dataset_name,
                            "dataset_slug": case["slug"],
                            "repeat": repeat,
                            "mu": mu,
                            "method": plan.display_name,
                            "method_code": output_method_code,
                            "base_model_code": model_code,
                            "preprocessing": args.preprocessing,
                            "llm_bounds_scaling": args.llm_bounds_scaling,
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
                            "total_mu": mu,
                            "preprocessing_mu": preprocessing_budget["preprocessing_mu"],
                            "training_total_mu": training_total_mu,
                            "final_training_mu": privacy_budget.final_training_mu,
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
        "dataset_slug",
        "repeat",
        "mu",
        "method",
        "method_code",
        "base_model_code",
        "preprocessing",
        "llm_bounds_scaling",
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
        "total_mu",
        "preprocessing_mu",
        "training_total_mu",
        "final_training_mu",
    ]
    write_csv(raw_csv_path, raw_rows, raw_fieldnames)

    summary_rows = build_summary_rows(raw_rows, group_keys=("method_code", "mu"))
    summary_fieldnames = [
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

    dataset_summary_rows = build_summary_rows(raw_rows, group_keys=("dataset", "method_code", "mu"))
    dataset_summary_fieldnames = [
        "dataset",
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

    print(f"Saved results pickle to {results_path}")
    print(f"Saved raw rows CSV to {raw_csv_path}")
    print(f"Saved overall summary CSV to {summary_csv_path}")
    print(f"Saved per-dataset summary CSV to {dataset_summary_csv_path}")


if __name__ == "__main__":
    main()
