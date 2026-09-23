"""Preprocessing used by the paper's end-to-end evaluation workflows."""
import numpy as np
from sklearn.model_selection import train_test_split

BENCHMARK_MAX_FEATURES = 120
STANDARDIZATION_EPS = 1e-6
STANDARDIZATION_CLIP = 10.0


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

    try:
        train_idx, test_idx = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
        return train_idx, test_idx
    except ValueError:
        pass

    rng = np.random.default_rng(seed)

    mandatory_train = []
    remaining = set(indices.tolist())

    for cls in classes:
        cls_idx = indices[y == cls]
        chosen = int(rng.choice(cls_idx))
        mandatory_train.append(chosen)
        remaining.remove(chosen)

    mandatory_train = np.array(mandatory_train, dtype=int)
    remaining = np.array(sorted(remaining), dtype=int)

    extra_needed = train_size - len(mandatory_train)
    if extra_needed > 0:
        extra_train = rng.choice(remaining, size=extra_needed, replace=False)
        train_idx = np.concatenate([mandatory_train, extra_train])
    else:
        train_idx = mandatory_train

    train_idx = np.sort(train_idx)
    test_mask = np.ones(n, dtype=bool)
    test_mask[train_idx] = False
    test_idx = indices[test_mask]

    return train_idx, test_idx


def split_len_from_fraction(n_rows, train_data_percentage):
    return int(round(n_rows * train_data_percentage))


def pre_process_and_split_data(train_data_percentage, x, y, indices=None):
    x = np.asarray(x, dtype=np.float32)
    if indices is None:
        train_idx, test_idx = split_indices_with_class_coverage(
            y, train_data_percentage
        )
        train_x = x[train_idx]
        train_y = y[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
    else:
        shuffle_x = x[indices]
        shuffle_y = y[indices]
        train_len = split_len_from_fraction(x.shape[0], train_data_percentage)
        train_x = shuffle_x[:train_len]
        train_y = shuffle_y[:train_len]
        test_x = shuffle_x[train_len:]
        test_y = shuffle_y[train_len:]
    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0)
    safe_std = train_std.copy()
    constant_mask = safe_std < STANDARDIZATION_EPS
    safe_std[constant_mask] = 1.0
    train_x = (train_x - train_mean) / safe_std
    train_x[:, constant_mask] = 0.0
    train_x = np.clip(train_x, -STANDARDIZATION_CLIP, STANDARDIZATION_CLIP)
    train_features_count = train_x.shape[1]
    train_x = np.concatenate([train_x, np.zeros((train_x.shape[0], BENCHMARK_MAX_FEATURES - train_features_count))],
                             axis=-1)
    test_x = (test_x - train_mean) / safe_std
    test_x[:, constant_mask] = 0.0
    test_x = np.clip(test_x, -STANDARDIZATION_CLIP, STANDARDIZATION_CLIP)
    test_features_count = test_x.shape[1]
    if train_features_count != test_features_count:
        raise ValueError(
            f"train_features_count ({train_features_count}) != test_features_count ({test_features_count})"
        )
    test_x = np.concatenate([test_x, np.zeros((test_x.shape[0], BENCHMARK_MAX_FEATURES - test_features_count))],
                            axis=-1)
    return train_x, train_y, test_x, test_y, train_features_count


def _apply_feature_minmax_m11(x, feature_mins, feature_maxs):
    feature_mins = np.asarray(feature_mins, dtype=np.float32)
    feature_maxs = np.asarray(feature_maxs, dtype=np.float32)
    feature_ranges = feature_maxs - feature_mins
    constant_mask = np.isnan(feature_ranges) | (feature_ranges < STANDARDIZATION_EPS)
    safe_feature_ranges = feature_ranges.copy()
    safe_feature_ranges[constant_mask] = 1.0

    x = (x - feature_mins) / safe_feature_ranges * 2.0 - 1.0
    x[:, constant_mask] = 0.0
    return x.astype(np.float32), constant_mask


def pre_process_and_split_data_public_minmax(
        train_data_percentage,
        x,
        y,
        feature_mins,
        feature_maxs,
        indices=None,
):
    """
    Min-max preprocessing using supplied public feature bounds.

    The bounds are not estimated from private training data, so this helper does
    not consume privacy budget. Features are mapped with
    z = 2 * (x - a) / (b - a) - 1 and padded to BENCHMARK_MAX_FEATURES.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if indices is None:
        train_idx, test_idx = split_indices_with_class_coverage(
            y, train_data_percentage
        )
        train_x = x[train_idx]
        train_y = y[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
    else:
        shuffle_x = x[indices]
        shuffle_y = y[indices]
        train_len = split_len_from_fraction(x.shape[0], train_data_percentage)
        train_x = shuffle_x[:train_len]
        train_y = shuffle_y[:train_len]
        test_x = shuffle_x[train_len:]
        test_y = shuffle_y[train_len:]

    train_features_count = train_x.shape[1]
    test_features_count = test_x.shape[1]
    if train_features_count != test_features_count:
        raise ValueError(
            f"train_features_count ({train_features_count}) != test_features_count ({test_features_count})"
        )

    train_x, _ = _apply_feature_minmax_m11(train_x, feature_mins, feature_maxs)
    test_x, _ = _apply_feature_minmax_m11(test_x, feature_mins, feature_maxs)

    train_x = np.concatenate(
        [train_x, np.zeros((train_x.shape[0], BENCHMARK_MAX_FEATURES - train_features_count))],
        axis=-1,
    )
    test_x = np.concatenate(
        [test_x, np.zeros((test_x.shape[0], BENCHMARK_MAX_FEATURES - test_features_count))],
        axis=-1,
    )
    return train_x, train_y, test_x, test_y, train_features_count


def pre_process_and_split_data_dp_range_zscore_with_public_ranges(train_data_percentage,
                                                                  x,
                                                                  y,
                                                                  mu,
                                                                  feature_mins,
                                                                  feature_maxs,
                                                                  indices=None):
    """
    DP z-score preprocessing on the original feature scale.

    The supplied feature bounds must be public for an end-to-end privacy
    guarantee. Empirical/oracle bounds are a research-only comparison.
    Noisy first and second moments use independent Gaussian perturbations.
    """
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    x = np.asarray(x, dtype=np.float32)
    feature_mins = np.asarray(feature_mins, dtype=np.float64)
    feature_maxs = np.asarray(feature_maxs, dtype=np.float64)
    if x.ndim != 2 or not 1 <= x.shape[1] <= BENCHMARK_MAX_FEATURES or not np.isfinite(x).all():
        raise ValueError("Expected finite numeric features with width 1–120.")
    if feature_mins.shape != (x.shape[1],) or feature_maxs.shape != feature_mins.shape:
        raise ValueError("Public bounds must have one lower/upper bound per feature.")
    if not np.isfinite(feature_mins).all() or not np.isfinite(feature_maxs).all() or (feature_maxs < feature_mins).any():
        raise ValueError("Public bounds must be finite and ordered.")
    rng = np.random.default_rng()  # Private randomness, independent of benchmark seeds.
    if indices is None:
        train_idx, test_idx = split_indices_with_class_coverage(
            y, train_data_percentage
        )
        train_x = x[train_idx]
        train_y = y[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
    else:
        shuffle_x = x[indices]
        shuffle_y = y[indices]
        train_len = split_len_from_fraction(x.shape[0], train_data_percentage)
        train_x = shuffle_x[:train_len]
        train_y = shuffle_y[:train_len]
        test_x = shuffle_x[train_len:]
        test_y = shuffle_y[train_len:]
    train_features_count = train_x.shape[1]
    test_features_count = test_x.shape[1]
    if train_features_count != test_features_count:
        raise ValueError(
            f"train_features_count ({train_features_count}) != test_features_count ({test_features_count})"
        )

    if len(train_x) == 0 or len(test_x) == 0:
        raise ValueError("Context and target splits must both be nonempty.")

    sensitivity = 1.0

    feature_ranges = feature_maxs - feature_mins
    constant_mask = feature_ranges < STANDARDIZATION_EPS
    safe_feature_ranges = feature_ranges.copy()
    safe_feature_ranges[constant_mask] = 1.0

    train_x_scaled = (train_x - feature_mins) / safe_feature_ranges
    test_x_scaled = (test_x - feature_mins) / safe_feature_ranges
    train_x_scaled[:, constant_mask] = 0.0
    test_x_scaled[:, constant_mask] = 0.0
    train_x_scaled = np.clip(train_x_scaled, 0.0, 1.0)
    test_x_scaled = np.clip(test_x_scaled, 0.0, 1.0)

    mu_mean = 0.5 * mu
    mu_std = np.sqrt(mu ** 2 - mu_mean ** 2)

    train_x_noisy_sum = np.sum(train_x_scaled, axis=0) + rng.normal(
        loc=0.0,
        scale=np.sqrt(train_features_count) * sensitivity / mu_mean,
        size=train_features_count,
    )

    train_x_squared = np.square(train_x_scaled)
    train_x_noisy_sum_of_squares = np.sum(train_x_squared, axis=0) + rng.normal(
        loc=0.0,
        scale=np.sqrt(train_features_count) * sensitivity / mu_std,
        size=train_features_count,
    )

    train_x_mu = train_x_noisy_sum / train_x.shape[0]
    train_x_var = train_x_noisy_sum_of_squares / train_x.shape[0] - train_x_mu ** 2
    low_var_mask = np.logical_or(train_x_var < STANDARDIZATION_EPS, constant_mask)
    train_x_std = np.sqrt(np.maximum(train_x_var, STANDARDIZATION_EPS))

    train_x = (train_x_scaled - train_x_mu) / train_x_std
    test_x = (test_x_scaled - train_x_mu) / train_x_std
    train_x[:, low_var_mask] = 0.0
    test_x[:, low_var_mask] = 0.0
    train_x = np.clip(train_x, -STANDARDIZATION_CLIP, STANDARDIZATION_CLIP)
    test_x = np.clip(test_x, -STANDARDIZATION_CLIP, STANDARDIZATION_CLIP)

    train_x = np.concatenate([train_x, np.zeros((train_x.shape[0], BENCHMARK_MAX_FEATURES - train_features_count))],
                             axis=-1)
    test_x = np.concatenate([test_x, np.zeros((test_x.shape[0], BENCHMARK_MAX_FEATURES - test_features_count))],
                            axis=-1)
    return train_x, train_y, test_x, test_y, train_features_count
