from __future__ import annotations

import math
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Calculates the standard deviation of a tensor, ignoring NaNs, using NumPy internally.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    dim : int or tuple[int], optional
        The dimension or dimensions to reduce. Defaults to None (reduce all dimensions).

    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not. Defaults to False.

    ddof : int, optional
        Delta Degrees of Freedom.

    dtype : torch.dtype, optional
        The desired data type of returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    device = input.device
    np_input = input.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)

    return torch.from_numpy(std).to(dtype=torch.float, device=device)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardizes features by removing the mean and scaling to unit variance.

    NaNs are ignored in mean/std calculation.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H), where T is sequence length, H is features.

    clip_value : float, optional, default=100
        The value to clip the standardized input to, preventing extreme outliers.

    Returns
    -------
    Tensor
        The standardized input, clipped between -clip_value and clip_value.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std

    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def quantile_minmax_m11_scaling(
    input: Tensor,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    eps: float = 1e-6,
) -> Tensor:
    """Clips features to quantile bounds and scales them to [-1, 1].

    NaNs are ignored when calculating quantiles. Constant and all-NaN columns are
    mapped to zero.
    """
    lower = torch.nanquantile(input, lower_q, dim=0)
    upper = torch.nanquantile(input, upper_q, dim=0)

    clipped_input = torch.clamp(input, min=lower, max=upper)
    feature_range = upper - lower
    invalid_range = torch.isnan(feature_range) | (feature_range < eps)
    safe_range = torch.where(invalid_range, torch.ones_like(feature_range), feature_range)

    scaled_input = (clipped_input - lower) / safe_range * 2.0 - 1.0
    scaled_input = torch.clamp(scaled_input, min=-1.0, max=1.0)

    return torch.where(invalid_range.unsqueeze(0), torch.zeros_like(scaled_input), scaled_input)


def approximate_public_minmax_m11_scaling(
    input: Tensor,
    min_public_occupancy: float = 0.05,
    max_public_occupancy: float = 1.0,
    max_public_center_offset: float = 0.75,
    public_bound_clipping_probability: float = 0.05,
    exact_public_bounds_probability: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:
    """Scale features with synthetic public bounds that are often, but not always, exact.

    The generated public interval is centered near the empirical midpoint and has a
    random occupancy relative to the empirical support. Most intervals cover the
    full observed support; a small fraction are intentionally shifted so values can
    fall outside [-1, 1], matching imperfect public-bound preprocessing at inference.
    """
    if not 0.0 < min_public_occupancy <= max_public_occupancy <= 1.0:
        raise ValueError(
            "Expected 0 < min_public_occupancy <= max_public_occupancy <= 1.0, "
            f"got ({min_public_occupancy}, {max_public_occupancy})."
        )
    if not 0.0 <= public_bound_clipping_probability <= 1.0:
        raise ValueError(
            "public_bound_clipping_probability must lie in [0, 1], "
            f"got {public_bound_clipping_probability}."
        )
    if not 0.0 <= exact_public_bounds_probability <= 1.0:
        raise ValueError(
            "exact_public_bounds_probability must lie in [0, 1], "
            f"got {exact_public_bounds_probability}."
        )

    valid_mask = ~torch.isnan(input)
    observed_min = torch.where(valid_mask, input, torch.inf).amin(dim=0)
    observed_max = torch.where(valid_mask, input, -torch.inf).amax(dim=0)
    has_observation = valid_mask.any(dim=0)

    feature_range = observed_max - observed_min
    invalid_range = (~has_observation) | torch.isnan(feature_range) | (feature_range < eps)
    safe_range = torch.where(invalid_range, torch.ones_like(feature_range), feature_range)
    data_center = 0.5 * (observed_min + observed_max)
    data_half_range = 0.5 * safe_range

    num_features = input.shape[1]
    occupancy = torch.empty(num_features, device=input.device, dtype=input.dtype).uniform_(
        math.log(min_public_occupancy),
        math.log(max_public_occupancy),
    ).exp()
    public_half_range = data_half_range / occupancy.clamp_min(min_public_occupancy)

    center_offsets = torch.empty(
        num_features, device=input.device, dtype=input.dtype
    ).uniform_(-max_public_center_offset, max_public_center_offset)

    clipping_mask = torch.rand(num_features, device=input.device) < public_bound_clipping_probability
    if clipping_mask.any():
        clipping_sign = torch.where(
            torch.rand(num_features, device=input.device) < 0.5,
            -torch.ones(num_features, device=input.device, dtype=input.dtype),
            torch.ones(num_features, device=input.device, dtype=input.dtype),
        )
        clipping_magnitude = torch.empty(
            num_features, device=input.device, dtype=input.dtype
        ).uniform_(0.5, max(0.5, max_public_center_offset))
        center_offsets = torch.where(
            clipping_mask,
            clipping_sign * clipping_magnitude,
            center_offsets,
        )

    exact_mask = torch.rand(num_features, device=input.device) < exact_public_bounds_probability
    center_offsets = torch.where(
        exact_mask,
        torch.zeros_like(center_offsets),
        center_offsets,
    )
    public_half_range = torch.where(
        exact_mask,
        data_half_range,
        public_half_range,
    )

    public_center = data_center + center_offsets * public_half_range
    scaled_input = (input - public_center) / public_half_range.clamp_min(eps)
    scaled_input = torch.where(invalid_range.unsqueeze(0), torch.zeros_like(scaled_input), scaled_input)
    return scaled_input


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamps outliers in the input tensor based on a specified number of standard deviations (threshold).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H).

    threshold : float, optional, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(
        masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0
    ).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking -> std is NaN or 0
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


def permute_classes(input: Tensor) -> Tensor:
    """Label encoding and permute classes.

    Parameters
    ----------
    input : Tensor
        Target of shape (T,) containing class labels.

    Returns
    -------
    Tensor
        Target with potentially permuted labels (T,).
    """
    unique_vals, _ = torch.unique(input, return_inverse=True)
    num_classes = len(unique_vals)

    if num_classes <= 1:  # No permutation needed for single class
        return input

    # Ensure labels are encoded from 0 to num_classes-1
    indices = unique_vals.argsort()
    mapped = indices[torch.searchsorted(unique_vals, input)]

    # Randomly permute classes
    perm = torch.randperm(num_classes, device=input.device)
    permuted = perm[mapped]

    return permuted


class BalancedBinarize(nn.Module):
    """Binarizes the input based on its median value."""

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).

        Returns
        -------
        Tensor
            Binarized output (0 or 1) of shape (T,).
        """
        return (input > torch.median(input)).float()


class MulticlassAssigner(nn.Module):
    """Transforms the input into discrete classes using rank-based or value-based thresholding.

    Input shape: (T,) -> Output shape: (T,)
    """

    def __init__(self, num_classes: int, mode: str = "rank", ordered_prob: float = 0.2):
        """
        Initializes the MulticlassAssigner.

        Parameters
        ----------
        num_classes : int
            The target number of discrete classes to output.

        mode : str, default="rank"
            The method used to determine class boundaries:
            - "rank": Boundaries are randomly sampled from the input.
            - "value": Boundaries are randomly sampled from a normal distribution.

        ordered_prob : float, default=0.2
            Probability of keeping the natural class order.
        """
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                "The number of classes must be at least 2 for MulticlassAssigner."
            )

        self.num_classes = num_classes
        self.ordered_prob = ordered_prob
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).

        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """

        T = input.shape[0]
        device = input.device

        if self.mode == "rank":
            boundary_indices = torch.randint(
                0, T, (self.num_classes - 1,), device=device
            )
            boundaries = input[boundary_indices]
        elif self.mode == "value":
            boundaries = torch.randn(self.num_classes - 1, device=device)

        # Compare input tensor with boundaries and sum across the boundary dimension to get classes
        classes = (input.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

        # Permute classes
        if random.random() > self.ordered_prob:
            classes = permute_classes(classes)

        # Reverse classes
        if random.random() > 0.5:
            classes = self.num_classes - 1 - classes

        return classes


class Reg2Cls(nn.Module):
    """Transforms a single regression dataset (features X, targets y) into a classification format
    through feature processing (categorical conversion, normalization) and target transformation
    (regression-to-classification).

    Parameters
    ----------
    hyperparameters : dict
        Configuration dictionary containing settings for feature processing and
        target transformation. Expected keys include:
        - num_classes (int): Number of classes for classification conversion.
        - max_features (int): Maximum number of features allowed (defines output feature dim).
        - multiclass_type (str): Strategy for multiclass conversion ('rank' or 'value').
        - balanced (bool): Whether to enforce balanced classes (currently only for binary).
        - multiclass_ordered_prob (float): Prob. of keeping natural class order.
        - cat_prob (float, optional): Probability of converting features to categorical.
        - max_categories (int, optional): Max categories for categorical conversion.
        - scale_by_max_features (bool): Whether to scale features by proportion used.
        - permute_features (bool, optional): Whether to randomly permute features. Defaults to True.
        - permute_labels (bool, optional): Whether to randomly permute final class labels. Defaults to True.

    Attributes
    ----------
    hp : dict
        Hyperparameters for the Reg2Cls module.

    class_assigner : nn.Module or None
        The module responsible for converting regression targets to class labels.
        None if num_classes is 0.
    """

    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp

        num_classes = self.hp["num_classes"]
        if num_classes == 0:
            self.class_assigner = None
        elif num_classes == 2 and self.hp.get("balanced", False):
            self.class_assigner = BalancedBinarize()
        elif num_classes >= 2:
            self.class_assigner = MulticlassAssigner(
                num_classes,
                mode=self.hp["multiclass_type"],
                ordered_prob=self.hp["multiclass_ordered_prob"],
            )
        else:
            raise ValueError(f"Invalid number of classes: {num_classes}")

    def forward(self, X: Tensor, y: Tensor, pre_processing_method : str = "zscore") -> tuple[Tensor, Tensor]:
        """Processes a single dataset (X, y) according to the initialized hyperparameters.

        Parameters
        ----------
        X : Tensor
            Features of shape (T, H), where H is the number of features.

        y : Tensor
            Targets of shape (T,).

        Returns
        -------
        tuple[Tensor, Tensor]
            A tuple containing:
            - Processed features of shape (T, max_features).
            - Processed targets of shape (T,).
        """
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Input shapes mismatch or incorrect dims. X: {X.shape}, y: {y.shape}"
            )

        X = self._num2cat(X)
        X = self._process_features(X, pre_processing_method)

        y = standard_scaling(y.unsqueeze(-1)).squeeze(-1)
        if self.class_assigner is not None:
            y = self.class_assigner(y)
            if self.hp.get("permute_labels", True):
                y = permute_classes(y)

        return X.float(), y.float()

    def _num2cat(self, X: Tensor) -> Tensor:
        """Converts some features to categorical based on hyperparameters.

        Operates inplace conceptually, returns the modified tensor.

        Parameters
        ----------
        X : Tensor
            Feature tensor of shape (T, H).

        Returns
        -------
        Tensor
            Feature tensor with some columns potentially converted to categorical (T, H).
        """

        if random.random() < self.hp.get("cat_prob", 0.2):
            col_prob = random.random()  # Probability for converting a specific column
            max_categories = self.hp.get("max_categories", 10)
            for col in range(X.shape[1]):
                if random.random() < col_prob:
                    # Determine number of categories for this feature
                    num_cats = min(
                        max(round(random.gammavariate(1, 10)), 2), max_categories
                    )
                    # Use MulticlassAssigner to convert this column
                    assigner = MulticlassAssigner(
                        num_cats, mode="rank", ordered_prob=0.3
                    )
                    X[:, col] = assigner(X[:, col]).float()
        return X

    def _process_features(self, X: Tensor, pre_processing_method : str = "zscore") -> Tensor:
        """Process inputs through outlier removal, shuffling, scaling, and padding to max features.

        Parameters
        ----------
        X : Tensor
            Feature tensor of shape (T, H).

        Returns
        -------
        Tensor
            Normalized feature tensor (T, H).
        """

        num_features = X.shape[1]
        max_features = self.hp["max_features"]

        if pre_processing_method == "zscore":
            X = outlier_removing(X, threshold=4)
            X = standard_scaling(X)
        elif pre_processing_method == "minmax":
            X = approximate_public_minmax_m11_scaling(
                X,
                min_public_occupancy=self.hp.get("min_public_occupancy", 0.05),
                max_public_occupancy=self.hp.get("max_public_occupancy", 1.0),
                max_public_center_offset=self.hp.get("max_public_center_offset", 0.75),
                public_bound_clipping_probability=self.hp.get(
                    "public_bound_clipping_probability",
                    0.05,
                ),
                exact_public_bounds_probability=self.hp.get(
                    "exact_public_bounds_probability",
                    0.25,
                ),
            )
        elif pre_processing_method == "quantile_minmax":
            X = quantile_minmax_m11_scaling(X, lower_q=0.05, upper_q=0.95)
        else:
            raise ValueError(f"Invalid pre-processing method: {pre_processing_method}")

        # Permute features if specified
        if self.hp.get("permute_features", True):
            perm = torch.randperm(num_features, device=X.device)
            X = X[:, perm]

        # Scale by the proportion of features used relative to max features
        if self.hp.get("scale_by_max_features", False):
            scaling_factor = num_features / max_features
            X = X / scaling_factor

        # Add empty features if needed to match max features
        if num_features < max_features:
            X = F.pad(X, (0, max_features - num_features), mode="constant", value=0.0)

        return X
