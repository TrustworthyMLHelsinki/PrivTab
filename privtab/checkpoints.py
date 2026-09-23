"""Load the tensor-only PrivTab checkpoint weights."""
from pathlib import Path

import torch

from .model import PrivTab


def load_model(path, device="cpu"):
    """Load a repository ``weights.pt`` without training dependencies."""
    model = PrivTab(normalize_perturbed_output=True)
    model.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()
