"""Save and reuse a single private summary without retaining context data."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import PrivTab


@dataclass(frozen=True)
class ReleasedSummary:
    """One task's DP summary and public schema; use the same model weights to predict."""

    tensor: torch.Tensor
    mu: float
    num_features: int
    num_classes: int

    def __post_init__(self):
        if self.tensor.shape != (1, 5, 128, 256) or not self.tensor.is_floating_point():
            raise ValueError("Expected one floating-point summary of shape [1, 5, 128, 256].")
        if not torch.isfinite(self.tensor).all():
            raise ValueError("Summary must contain only finite values.")
        if not math.isfinite(self.mu) or self.mu <= 0:
            raise ValueError("mu must be finite and positive.")
        if not 1 <= self.num_features <= 120 or not 2 <= self.num_classes <= 10:
            raise ValueError("Feature and class counts must be valid public schema values.")

    @classmethod
    def create(cls, model: PrivTab, xc: torch.Tensor, yc: torch.Tensor,
               mu: float, num_classes: int):
        """Spend one mu-GDP release with secure noise from preprocessed context."""
        if xc.ndim != 2 or yc.ndim != 1 or yc.shape[0] != xc.shape[0]:
            raise ValueError("Expected xc [rows, features] and yc [rows].")
        if (yc < 0).any() or (yc >= num_classes).any():
            raise ValueError("Context label outside the public class vocabulary.")
        device = next(model.parameters()).device
        summary = model.release_summary(xc.to(device).unsqueeze(0), yc.to(device).unsqueeze(0), mu)
        return cls(summary.detach().cpu(), float(mu), xc.shape[1], num_classes)

    def save(self, path):
        """Write tensor-only data; refuse overwrite to avoid accidental re-release."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "summary": self.tensor.detach().cpu().clone(),
                   "mu": self.mu, "num_features": self.num_features,
                   "num_classes": self.num_classes}
        with path.open("xb") as output:
            torch.save(payload, output)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported summary format version.")
        return cls(payload["summary"], payload["mu"], payload["num_features"],
                   payload["num_classes"])

    @torch.inference_mode()
    def predict_proba(self, model: PrivTab, xt: torch.Tensor, batch_size: int = 1024):
        """Reuse the saved release with the same checkpoint; no context access."""
        if batch_size <= 0 or xt.ndim != 2 or xt.shape[1] != self.num_features:
            raise ValueError("Expected [rows, num_features] queries and a positive batch size.")
        if xt.shape[0] == 0:
            return torch.empty((0, self.num_classes))
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            summary = self.tensor.to(device)
            outputs = [model.predict_from_summary(summary, part.to(device).unsqueeze(0),
                                                  d=self.num_features)[0, :, :self.num_classes]
                       .softmax(-1).cpu() for part in xt.split(batch_size)]
            return torch.cat(outputs)
        finally:
            model.train(was_training)
