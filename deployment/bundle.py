"""Tensor-only deployment format, with no context data or private random seed."""
from pathlib import Path

import torch
from torch import nn

from privtab.model import AttentionLayer, Decoder, Encoder, PrivTab
from privtab.release import ReleasedSummary


class Predictor(nn.Module):
    """Only row encoding, target-to-summary attention, and the output decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(include_context_encoder=False)
        self.layers = nn.ModuleList([AttentionLayer() for _ in range(5)])
        self.decoder = Decoder()
        self.register_buffer("summary", torch.empty(1, 5, 128, 256))
        self.num_features = 120
        self.num_classes = 10

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.num_features:
            raise ValueError(f"Expected [rows, {self.num_features}] preprocessed features.")
        x, d = PrivTab._features(x.unsqueeze(0), None)
        labels = torch.full(x.shape[:2], 10, dtype=torch.long, device=x.device)
        targets = self.encoder.encode(x, labels, d)
        for index, layer in enumerate(self.layers):
            targets = layer(targets, self.summary[:, index])
        return self.decoder(targets)[0, :, :self.num_classes]

    @torch.inference_mode()
    def predict_proba(self, x, batch_size=1024):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if x.ndim != 2 or x.shape[-1] != self.num_features:
            raise ValueError(f"Expected [rows, {self.num_features}] preprocessed features.")
        device = self.summary.device
        outputs = [self(chunk.to(device)).softmax(-1).cpu() for chunk in x.split(batch_size)]
        return torch.cat(outputs)


@torch.inference_mode()
def export_bundle(model, xc, yc, mu, num_classes, path):
    """Release one task; class count and feature schema must be public.

    xc: [context rows, features], already using a fixed public or DP transform.
    No data-dependent preprocessing is fitted here. Re-exporting consumes a new
    mu-GDP budget; loading and querying the same bundle is post-processing.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {path}")
    release = ReleasedSummary.create(model, xc, yc, mu, num_classes)
    export_bundle_from_summary(model, release, path)


@torch.inference_mode()
def export_bundle_from_summary(model, release: ReleasedSummary, path):
    """Save a prediction-only bundle from an existing release without new noise."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {path}")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        predictor = Predictor()
        encoder_weights = {k: v for k, v in model.encoder.state_dict().items()
                           if not k.startswith("transformer_encoder.")}
        predictor.encoder.load_state_dict(encoder_weights, strict=True)
        transformer = model.encoder.transformer_encoder
        for destination, source in zip(predictor.layers, list(transformer.mhca_qtot_layers)
                                       + list(transformer.post_processing_mhca_qtot_layers)):
            destination.load_state_dict(source.state_dict(), strict=True)
        predictor.decoder.load_state_dict(model.decoder.state_dict(), strict=True)
        predictor.summary.copy_(release.tensor.cpu())
        bundle = {"format_version": 1, "mu": release.mu, "num_classes": release.num_classes,
                  "num_features": release.num_features, "state_dict": predictor.state_dict()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            torch.save(bundle, output)
    finally:
        model.train(was_training)


def load_bundle(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["format_version"] != 1:
        raise ValueError("Unsupported deployment bundle version.")
    model = Predictor()
    model.num_features = bundle["num_features"]
    model.num_classes = bundle["num_classes"]
    if not 1 <= model.num_features <= 120 or not 2 <= model.num_classes <= 10:
        raise ValueError("Invalid feature or class count in bundle.")
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model.to(device).eval()
