"""PrivTab architecture with bounded context attention and Gaussian noise."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .secure_noise import secure_randn_like


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(256, 512)
        self.proj_b = nn.Linear(256, 512)

    def forward(self, x):
        return F.silu(self.proj_a(x)) * self.proj_b(x)


class Attention(nn.Module):
    def __init__(self, private=False):
        super().__init__()
        self.private = private
        self.num_heads = 1 if private else 4
        self.head_dim = 256 // self.num_heads
        self.to_q = nn.Linear(256, 256, bias=False)
        self.to_k = nn.Linear(256, 256, bias=False)
        self.to_v = nn.Linear(256, 256, bias=False)
        self.to_out = nn.Identity() if private else nn.Sequential(nn.Linear(256, 256), nn.Dropout(0))

    def forward(self, xq, xkv, sigma=None, normalize=False, noise_sampler=None):
        def heads(x):
            return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.to_q(xq)), heads(self.to_k(xkv)), heads(self.to_v(xkv))
        if self.private:
            v = F.normalize(v, p=2, dim=-1)
            out = torch.tanh((q @ k.transpose(-1, -2)) * self.head_dim**-0.5) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        out = out.transpose(1, 2).reshape(xq.shape[0], xq.shape[1], 256)
        if self.private:
            noise = torch.randn_like(out) if noise_sampler is None else noise_sampler(out)
            out = out + noise * sigma[:, None, None]
            if normalize:
                norm = torch.linalg.vector_norm(out, dim=-1, keepdim=True).amax(dim=1, keepdim=True)
                out = out / (norm + 1e-6) * 512
        return self.to_out(out)


class AttentionLayer(nn.Module):
    def __init__(self, private=False):
        super().__init__()
        self.attn = Attention(private)
        self.ff_block = nn.Sequential(SwiGLU(), nn.Dropout(0), nn.Linear(512, 256), nn.Dropout(0))
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)

    def forward(self, xq, xkv=None, sigma=None, normalize=False, noise_sampler=None):
        xkv = xq if xkv is None else xkv
        xq = xq + self.attn(self.norm1(xq), self.norm1(xkv), sigma, normalize, noise_sampler)
        return xq + self.ff_block(self.norm2(xq))


class Perceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(128, 256))
        self.mhsa_layers = nn.ModuleList([AttentionLayer() for _ in range(3)])
        self.mhca_ctoq_layers = nn.ModuleList([AttentionLayer(private=True) for _ in range(3)])
        self.mhca_qtot_layers = nn.ModuleList([AttentionLayer() for _ in range(3)])
        self.post_processing_mhsa_layers = nn.ModuleList([AttentionLayer() for _ in range(2)])
        self.post_processing_mhca_qtot_layers = nn.ModuleList([AttentionLayer() for _ in range(2)])

    def summarize(self, context, mu, normalize, noise_sampler=None):
        # Replace-one sensitivity 2 sqrt(H M); three adaptive GDP releases.
        sigma = 2 * math.sqrt(128 * 3) / mu
        latents = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)
        states = []
        for private, self_attention in zip(self.mhca_ctoq_layers, self.mhsa_layers):
            latents = self_attention(private(latents, context, sigma, normalize, noise_sampler))
            states.append(latents)
        for self_attention in self.post_processing_mhsa_layers:
            latents = self_attention(latents)
            states.append(latents)
        return torch.stack(states, dim=1)

    def predict(self, targets, states):
        layers = list(self.mhca_qtot_layers) + list(self.post_processing_mhca_qtot_layers)
        for index, layer in enumerate(layers):
            targets = layer(targets, states[:, index])
        return targets


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, 512),
                                 nn.GELU(), nn.Linear(512, output_dim))

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, include_context_encoder=True):
        super().__init__()
        self.naive_pad_token = nn.Parameter(torch.tensor(0.0))
        if include_context_encoder:
            self.transformer_encoder = Perceiver()
        self.xy_encoder = MLP(250, 256)
        self.y_encoder = nn.Embedding(11, 10)

    def encode(self, x, labels, d):
        valid = torch.arange(120, device=x.device)[None, :] < d[:, None]
        values = torch.where(valid[:, None, :], x, self.naive_pad_token.to(x.dtype))
        mask = valid[:, None, :].expand(-1, x.shape[1], -1).to(x.dtype)
        features = torch.cat((values, mask), dim=-1) * (120 / d.to(x.dtype)).sqrt()[:, None, None]
        return self.xy_encoder(torch.cat((features, self.y_encoder(labels.long())), dim=-1))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_decoder = MLP(256, 10)

    def forward(self, x):
        return self.z_decoder(x)


class PrivTab(nn.Module):
    """Fixed 120-feature, 10-class classifier with a reusable private summary.

Inputs are already transformed using public or separately privatized statistics.
Feature counts and the class vocabulary must be public. Fresh calls to
``summarize`` compose; repeated ``predict_from_summary`` calls do not.
"""

    def __init__(self, normalize_perturbed_output=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.normalize_perturbed_output = normalize_perturbed_output

    @staticmethod
    def _features(x, d):
        if x.ndim != 3 or not 1 <= x.shape[-1] <= 120 or x.shape[0] == 0:
            raise ValueError("Features must have shape [batch, rows, features], with 1–120 features.")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Features must be finite floating-point values.")
        d = torch.as_tensor(x.shape[-1] if d is None else d, device=x.device)
        if d.ndim == 0:
            d = d.expand(x.shape[0])
        if d.shape != (x.shape[0],) or not torch.isfinite(d).all() or (d != d.long()).any() or ((d < 1) | (d > x.shape[-1])).any():
            raise ValueError("d must contain one valid integer feature count per task.")
        return F.pad(x, (0, 120 - x.shape[-1])), d.long()

    def summarize(self, xc, yc, mu, d=None, normalize_perturbed_output=None, noise_sampler=None):
        xc, d = self._features(xc, d)
        if xc.shape[1] == 0 or yc.shape != xc.shape[:2] or not torch.isfinite(yc).all() or (yc != yc.long()).any() or ((yc < 0) | (yc >= 10)).any():
            raise ValueError("Context labels must be integer class IDs 0–9, with at least one context row.")
        mu = torch.as_tensor(mu, device=xc.device, dtype=xc.dtype)
        if mu.ndim == 0:
            mu = mu.expand(xc.shape[0])
        if mu.shape != (xc.shape[0],) or not torch.isfinite(mu).all() or (mu <= 0).any():
            raise ValueError("mu must be finite and positive, with one value per task.")
        normalize = self.normalize_perturbed_output if normalize_perturbed_output is None else normalize_perturbed_output
        context = self.encoder.encode(xc, yc, d)
        return self.encoder.transformer_encoder.summarize(context, mu, normalize, noise_sampler)

    @torch.inference_mode()
    def release_summary(self, xc, yc, mu, d=None, normalize_perturbed_output=True):
        """Create one private release using fresh OS-CSPRNG Gaussian noise.

        Training and public benchmark evaluation continue to use ``summarize``.
        Each call here is a separate release and must be privacy-accounted.
        """
        was_training = self.training
        self.eval()
        try:
            return self.summarize(xc, yc, mu, d, normalize_perturbed_output,
                                  noise_sampler=secure_randn_like)
        finally:
            self.train(was_training)

    def predict_from_summary(self, summary, xt, d=None):
        xt, d = self._features(xt, d)
        if summary.shape != (xt.shape[0], 5, 128, 256):
            raise ValueError("Summary must have shape [batch, 5, 128, 256].")
        labels = torch.full(xt.shape[:2], 10, dtype=torch.long, device=xt.device)
        targets = self.encoder.encode(xt, labels, d)
        return self.decoder(self.encoder.transformer_encoder.predict(targets, summary))

    def forward(self, xc, yc, xt, mu, d=None, normalize_perturbed_output=None):
        if xc.shape[-1] != xt.shape[-1]:
            raise ValueError("Context and target feature widths must agree.")
        summary = self.summarize(xc, yc, mu, d, normalize_perturbed_output)
        return self.predict_from_summary(summary, xt, d)
