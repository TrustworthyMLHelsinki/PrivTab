"""OS-CSPRNG-backed Gaussian noise for private inference releases.

Synthetic pretraining keeps PyTorch's fast, reproducible RNG. A private release
uses fresh bytes from the operating system for every Gaussian coordinate.
"""

import math
import secrets

import numpy as np
import torch


def secure_randn_like(reference: torch.Tensor) -> torch.Tensor:
    """Sample independent standard normals from CSPRNG bytes via Box–Muller.

    Uniforms are the midpoints of 53-bit bins, so log(0) is impossible. The
    float64 result is rounded to the reference dtype after sampling on CPU.
    No seed or generator state is stored with the returned tensor.
    """
    if not reference.is_floating_point():
        raise TypeError("Gaussian noise requires a floating-point tensor.")
    count = reference.numel()
    pairs = (count + 1) // 2
    words = np.frombuffer(secrets.token_bytes(16 * pairs), dtype=np.uint64)
    uniforms = ((words >> 11).astype(np.float64) + 0.5) / (1 << 53)
    radius = np.sqrt(-2.0 * np.log(uniforms[:pairs]))
    angle = 2.0 * math.pi * uniforms[pairs:]
    samples = np.empty(2 * pairs, dtype=np.float64)
    samples[0::2] = radius * np.cos(angle)
    samples[1::2] = radius * np.sin(angle)
    return torch.from_numpy(samples[:count].copy().reshape(reference.shape)).to(
        device=reference.device, dtype=reference.dtype
    )
