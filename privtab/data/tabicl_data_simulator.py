"""Synthetic classification tasks from the mixed SCM prior."""
from dataclasses import dataclass
import math
import random

import numpy as np
import torch

from .tabicl_prior.dataset import PriorDataset


@dataclass
class Batch:
    xc: torch.Tensor
    yc: torch.Tensor
    xt: torch.Tensor
    yt: torch.Tensor
    mu: torch.Tensor
    d: torch.Tensor


class TabularDataSimulator:
    def __init__(self, *, dim, min_nc, max_nc, min_nt, max_nt, max_classes,
                 dummy_fill_value, min_mu, max_mu, samples_per_epoch, batch_size,
                 mu_curriculum_end_epoch=None, deterministic=False, deterministic_seed=1):
        if dim != 120 or max_classes != 10 or dummy_fill_value != 0:
            raise ValueError('The prior uses 120 features, at most 10 classes, and zero padding.')
        if not 0 < min_mu <= max_mu or samples_per_epoch < batch_size:
            raise ValueError('Invalid privacy range or task count.')
        self.min_nc, self.max_nc = min_nc, max_nc
        self.min_nt, self.max_nt = min_nt, max_nt
        self.min_mu, self.max_mu = min_mu, max_mu
        self.samples_per_epoch, self.batch_size = samples_per_epoch, batch_size
        self.mu_curriculum_end_epoch = mu_curriculum_end_epoch
        self.deterministic, self.seed = deterministic, deterministic_seed
        self.epoch = 0

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        seed = self.seed if self.deterministic else self.seed + self.epoch * 1_000_003
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        for _ in range(len(self)):
            nc = int(torch.randint(self.min_nc, self.max_nc + 1, ()))
            nt = int(torch.randint(self.min_nt, self.max_nt + 1, ()))
            prior = PriorDataset(batch_size=self.batch_size, batch_size_per_gp=4,
                                 min_features=2, max_features=120, max_classes=10,
                                 min_seq_len=nc + nt, max_seq_len=nc + nt + 1,
                                 min_train_size=nc, max_train_size=nc + 1,
                                 prior_type='mix_scm', device='cpu', n_jobs=1,
                                 pre_processing_method='zscore')
            x, y, d, _, _ = prior.get_batch()
            for i, features in enumerate(d):
                x[i, :, int(features):] = 0
            minimum = self.min_mu
            end = self.mu_curriculum_end_epoch
            if end is not None and self.epoch < end:
                minimum = math.exp(math.log(self.max_mu) - self.epoch / end *
                                   (math.log(self.max_mu) - math.log(self.min_mu)))
            mu = (torch.rand(self.batch_size) * (math.log(self.max_mu) - math.log(minimum))
                  + math.log(minimum)).exp()
            yield Batch(x[:, :nc], y[:, :nc].long(), x[:, nc:], y[:, nc:].long(), mu, d.long())
