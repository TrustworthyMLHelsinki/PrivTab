from pathlib import Path

import pytest
import torch
import yaml

from experiments.train import make_scheduler


def test_stage1_schedule_matches_optimizer_step_counts():
    config = yaml.safe_load(Path('experiments/configs/models/privtab_stage1.yml').read_text())
    training = config['training']
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=training['learning_rate'])
    schedule = make_scheduler(optimizer, training, batches=1024)
    factor = schedule.lr_lambdas[0]
    assert factor(0) == 0
    assert factor(6400) == pytest.approx(1.0)
    assert factor(51200) == pytest.approx(0.1)
    assert factor(128000) == pytest.approx(0.1)
