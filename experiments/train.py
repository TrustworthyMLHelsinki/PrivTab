"""Pretrain or continue the fixed PrivTab architecture on synthetic tasks."""
import argparse
import dataclasses
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from privtab import PrivTab
from privtab.data.tabicl_data_simulator import TabularDataSimulator


def loss_fn(model, batch, smoothing):
    logits = model(batch.xc, batch.yc, batch.xt, batch.mu, batch.d)
    return torch.stack([
        F.cross_entropy(logits[i, :, :int(batch.yc[i].max()) + 1], batch.yt[i].long(),
                        label_smoothing=smoothing)
        for i in range(len(logits))
    ]).mean()


def make_scheduler(optimizer, training, batches):
    spec = training['scheduler']
    if spec['type'] == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    if spec['type'] != 'warmup_cosine_constant':
        raise ValueError('Only the warmup/cosine/constant schedule is supported.')
    # The schedule counts optimizer steps after gradient accumulation.
    steps_per_epoch = math.ceil(batches / training['accumulate_grad_batches'])
    warmup = int(training['epochs'] * steps_per_epoch * spec['warmup']['fraction'])
    end = spec['cosine']['T_max']
    minimum = spec['cosine']['eta_min'] / training['learning_rate']
    if not 0 < warmup < end:
        raise ValueError('Warmup must finish before the cosine schedule.')
    def factor(step):
        if step < warmup:
            return step / warmup
        if step >= end:
            return minimum
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * (step - warmup) / (end - warmup))) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def move_batch(batch, device):
    return dataclasses.replace(batch, **{
        field.name: getattr(batch, field.name).to(device)
        for field in dataclasses.fields(batch) if isinstance(getattr(batch, field.name), torch.Tensor)
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=Path, help='Resume a checkpoint produced by this training command.')
    parser.add_argument('--smoke-test', action='store_true', help='Two small synthetic batches and one validation batch.')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    training, data = config['training'], dict(config['data'])
    torch.manual_seed(training['seed'])
    np.random.seed(training['seed'])
    random.seed(training['seed'])
    train_tasks, validation_tasks = data.pop('train_tasks'), data.pop('validation_tasks')
    if args.smoke_test:
        data.update(batch_size=4, min_nc=16, max_nc=16, min_nt=8, max_nt=8)
        train_tasks, validation_tasks = 8, 4
        training.update(epochs=1, accumulate_grad_batches=1, validation_interval=1,
                        scheduler={'type': 'constant'})
    train_generator = TabularDataSimulator(**data, samples_per_epoch=train_tasks, deterministic_seed=training['seed'])
    validation_generator = TabularDataSimulator(**data, samples_per_epoch=validation_tasks,
                                                deterministic=True, deterministic_seed=training['seed'])
    config['data'] = dict(data, train_tasks=train_tasks, validation_tasks=validation_tasks)
    model = PrivTab(**config['model']).to(args.device)
    if training.get('initial_weights') and args.resume is None:
        model.load_state_dict(torch.load(training['initial_weights'], map_location=args.device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                  betas=tuple(training['betas']), eps=training['eps'],
                                  weight_decay=training['weight_decay'])
    scheduler = make_scheduler(optimizer, training, len(train_generator))
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start = state['epoch'] + 1
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(('MT19937', state['numpy_rng'][0].numpy().astype(np.uint32),
                             *state['numpy_rng'][1:]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    for epoch in range(start, training['epochs']):
        if epoch >= training['encoder_freeze_epoch']:
            model.encoder.requires_grad_(False)
        model.train()
        train_generator.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accumulation = training['accumulate_grad_batches']
        for index, batch in enumerate(train_generator):
            loss = loss_fn(model, move_batch(batch, args.device), training['label_smoothing'])
            group_size = min(accumulation, len(train_generator) - (index // accumulation) * accumulation)
            (loss / group_size).backward()
            total_loss += loss.item()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_generator):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip_val'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = f'epoch={epoch} train_loss={total_loss / len(train_generator):.6f}'
        if (epoch + 1) % training['validation_interval'] == 0:
            model.eval()
            validation_generator.set_epoch(epoch)
            with torch.no_grad():
                val_loss = sum(loss_fn(model, move_batch(b, args.device), training['label_smoothing']).item()
                               for b in validation_generator) / len(validation_generator)
            metrics += f' validation_loss={val_loss:.6f}'
        print(metrics, flush=True)
        if (epoch + 1) % training['checkpoint_interval'] == 0 or epoch + 1 == training['epochs']:
            numpy_rng = np.random.get_state()
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'epoch': epoch,
                     'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     'numpy_rng': (torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:])}
            torch.save(state, args.output_dir / 'last.pt')
            torch.save(model.state_dict(), args.output_dir / 'weights.pt')


if __name__ == '__main__':
    main()
