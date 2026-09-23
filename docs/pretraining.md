# Synthetic pretraining and continuations

PrivTab is prior-fitted on synthetic tabular classification tasks before it sees any real benchmark context. A training example is a whole small dataset, split into labeled **context** rows and held-out **target** rows. The loss asks the model to predict target labels from the context. The three supplied checkpoints share the [same architecture](architecture.md); the YAML configurations change the task sizes and training schedule.

The implementation is [`experiments/train.py`](../experiments/train.py) with the [`TabularDataSimulator`](../privtab/data/tabicl_data_simulator.py). The synthetic prior is implemented under [`privtab/data/tabicl_prior`](../privtab/data/tabicl_prior/).

## Task generation

For each batch, the simulator samples an integer context length `Nc` and target length `Nt` uniformly from the configured **inclusive** ranges. All 32 tasks in that batch use those lengths, but each task has its own generated features, labels, active feature count, and privacy parameter `mu`.

The `mix_scm` prior generates datasets from MLP-based and tree-based structural causal models (70% and 30% respectively). It samples feature/label relationships and converts the generated response to a classification problem with 2–10 classes. It applies its synthetic-data z-score preprocessing, pads feature tensors to width 120 with zeros, and records each task's active feature count `d`. The model then replaces padded positions with its learned pad token and appends a valid-feature mask, as described in the [architecture guide](architecture.md#row-encoder).

Each yielded batch has these shapes:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `xc` | `[32, Nc, 120]` | Context features |
| `yc` | `[32, Nc]` | Visible context class IDs |
| `xt` | `[32, Nt, 120]` | Target features |
| `yt` | `[32, Nt]` | Target labels used only by the loss |
| `d` | `[32]` | Active feature count for each task |
| `mu` | `[32]` | Privacy level sampled for each task |

The generator uses the `mix_scm` prior on CPU with one data-generation job. It creates 32,768 training tasks and 8,192 validation tasks per epoch: 1,024 training batches and 256 validation batches at batch size 32. The validation generator uses a fixed seed; training batches change with the epoch. These are simulated data, so synthetic-pretraining noise uses PyTorch's random generator. Releasing a summary from private data instead uses the [secure deployment path](architecture.md#decoder-and-deployment).

## Checkpoint stages and context lengths

| Stage | Configuration | Start weights | Context rows `Nc` | Target rows `Nt` | Epochs | Noisy-output normalization during training |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Stage 1 | [`privtab_stage1.yml`](../experiments/configs/models/privtab_stage1.yml) | Random initialization | 100–2,048 | 100–2,048 | 500 | Off |
| Short-context continuation | [`privtab_stage2_short_contexts.yml`](../experiments/configs/models/privtab_stage2_short_contexts.yml) | Stage 1 | 100–2,048 | 100–2,048 | 1,000 | Off |
| Long-context continuation | [`privtab_stage2_long_contexts.yml`](../experiments/configs/models/privtab_stage2_long_contexts.yml) | Stage 1 | 1,024–8,192 | 100–2,048 | 1,000 | On |

Both continuations load `models/stage1/weights.pt` independently. The long-context run does **not** start from the short-context continuation. All stages use a maximum of 120 features and 10 classes. The optional normalization acts on the *already-noisy private attention update*, not on raw dataset features; see [its formula](architecture.md#optional-normalization-of-the-noisy-latent-update).

## Privacy-parameter sampling and stage-1 curriculum

`mu` controls the Gaussian noise level in the three private context-to-latent layers: smaller `mu` produces more noise. Each task draws its own `mu` **uniformly in log space** between the current lower bound and 64:

```text
u ~ Uniform(0, 1)
mu = exp(log(mu_min(epoch)) + u * (log(64) - log(mu_min(epoch))))
```

Only stage 1 has a curriculum. For zero-based epoch `e`, its lower bound is

```text
mu_min(e) = exp(log(64) - (e / 150) * (log(64) - log(0.16)))   for 0 <= e < 150
mu_min(e) = 0.16                                                for e >= 150
```

At epoch 0, `mu_min = mu_max = 64`, so every task uses `mu = 64` (little noise). As the lower bound falls, training progressively includes harder, noisier tasks. From epoch 150 onward it samples log-uniformly over `0.16–64`. Both continuation configurations omit the curriculum and use the full `0.16–64` range from their first epoch. The training generator supplies `mu` directly; it does not sample an `(epsilon, delta)` pair.

## Loss, optimization, and checkpoints

The model produces ten logits per target row. For each synthetic task, training restricts them to the class count inferred from its context labels, computes target cross-entropy with label smoothing `0.1`, and averages the task losses. There is no gradient-based adaptation on a real evaluation dataset.

All stages use AdamW with `betas=(0.9, 0.98)`, `eps=1e-8`, weight decay `0.001`, gradient clipping at norm `1.0`, and accumulation over four batches. With 32 tasks per batch, that is an effective batch of 128 tasks and 256 optimizer steps per epoch.

| Stage | Initial learning rate | Schedule | Encoder frozen from epoch |
| --- | ---: | --- | ---: |
| Stage 1 | `1e-4` | Warm up for 5% of training (6,400 optimizer steps), cosine decay to `1e-5` by step 51,200, then constant | 470 |
| Short-context continuation | `1e-5` | Constant | 950 |
| Long-context continuation | `1e-5` | Constant | 950 |

At the freeze epoch, the row encoder and Perceiver stop receiving gradients; the decoder remains trainable. Every run saves `last.pt` with optimizer, scheduler, epoch, and random-generator state for `--resume`; it also saves tensor-only `weights.pt` and the effective `config.yml`. The training runner uses one device.

To start a training run, use the commands in the [README](../README.md#synthetic-training). The supplied checkpoint files already contain the pretrained weights; retraining is optional for using PrivTab.
