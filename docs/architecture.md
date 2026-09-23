# PrivTab architecture

PrivTab is a tabular classifier trained on synthetic classification tasks. At inference, it compresses a labeled context dataset into a differentially private summary, then classifies any number of query rows from that summary. Inference does not update the model weights.

The implementation is [the model module](../privtab/model.py). The three supplied checkpoints have identical layers; their training context lengths and output-normalization settings differ. The task generator, `mu` curriculum, and optimizer settings are described in the [pretraining guide](pretraining.md).

## Data flow

```text
context features [B, Nc, D] + context labels [B, Nc]
    -> row encoder -> context tokens [B, Nc, 256]
    -> 3 noisy context-to-latent layers, each followed by latent self-attention
    -> 2 further latent self-attention layers
    -> five saved latent states [B, 5, 128, 256]  (the DP summary)

query features [B, Nt, D] + masked labels
    -> the same row encoder -> query tokens [B, Nt, 256]
    -> 5 query-to-state attention layers, one for each saved state
    -> decoder -> logits [B, Nt, 10]
    -> first C logits + softmax -> probabilities [B, Nt, C]
```

Here `B` is the number of independent tasks, `Nc` and `Nt` are context and query row counts, `1 <= D <= 120` is the public number of features, and `2 <= C <= 10` is the public number of classes. Class IDs are `0, ..., C-1`. The model supports different context and query row counts, but both sides of a task must use the same feature order and transform.

## Row encoder

Features are padded to 120 columns. For a task with `D` real columns, the other positions receive a *learned scalar pad token*. A 120-element valid-column mask is appended to every row. The resulting 240 values are multiplied by `sqrt(120 / D)`. `D` and the mask must describe a public schema; inferring them from confidential rows would require separate privacy accounting.

Context rows use an embedding of their class ID; query rows use the extra mask ID `10`. The label embedding has width 10. Concatenating it with the 240 feature values gives a 250-dimensional input to the shared row MLP:

```text
Linear(250, 512) -> GELU -> Linear(512, 512) -> GELU -> Linear(512, 256)
```

Each row is encoded independently before the private attention. The code does not compute cross-row feature statistics in this encoder. Inputs should already be transformed with a fixed public transform or a separately privatized one.

## Latent summary layers

The Perceiver begins with 128 learned latent vectors of width 256, shared across tasks. Its three private layers each perform a context-to-latent attention update, followed by latent self-attention. Two more latent self-attention layers follow. A copy of the latent state is retained after each of these five stages, giving the summary shape `[B, 5, 128, 256]`.

| Layer group | Count | Queries | Keys and values | Attention heads | Reads context rows? |
| --- | ---: | --- | --- | ---: | --- |
| Private context-to-latent | 3 | 128 latents | Encoded context rows | 1 × 256 | Yes |
| Latent self-attention | 3, after private updates | Latents | Latents | 4 × 64 | No |
| Additional latent self-attention | 2 | Latents | Latents | 4 × 64 | No |
| Query-to-state cross-attention | 5 | Encoded query rows | One saved latent state per layer | 4 × 64 | No |

Every attention layer uses residual connections and pre-layer normalization. Its feed-forward branch is `SwiGLU(256 -> 512) -> Linear(512, 256)`, with another residual connection. Dropout is zero. Non-private attention uses standard scaled dot-product softmax attention. The model's query path runs after summary construction; it does not need context rows or labels.

The query path applies its five cross-attention layers sequentially to the same query tokens, using summary states 1 through 5 in order. Keeping only the final latent state would therefore change the trained predictor.

## Private context-to-latent attention

For each private layer, linear projections form latent queries `q` and context keys `k` and values `v`. Each value is normalized to at most unit Euclidean norm. With one attention head and 256-dimensional head width, the aggregation is

```text
A(q, k, v) = tanh(q k^T / sqrt(256)) v
```

The `tanh` score lies in `[-1, 1]`; this is a bounded sum over context rows, not softmax attention. The [Lean tanh-attention theorem](../privtab-lean/PrivTabLean/TanhAttention.lean) proves, for this real-valued scaled-tanh and epsilon-normalized-value formula, that replacing one context row changes each of the 128 output vectors by at most 2 in Euclidean norm for fixed queries and row-wise projections. The concatenated output therefore has replace-one sensitivity at most `2 sqrt(128)` per private layer. Later latent queries can depend on earlier *noisy* states; the three layers are composed adaptively.

For a requested total privacy parameter `mu`, independent Gaussian noise with standard deviation

```text
sigma = 2 sqrt(128) sqrt(3) / mu
```

is added to every output coordinate at each of the three private layers. This allocates `mu / sqrt(3)` GDP to each layer, giving total `mu` under adaptive GDP composition. Smaller `mu` means more noise.

The private boundary assumes the row encoder applies independently to each row, the model weights are independent of the private context, and feature preprocessing, feature count, and class vocabulary are public or separately privatized. The [row-wise and noise audits](../README.md#privacy-implementation-audits) check the Python implementation alongside the mathematical proof; they do not account for external preprocessing or establish a bit-level floating-point guarantee.

The [Lean proof-to-code map](lean_code_map.md) traces this bounded attention sum, its noise calibration, and prediction post-processing to the formal definitions and theorems. It also identifies which properties of the Python release path the Lean project assumes.

## Optional normalization of the noisy latent update

PrivTab has an optional normalization for the output of **each private attention layer**. It acts after Gaussian noise is added and before the residual connection and feed-forward block. It does not normalize the raw context rows or apply a single normalization to the final five-state summary.

For one task, let `u[m, :]` be the noisy attention output for latent token `m`, with `m = 1, ..., 128`. The implementation computes

```text
r = max_m ||u[m, :]||_2
u_normalized[m, :] = 512 * u[m, :] / (r + 1e-6)
```

The same scale is used for all 128 tokens in that task; different tasks in a batch are scaled independently. This stabilizes the magnitude of the noisy update across privacy levels and context sizes. Because normalization uses only the already-noisy output, it is post-processing and does not add a privacy release. It is distinct from the pre-layer `LayerNorm` inside attention blocks and from dataset feature standardization.

The switch is `normalize_perturbed_output`. Stage 1 and the short-context continuation trained with it off; the long-context continuation trained with it on. [`load_model`](../privtab/checkpoints.py) enables it for evaluation with any supplied checkpoint, and `release_summary` also defaults to on for private deployment. A caller can explicitly override it through the model API. The saved summary contains the resulting latent states, so inference does not run this normalization again.

## Decoder and deployment

The decoder maps each final 256-dimensional query token to ten logits:

```text
Linear(256, 512) -> GELU -> Linear(512, 512) -> GELU -> Linear(512, 10)
```

Only the first `C` logits are used for a task with `C` public classes. Softmax produces class probabilities. The same released summary can answer more queries without another privacy release. Generating a **new** summary, even for the same context, spends privacy budget again.

For deployment, [`ReleasedSummary.create`](../privtab/release.py) calls `release_summary`, which samples fresh Gaussian noise from an OS-backed cryptographic RNG and can save the five-state tensor. The [prediction bundle](../deployment/bundle.py) contains that tensor, public schema, row encoder, five query-to-state attention layers, and decoder. It excludes context rows, labels, context-to-latent layers, and noise seeds. The lower-level `summarize` and `forward` methods use PyTorch randomness for synthetic training and public benchmark evaluation; private-data releases should use `release_summary` or `ReleasedSummary.create`.

## Supplied checkpoints

| Checkpoint | Synthetic training context rows | Output normalization during training |
| --- | ---: | --- |
| [`stage1`](../models/stage1/weights.pt) | 100–2,048 | Off |
| [`stage2_short_contexts`](../models/stage2_short_contexts/weights.pt) | 100–2,048 | Off |
| [`stage2_long_contexts`](../models/stage2_long_contexts/weights.pt) | 1,024–8,192 | On |

Both stage-2 checkpoints start from stage 1. The supplied inference loader enables output normalization for all three, matching the evaluation path. See the [pretraining guide](pretraining.md) for the full training process and the [README](../README.md#public-benchmark-evaluation-and-dp-hpo) for benchmark evaluation.
