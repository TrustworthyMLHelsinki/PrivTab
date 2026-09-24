# PrivTab

PrivTab is a tabular classifier that compresses a labeled context dataset into a reusable differentially private summary. Predictions on new rows use that summary without accessing the context again.

This repository includes three PrivTab checkpoints, a synthetic task generator for pretraining, and two DP-SGD baselines: DP MLP and DP Logistic Regression with private learning-rate selection. It also provides TabArena evaluation, ELO ranking, and end-to-end case studies.

For layer descriptions and the private summary boundary, see the [architecture guide](docs/architecture.md). The [pretraining guide](docs/pretraining.md) describes synthetic tasks, context lengths, and the `mu` curriculum.

## Install

Python 3.11+ and PyTorch 2.6+ are required. Make sure you first run:
```bash
pip install --upgrade pip setuptools wheel
```
Then from this directory:
```bash
pip install -e .                         # model and deployment only
pip install -e '.[experiments,test]'     # training, benchmarks, tests
```

Inference needs only PyTorch and NumPy. Third-party source attributions are in `LICENSE` and the prior/accountant modules.

## Included checkpoints

| Local weights | Training configuration |
| ---  | --- |
| `models/stage1/weights.pt` | `experiments/configs/models/privtab_stage1.yml` |
| `models/stage2_short_contexts/weights.pt`  | `experiments/configs/models/privtab_stage2_short_contexts.yml` |
| `models/stage2_long_contexts/weights.pt` | `experiments/configs/models/privtab_stage2_long_contexts.yml` |

Both stage-2 checkpoints start from stage 1. Choose the checkpoint explicitly for deployment; benchmark evaluation can route by public context size.

The three tensor-only `weights.pt` files are included in the repository. Each contains the model parameters without optimizer state.

## Digits playground

The [digits playground](examples/digits_playground.py) runs a 0-versus-1 example using scikit-learn's bundled 8×8 digits dataset. Its [documented pixel range](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) is 0–16, so the script clips to those public bounds and maps them to −1–1 without fitting any statistic on the context. It makes one secure summary release, then reuses that summary for several predictions. The bundled dataset is public; this demonstrates the workflow without claiming its test queries are private.

```bash
pip install -e . scikit-learn
python -m examples.digits_playground --mu 0.4
python -m examples.digits_playground --mu 0.4 --summary releases/digits.pt
```

The optional `--summary` saves and reloads the release, and refuses to overwrite an existing file. Each invocation creates a fresh summary and spends a new privacy budget if used with real private context. For real data, keep the feature bounds and class vocabulary public or account separately for how they were obtained; do not publish accuracy or other metrics computed from private query labels without accounting for that release.

## Export once, predict separately

Prepare a numerical `context.npz` containing `x` of shape `[context_rows, features]` and integer `y` of shape `[context_rows]`. There may be 1–120 features and 2–10 public classes, encoded as `0, …, C-1`. Inputs must be finite. Use the same column order and public or separately privatized preprocessing for context and queries.

On the data owner's machine:

```bash
python -m deployment.export \
  --weights models/stage2_short_contexts/weights.pt \
  --context context.npz --mu 0.4 --num-classes 2 \
  --output releases/task.pt
```

Copy `releases/task.pt` to the prediction machine. Export uses fresh OS-CSPRNG-backed Gaussian noise for each private attention layer. The bundle contains the noisy summary, predictor weights, and public feature/class counts. It contains no context rows, labels, private preprocessing statistics, optimizer state, or noise seeds. The context encoder's private attention weights are also omitted.

On the prediction machine, provide `queries.npy` with shape `[query_rows, features]`:

```bash
python -m deployment.predict \
  --bundle releases/task.pt --features queries.npy \
  --output probabilities.npy --batch-size 1024
```

The output has shape `[query_rows, C]`. Repeated predictions and changing query batch sizes reuse the same release. Export refuses to overwrite an existing bundle to help avoid accidental re-release.

The Python library can save the summary separately. `ReleasedSummary.create` spends the privacy budget once; loading it and predicting from it do not create another release. Use the same checkpoint weights when loading a standalone summary:

```python
from privtab import ReleasedSummary
from privtab.checkpoints import load_model

model = load_model("models/stage2_short_contexts/weights.pt")
# Preprocessed tensors: xc [context_rows, features],
# yc [context_rows], xt [query_rows, features].
release = ReleasedSummary.create(model, xc, yc, mu=0.4, num_classes=2)
release.save("summary.pt")

saved = ReleasedSummary.load("summary.pt")
probabilities = saved.predict_proba(model, xt)
```

To create a self-contained prediction-only bundle from the *same* saved summary, without spending privacy budget again:

```python
from deployment import export_bundle_from_summary, load_bundle

export_bundle_from_summary(model, saved, "predictor.pt")
predictor = load_bundle("predictor.pt")
probabilities = predictor.predict_proba(xt)
```

The prediction machine can load `predictor.pt` without the full PrivTab checkpoint. Use `model.release_summary(xc[None], yc[None], mu)` if a raw summary tensor is needed in memory; this is also CSPRNG-backed. The lower-level `model.summarize(...)` and `model.forward(...)` use PyTorch randomness for synthetic pretraining and public benchmark evaluation. Do not use those lower-level methods to release private-context outputs.

The summary has shape `[B, 5, 128, 256]`. Three private attention layers are followed by two post-processing layers. Targets attend to the intermediate latent states at every layer, so retaining only the final state would change the trained predictor. The five-state summary is 640 KiB per task in float32.

For replace-one row adjacency, each private layer uses tanh-bounded attention, unit-normalized values, and Gaussian standard deviation `2 * sqrt(128 * 3) / mu`. The three adaptive releases compose to the supplied `mu`. Query prediction is post-processing. A fresh summary is a fresh privacy release, even for the same dataset; repeated releases compose as `sqrt(sum(mu_i**2))`.

The model does **not** fit preprocessing on private data. Empirical context means, standard deviations, class vocabularies, or feature selection are not automatically covered by its guarantee. Supply a public schema and a fixed public transform, or account separately for private preprocessing. The exported bundle expects already transformed queries. Private releases use `secrets.token_bytes` and Box–Muller to generate Gaussian noise from the OS CSPRNG; they do not accept or save a reproducible noise seed. This is a floating-point implementation of the ideal Gaussian mechanism, so an exact bit-level DP proof is separate from the real-valued analysis. Track the privacy budget across every summary or other private-data release, including repeated exports and any model or parameter selection based on private data.

## Privacy implementation audits

The row-wise and empirical DP-MHCA audits check the PrivTab implementation. Install `pip install -e '.[audits]'`, then run:

```bash
python -m experiments.audit_rowwise \
  --weights models/stage2_short_contexts/weights.pt \
  --output runs/audits/rowwise.json

python -m experiments.dp_audit \
  --weights models/stage2_short_contexts/weights.pt \
  --mu-values 0.05 0.1 0.2 0.4 0.8 1.6 \
  --steps 200 --samples 512 --output-dir runs/audits/mhca
```

Run the same commands with `models/stage2_long_contexts/weights.pt` to audit the other continuation. `audit_rowwise` hooks the actual first private-attention boundary and checks gradients for cross-row, target-to-context, and initial-query dependencies; it also changes one integer label to check for effects on other rows. `dp_audit` searches for a high-sensitivity replace-one context-token pair in one checkpoint layer, checks the sigma passed to all three private layers, samples the selected layer's actual noise path (`--rng secure` by default), and writes a JSON report plus per-μ ROC data and PDF plots. The NPZ data and plots compare a known-mean likelihood-ratio test with the bound μ/√3 for one layer. Increase `--samples` for smoother curves; `--layer-index 0|1|2` selects the layer. The Lean theorem below proves the universal bound for the real-valued attention formula. These empirical audits complement it by checking the Python path, including noise scale; they do not certify floating-point execution or private preprocessing.

## Lean privacy proofs

The [PrivTab Lean project](privtab-lean/README.md) proves the universal `2 * sqrt(H * m)` replace-one sensitivity bound for the exact real-valued tanh attention and epsilon-normalized value formula used by PrivTab, with arbitrary fixed row-wise projections. It also formalizes adaptive μ-GDP composition of private encoder layers and privacy preservation through prediction. The [proof-to-code map](docs/lean_code_map.md) explains every Lean definition, lemma, and theorem and identifies the corresponding Python operation.

| Lean result | PrivTab operation |
| --- | --- |
| `tanhAttention_hasL2Sensitivity`: the concrete tanh/normalized-value sum has sensitivity `2 * sqrt(H * m)` | `Attention.forward` computes the scaled `tanh` scores and epsilon-normalized values; here `H=1`, `m=128` |
| Three adaptive private layers compose to `mu` | `Perceiver.summarize` uses noise standard deviation `2 * sqrt(128 * 3) / mu` in each layer |
| Prediction is post-processing | `predict_from_summary` reads the released five-state summary without context rows |

The proof audit pins the statements, rejects proof shortcuts, checks the Lean kernel's axiom dependencies, and builds the project:

```bash
cd privtab-lean
python scripts/audit_lean.py
```

Lean and Lake are required for the proof audit. The formalization assumes the Gaussian-mechanism and GDP composition rules as explicit interfaces. It does not prove a concrete correspondence between Lean states and Python tensors, or verify the floating-point sampler or private-data preprocessing. The empirical audits above test parts of that correspondence.

## Synthetic training

The architecture is defined in `privtab/model.py`, with training settings in three YAML files. Stage 1 trains with contexts of 100–2048 rows. The two continuations use 100–2048 and 1024–8192 respectively. Output normalization is enabled during deployment for all checkpoints, and during long-context continuation training. The [pretraining guide](docs/pretraining.md) gives the exact `mu` curriculum and stage settings.

```bash
python -m experiments.train \
  --config experiments/configs/models/privtab_stage1.yml \
  --output-dir runs/stage1 --device cuda

# Small CPU validation of data generation, training, and checkpoint writing:
OMP_NUM_THREADS=2 python -m experiments.train \
  --config experiments/configs/models/privtab_stage1.yml \
  --output-dir runs/smoke --smoke-test
```

Continuations initialize from `models/stage1/weights.pt`; change `training.initial_weights` to use your own stage-1 weights. Each run saves `last.pt` for `--resume`, tensor-only `weights.pt`, and its configuration. The single-device runner uses AdamW, gradient accumulation, encoder freezing, and the schedules specified in the YAML files.

## Public benchmark evaluation and DP HPO

For the 33 TabArena v0.1 datasets, create the cache from OpenML. The command applies class encoding, categorical ordinal encoding, numeric imputation, a constant-column filter, and limits of 100,000 rows, 120 features, and 10 classes:

```bash
pip install -e '.[experiments,case-studies]'
python -m experiments.prepare_tabarena --output runs/tabarena_datasets.pickle
```

The commands below accept a dataset cache containing `(dataset_name, (x, y))` pairs. A directory of prepared numerical NPZ datasets can instead be cached with `python -m experiments.prepare_data --input-dir data --output runs/datasets.pickle`.

Run all three methods on the same saved 80/20 splits:

```bash
python -m experiments.evaluate \
  --datasets-cache runs/tabarena_datasets.pickle \
  --indices-dir runs/indices \
  --output runs/privtab-results.json --device cuda

python -m experiments.dp_hpo \
  --datasets-cache runs/tabarena_datasets.pickle \
  --indices-dir runs/indices \
  --output-dir runs/baselines \
  --models dp_mlp dp_logistic_regression
```

`experiments.evaluate` selects the short-context checkpoint below 4,096 context rows and the long-context checkpoint at or above 4,096. It samples one DP summary per dataset, repeat, and privacy level, then batches target queries from that same summary. It writes metrics, repeat IDs, checkpoint paths, and context sizes to JSON. Existing split files are validated and reused; missing ones are generated with the baseline’s split code. Use `--dataset NAME --repeats 1` for a small run, or `--weights PATH` to evaluate a single checkpoint explicitly.

Only load trusted joblib caches. Both benchmark runners share saved outer 80/20 splits. Use the same `--indices-dir` for comparisons. These are **public research benchmark commands**: their empirical context standardization and label-aware outer splits are not an end-to-end private preprocessing pipeline. For private deployments, use the model/baseline functions on already transformed arrays and supply a public class count.

The DP HPO baselines fix all hyperparameters except the learning rate:

| Baseline | Learning-rate grid | Gradient norm | Sampling probability | Epochs |
| --- | --- | --- | --- | --- |
| DP Logistic Regression | 0.005, 0.02, 0.05, 0.1 | 0.1 | 0.05 | 1000 |
| DP MLP (one hidden layer, width 64) | 0.0003, 0.001, 0.003, 0.01 | 5.0 | 0.05 | 250 |

The context is partitioned into subtraining/validation, with 16% reserved for validation. With `K` candidates, selection receives `mu_selection = 0.5 * mu_total`; each candidate's training and validation receives `mu_selection / sqrt(K)`. Their disjoint, label-independent partitions compose in parallel. Final training uses the full context and `sqrt(mu_total**2 - mu_selection**2)`. Selection minimizes a noisy mean cross-entropy, clipping individual losses at 5 before adding Gaussian noise. The numerical replace-one DP-SGD accountant uses PLD discretization and GDP-conversion tolerances.

The inner split is label-independent, DP noise uses fresh randomness, and empty Poisson batches receive Gaussian updates. The replace-one accountant receives the clipping radius; Google's `REPLACE_ONE` PLD internally compares contributions at `+C` and `-C`, accounting for their `2C` difference. The public vocabulary is explicit in the training and selection functions.

## ELO ranking

The ELO implementation in `experiments/bencheval` ranks saved PrivTab and DP HPO results for each privacy level and pooled across levels:

```bash
python -m experiments.eval_tabular_with_ranking \
  --privtab-results runs/privtab-results.json \
  --baseline-results runs/baselines/tabarena-dp-lr-grid-all-results.pickle \
  --output-dir runs/rankings
```

To include a non-DP Logistic Regression reference as a fourth ELO opponent, pass `--reference-results PATH` with its saved result pickle. The reference is ranked from saved metrics. If running fewer than 33 datasets, set `--expected-datasets N` accordingly.

Binary tasks use `1 - AUC`; multiclass tasks use log loss. DP Logistic Regression is calibrated to ELO 1000. CSV files contain per-mu and pooled rankings with task-bootstrap confidence intervals. Comparisons reject missing methods or mismatched repeat identifiers. The ranking command reads results; it does not train models.

## End-to-end case studies

The case-study workflow includes dataset loaders, public-bound inference and caches, DP preprocessing comparisons, CSV/Markdown reports, and ELO ranking. Public descriptions and cached bounds are in `experiments/prompts_cache` and `experiments/bounds_cache`. Cached-bound evaluation does not call an LLM; optional regeneration requires `--allow-llm-bounds-refresh`, the `bounds-refresh` extra, and credentials supplied through the environment.

To download the public case datasets and create a reusable cache:

```bash
pip install -e '.[experiments,case-studies]'
python -m experiments.end_to_end_cases \
  --case-datasets-cache runs/case-datasets.pickle \
  --initialize-case-datasets-cache
```

Then evaluate PrivTab and both DP HPO baselines using the same saved splits:

```bash
python -m experiments.end_to_end_cases \
  --case-datasets-cache runs/case-datasets.pickle \
  --indices-dir runs/end_to_end_indices \
  --output runs/end_to_end_cases_privtab.pickle

python -m experiments.run_end_to_end_dp_baselines \
  --case-datasets-cache runs/case-datasets.pickle \
  --indices-dir runs/end_to_end_indices \
  --output-dir runs/end_to_end_dp_baselines \
  --models dp_mlp dp_logistic_regression

python -m experiments.end_to_end_cases_elo \
  --privtab-results runs/end_to_end_cases_privtab.pickle \
  --baselines-dir runs/end_to_end_dp_baselines \
  --output runs/end_to_end_cases_elo.csv
```

Use `--dataset pima_diabetes` and `--repeats 1` to limit a run. The PrivTab case-study runner uses the short-context checkpoint below 4,096 context rows and the long-context checkpoint at or above 4,096. Both weights paths and the threshold are configurable. This routing is a public benchmark convention, separate from the deployment export command’s explicit checkpoint choice.

By default, PrivTab compares exact z-score normalization, private z-score normalization with oracle bounds, and private z-score normalization with cached public/LLM bounds. The first two are research comparisons; empirical normalization or empirical bounds are not an end-to-end privacy guarantee. In the public-bound DP pipeline, `mu=0.4` allocates `0.24` to preprocessing and `0.32` to prediction or private HPO plus training. Preprocessing noise uses fresh randomness. The dataset loaders and outer split conventions are for public benchmark reproduction; deployment still requires a public feature/class schema and separately accounted preprocessing.

Case-study caches include column descriptions and bounds evidence used by the Markdown report.

## Verification

```bash
python -m pytest -q
```

Tests cover checkpoint logits with normalization on and off, strict weight loading, training gradients, bundle serialization, query batching, privacy-budget allocation, label-independent splitting, per-example gradients, private selection and final training for both baselines, case-study workflows on synthetic datasets, and per-mu/pooled ELO calculations. Checkpoint tests use the included weights and synthetic inputs.
