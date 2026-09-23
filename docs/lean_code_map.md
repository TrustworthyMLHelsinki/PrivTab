# PrivTab: Lean proof to Python code map

This guide explains every definition, lemma, and theorem in the [`PrivTabLean`](../privtab-lean/README.md) proof, and identifies the Python operation each one models. Line references below refer to the files in this repository.

The proof establishes that the exact real-valued scaled-tanh attention sum with epsilon-normalized values has universal replace-one sensitivity at most $2\sqrt{Hm}$ for fixed queries and row-wise projections. It also shows that $L$ adaptive private releases compose to a total parameter $\mu$ when each gets $\mu/\sqrt L$. PrivTab uses $H=1$, $m=128$, and $L=3$, giving sensitivity $2\sqrt{128}$ and noise standard deviation $2\sqrt{128\cdot3}/\mu$. The Lean privacy theorem is **conditional** on abstract GDP composition and Gaussian-mechanism rules. It does not certify Python execution, floating-point Gaussian sampling, or data preprocessing.

The generic declarations and proofs are in [`FrozenStatements.lean`](../privtab-lean/PrivTabLean/FrozenStatements.lean) and [`DPMHCA.lean`](../privtab-lean/PrivTabLean/DPMHCA.lean). The concrete formula and proof are in [`FrozenTanhAttention.lean`](../privtab-lean/PrivTabLean/FrozenTanhAttention.lean) and [`TanhAttention.lean`](../privtab-lean/PrivTabLean/TanhAttention.lean). Exact-statement and axiom checks are in [`ProofAudit.lean`](../privtab-lean/PrivTabLean/ProofAudit.lean). Python counterparts are mainly [`model.py`](../privtab/model.py), [`release.py`](../privtab/release.py), and [`secure_noise.py`](../privtab/secure_noise.py).

## From rows to the bounded attention sum

### `Vec`, `Dataset`, `DPMHCAOutput`

*Lean definitions, `FrozenStatements.lean:18–20`.*

`Vec d` is a real Euclidean vector, `Dataset n d` is a fixed-length sequence of $n$ such rows, and `DPMHCAOutput H m d` holds one output vector per attention head and latent query. In Python, an encoded labeled context row has width 256, and private attention produces a tensor with shape `[B, 1, 128, 256]` before the head dimension is flattened (`model.py:37–43`). The Lean `Dataset` represents the **encoded** context tokens, not the raw feature matrix or the target rows.

### `SubstituteAdjacent`

*Lean definition, `FrozenStatements.lean:22–25`.*

Two fixed-size datasets are adjacent when exactly one row differs. This is the replace-one relationship used by the sensitivity bound. For raw labeled datasets to inherit it, the feature transform and `Encoder.encode` (`model.py:116–121`) must preserve row independence for a public feature count and schema. The Lean project assumes encoded-row adjacency; it does not prove that the Python preprocessing pipeline preserves it. [`audit_rowwise.py`](../experiments/audit_rowwise.py) checks important instances empirically.

### `DPMHCASpec`

*Lean structure, `FrozenStatements.lean:27–38`.*

It describes state-dependent latent queries, row-wise key and value transforms, a score, and two bounds: value norm at most one and absolute score at most one. In `Attention.forward` (`model.py:33–42`), `to_q`, `to_k`, and `to_v` supply the projections; `F.normalize(v, p=2, dim=-1)` bounds values; and `tanh(q @ kᵀ / sqrt(256))` bounds scores. `AttentionLayer.forward` applies layer normalization separately to each context token before those projections (`model.py:61–64`). The generic specification carries these bounds as fields; the concrete real-valued theorem below proves them for tanh and epsilon normalization. Neither theorem asserts a bound on PyTorch's finite-precision operations.

### `rowContribution`

*Lean definition, `FrozenStatements.lean:40–47`.*

One row contributes `score(query, key(row)) * normalizedValue(row)` to one head/query output. It is the individual term in the matrix product `tanh(...) @ v` (`model.py:40`). The latent query may depend on earlier noisy states, but it is held fixed when bounding the sensitivity of this layer.

### `preNoiseDPMHCA`

*Lean definition, `FrozenStatements.lean:49–53`.*

It sums `rowContribution` over context rows before noise. This corresponds to `out` immediately after `tanh(...) @ v` in `Attention.forward` (`model.py:40–43`). There is no softmax over private rows.

### `outputDistanceSquared`

*Lean definition, `FrozenStatements.lean:55–57`.*

It adds squared Euclidean differences over all heads and latent queries. This is the squared norm of the flattened attention output, the quantity calibrated by isotropic Gaussian noise.

### `HasL2Sensitivity`

*Lean definition, `FrozenStatements.lean:59–63`.*

For all adjacent encoded contexts, it requires `outputDistanceSquared <= Δ²`. It is a mathematical property of the pre-noise query; it is not a runtime assertion. The concrete theorem below proves this universal bound for the real-valued tanh formula. The sensitivity search in [`dp_audit.py`](../experiments/dp_audit.py) complements the proof by checking actual Python execution at sampled inputs.

### `dpMhcaSensitivityBound`

*Lean definition, `FrozenStatements.lean:65–67`.*

It sets $\Delta=2\sqrt{Hm}$. For PrivTab's one private head and 128 latents, this is $2\sqrt{128}$ (`model.py:25–27,70–72`). The bound does not scale with context length because a replace-one change affects only one row's contribution.

## Exact real-valued tanh attention

### `defaultNormalizeEps`, `torchL2Normalize`

*Lean definitions, `FrozenTanhAttention.lean:15–19`.*

They model PyTorch's default `F.normalize` epsilon of $10^{-12}$ and its value rule $v/\max(\|v\|_2,10^{-12})$ (`model.py:39`). The zero vector remains zero. `norm_torchL2Normalize_le_one` (`TanhAttention.lean:13–25`) proves the norm is at most one for every real vector, including vectors below the epsilon threshold.

### `scaledTanhScore`

*Lean definition, `FrozenTanhAttention.lean:22–23`.*

This is $\tanh(\langle q,k\rangle/\sqrt{d_{\rm key}})$, matching the scaled matrix product in `Attention.forward` (`model.py:40`). `abs_scaledTanhScore_le_one` (`TanhAttention.lean:28–30`) proves the score bound for every real query and key, regardless of their norms.

### `tanhAttentionPreNoise`

*Lean definition, `FrozenTanhAttention.lean:27–37`.*

It sums `scaledTanhScore(q,k(row)) * torchL2Normalize(v(row))` over the encoded context rows for every head and latent query. Its four function arguments represent state-dependent queries and arbitrary fixed row-wise query, key, and value projections, including learned weights. This matches the pre-noise `tanh(...) @ v` operation (`model.py:33–43`). Row-wise context transformations can be absorbed into the key and value projections; data-dependent cross-row transformations cannot.

### `tanhAttention_hasL2Sensitivity`

*Frozen statement, `FrozenTanhAttention.lean:42–52`; proved theorem, `TanhAttention.lean:53–72`.*

For every fixed state, projections, and pair of replace-one adjacent encoded datasets, this theorem proves sensitivity $2\sqrt{Hm}$ for the exact real-valued tanh attention formula. `tanhDPMHCASpec` (`TanhAttention.lean:32–49`) builds the generic specification using the proved tanh and normalization bounds, and the theorem identifies its pre-noise sum with `tanhAttentionPreNoise`. Thus the generic sensitivity theorem applies without assuming bounded scores or values as extra premises for this formula. For PrivTab, $H=1$ and $m=128$.

## Sensitivity proof: each step

### `difference_of_sums_when_one_entry_changes`

*private lemma, `DPMHCA.lean:18–27`.*

If two sequences agree except at one position, subtracting their sums cancels every unchanged term. Its Python counterpart is the row-sum matrix multiplication in `model.py:40`; the cancellation requires independent row contributions.

### `row_contribution_norm_le_one`

*private lemma, `DPMHCA.lean:30–46`.*

One row contributes a vector with norm at most $|\text{score}|\,\|v\|\le1$. It uses the two bounds supplied by `DPMHCASpec`. The corresponding operations are `F.normalize` and `tanh` (`model.py:39–40`).

### `head_query_output_difference_norm_le_two`

*private lemma, `DPMHCA.lean:49–72`.*

Under replace-one adjacency, the output for any one head and latent query changes by at most two: one bounded row contribution is removed and another is added. Its code-level subject is one latent output vector from the private attention sum (`model.py:40–43`).

### `full_output_difference_squared_le_four_times_heads_times_queries`

*private lemma, `DPMHCA.lean:75–98`.*

There are $H m$ head/query vectors; each has squared change at most four. Their concatenated squared change is therefore at most $4Hm$. In Python this is the full pre-noise tensor, with $H=1,m=128$ (`model.py:25–27,43`).

### `dpMhca_sensitivity_bound_squared`

*private lemma, `DPMHCA.lean:101–108`.*

It proves the arithmetic identity $(2\sqrt{Hm})^2=4Hm$, connecting the tensor bound to `dpMhcaSensitivityBound`.

### `preNoiseDPMHCA_hasL2Sensitivity`

*public theorem, `DPMHCA.lean:111–124`.*

Combining the preceding lemmas, every `DPMHCASpec` satisfying its bounds has replace-one sensitivity at most $2\sqrt{Hm}$ for a fixed state. `Frozen.preNoiseDPMHCA_hasL2Sensitivity` (`FrozenStatements.lean:187–191`) pins this exact statement. It supports the sensitivity used in `Perceiver.summarize` (`model.py:77–83`), provided the Python row encoder and bounded attention meet the abstract premises.

## Private layers and adaptive composition

### `Mechanism`

*Lean abbreviation, `FrozenStatements.lean:71–72`.*

A randomized mechanism maps a private input to an abstract probabilistic result. Python's `release_summary` (`model.py:173–186`) is the intended concrete release, but no Python mechanism is instantiated in Lean.

### `adaptiveCompose`

*Lean definition, `FrozenStatements.lean:74–81`.*

It runs one private mechanism and allows the next mechanism to depend on its output. This captures `Perceiver.summarize`, where the latent queries in later private layers depend on earlier noisy latent states (`model.py:80–84`).

### `postprocess` and `constMechanism`

*Lean definitions, `FrozenStatements.lean:83–93`.*

`postprocess` applies a deterministic function to an existing private output; `constMechanism` ignores the private dataset. In Python, normalization of the **already-noisy** attention output (`model.py:47–49`), residual/feed-forward updates (`model.py:61–64`), latent self-attention (`model.py:83–87`), and query prediction (`model.py:90–94,188–200`) are intended post-processing. The learned initial latents (`model.py:70,80`) are the intended data-independent initial state.

### `composeMu`

*Lean definition, `FrozenStatements.lean:95–97`.*

It combines two GDP parameters as $\sqrt{\mu_1^2+\mu_2^2}$. The three private layers therefore use $\mu/\sqrt3$ each. This is why Python sets `sigma = 2 * sqrt(128 * 3) / mu` (`model.py:77–79`).

### `MuGDPRules`

*assumed Lean interface, `FrozenStatements.lean:99–122`.*

Its `IsPrivate`, `adaptive_compose`, `postprocess`, and `const` fields state the GDP rules used by the proof. Lean does **not** construct a probability space or prove these rules from a concrete GDP definition. They are assumptions that a concrete probabilistic instance must justify.

### `DPMHCAGaussianMechanism`

*assumed Lean interface, `FrozenStatements.lean:124–138`.*

`privatize` represents adding isotropic Gaussian noise with intended standard deviation $\Delta/\mu_{\rm layer}$; `isPrivate` supplies the Gaussian-mechanism privacy theorem. Python multiplies standard-normal samples by `sigma` (`model.py:44–46`) and uses OS-backed Box–Muller samples for private releases (`secure_noise.py:14–34`). Lean does not prove that this floating-point sampler implements an ideal continuous Gaussian.

### `DPMHCAEncoderLayer`

*Lean structure, `FrozenStatements.lean:140–145`.*

It pairs a private attention specification with arbitrary deterministic post-processing of its noisy output and prior state. One Python iteration is `private(...)` followed by latent self-attention (`model.py:82–84`); the private attention layer itself also performs optional noisy-output normalization and a residual/feed-forward update (`model.py:44–64`).

### `runPrivateDPMHCALayer`

*Lean definition, `FrozenStatements.lean:157–166`.*

It applies the abstract Gaussian mechanism to a sensitive attention sum, then post-processes that release. This is the one-layer model for the preceding Python operations. The Lean definition does not itself generate random bytes.

### `privateDPMHCAEncoder`

*Lean definition, `FrozenStatements.lean:168–179`.*

It chains any number of private layers adaptively. Python has three such layers, then two context-free latent self-attention stages (`model.py:82–88`). The abstract Lean `State` can include both current latents and a history of saved states, so a five-state summary can be represented; the project does **not** define a concrete `State` equal to Python's `[B,5,128,256]` tensor or prove a correspondence theorem for it.

### `compose_one_with_k_equal_budgets`

*private theorem, `DPMHCA.lean:136–143`.*

It proves the GDP algebra for adding one layer to $k$ equal-budget layers: $\sqrt{\mu_0^2+k\mu_0^2}=\sqrt{(k+1)\mu_0^2}$.

### `private_encoder_with_equal_per_layer_budget`

*private theorem, `DPMHCA.lean:146–169`.*

Induction proves that $L$ adaptive layers at per-layer parameter $\mu_0$ have parameter $\sqrt L\,\mu_0$. It uses the assumed Gaussian theorem, adaptive composition, and post-processing rule. The induction generalizes over every state, which is what allows subsequent queries to depend on previous releases.

### `privateDPMHCAEncoder_muGDP`

*public theorem, `DPMHCA.lean:172–187`.*

For positive $\mu$ and $L>0$, substitute $\mu_0=\mu/\sqrt L$ into the induction result to get a total $\mu$-GDP encoder. `Frozen.privateDPMHCAEncoder_muGDP` (`FrozenStatements.lean:193–203`) pins the statement. Python has $L=3$ and computes the corresponding noise scale (`model.py:77–83`).

### `privTabPrediction_muGDP`

*public theorem, `DPMHCA.lean:190–201`.*

A deterministic prediction function applied to the private encoder state remains $\mu$-GDP. `Frozen.privTabPrediction_muGDP` (`FrozenStatements.lean:205–216`) pins the statement. Python's `predict_from_summary` and `ReleasedSummary.predict_proba` (`model.py:188–200`; `release.py:61–78`) read the saved summary without reading context data. The theorem is for a fixed public query/prediction function; it does not cover private, data-dependent query selection outside the model.

## What the audit checks, and what remains outside the proof

`ProofAudit.lean:13–23` checks that all four proved theorems, including the concrete tanh theorem, inhabit their checksum-pinned `Frozen` statements. It also rejects proof placeholders and unexpected kernel axioms; [`scripts/audit_lean.py`](../privtab-lean/scripts/audit_lean.py) verifies both frozen-file checksums, scans for banned proof shortcuts and custom axioms, and runs the Lean builds. Run it from `privtab-lean/` with `python scripts/audit_lean.py`.

The concrete sensitivity theorem is **universal for the specified real-valued tanh attention formula**; its score and value bounds are proved rather than assumed. The GDP result still relies on trusted GDP/Gaussian interfaces. Correspondence to the Python release depends on independently checking that (1) preprocessing and feature/class schema do not introduce unaccounted private cross-row effects; (2) trained weights and the initial state are fixed independently of the private context; (3) the executed finite-precision attention respects the modeled bounds; (4) each private layer receives independent noise at the specified scale; and (5) all released intermediate states are included in the composed private state. The [row-wise audit](../experiments/audit_rowwise.py) and [noise audit](../experiments/dp_audit.py) test parts of this correspondence. They complement the universal mathematical sensitivity proof; they do not establish a bit-level DP guarantee.
