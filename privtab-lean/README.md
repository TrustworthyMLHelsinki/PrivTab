# PrivTab Lean formalization

This Lean project proves the universal replace-one sensitivity bound for the
real-valued tanh attention formula used by PrivTab and formalizes its privacy
accounting.

The [proof-to-code map](../docs/lean_code_map.md) explains each definition, lemma, and theorem alongside the PrivTab Python operation it models. The real-valued sensitivity proof is unconditional for the specified attention formula. The GDP privacy result uses abstract Gaussian-mechanism and composition rules; the map states the remaining code-level assumptions.

The generic definitions and privacy interfaces are in
[`PrivTabLean/FrozenStatements.lean`](PrivTabLean/FrozenStatements.lean), with
their proofs in [`PrivTabLean/DPMHCA.lean`](PrivTabLean/DPMHCA.lean). The exact
real-valued tanh score, scaled query-key inner product, and epsilon-normalized
value rule are in [`PrivTabLean/FrozenTanhAttention.lean`](PrivTabLean/FrozenTanhAttention.lean),
with proofs in [`PrivTabLean/TanhAttention.lean`](PrivTabLean/TanhAttention.lean):

1. **Concrete sensitivity.** The proof establishes the score and value bounds
   for the exact real-valued tanh/normalization formulas, then proves that their
   deterministic pre-noise sum has concatenated ℓ₂ sensitivity at most
   `2 * sqrt (H * m)` for every fixed state and fixed row-wise projections.
2. **Encoder privacy.** Each encoder layer explicitly performs a private DP-MHCA
   release followed by deterministic post-processing such as MHSA, normalization,
   or an MLP. Lean proves that `L` such adaptive layers are μ-GDP when each receives
   budget `μ / sqrt L`, and that the final PrivTab prediction remains μ-GDP.

`MuGDPRules` contains the standard composition and post-processing rules.
`DPMHCAGaussianMechanism` isolates the continuous Gaussian mechanism theorem; the
development does not construct Gaussian probability measures or verify a sampler.

From `privtab-lean/`, build the project with:

```bash
lake build
```

## Proof audit

The audit uses two frozen statement files and a kernel check:

- [`PrivTabLean/FrozenStatements.lean`](PrivTabLean/FrozenStatements.lean) records the
  generic model definitions, privacy interfaces, and three main theorem
  statements as proof-free propositions.
- [`PrivTabLean/FrozenTanhAttention.lean`](PrivTabLean/FrozenTanhAttention.lean)
  records the concrete tanh/normalization formula and its sensitivity statement.
- [`PrivTabLean/ProofAudit.lean`](PrivTabLean/ProofAudit.lean) checks that the actual
  proved theorems have exactly the frozen types. It also uses `assert_no_sorry`
  and `Lean.collectAxioms` to inspect their transitive kernel dependencies.
- [`scripts/frozen.sha256`](scripts/frozen.sha256) pins both frozen statement files,
  covering both the definitions and expected statements. Changes are detected
  unless the checksum is also updated.

Run the complete audit with:

```bash
python scripts/audit_lean.py
```

The script verifies all of the following:

1. Both frozen-statement checksums have not changed.
2. Both proof files and both frozen statement files contain none of the banned proof shortcuts:
   `sorry`, `sorryAx`, `native_decide`, `admit`, `unsafe`, `implemented_by`, or
   `ofReduceBool`.
3. None of the four files contains a custom `axiom` declaration.
4. The complete project builds successfully with `lake build`.
5. Each proved theorem has exactly its checksum-pinned statement.
6. The proved theorems depend only on the explicitly allowed foundational axioms:
   `propext`, `Classical.choice`, and `Quot.sound`.

If a frozen definition or theorem statement is intentionally changed, review the change first and then
update its checksum in `scripts/frozen.sha256` using the output of:

```bash
sha256sum PrivTabLean/FrozenStatements.lean PrivTabLean/FrozenTanhAttention.lean
```
