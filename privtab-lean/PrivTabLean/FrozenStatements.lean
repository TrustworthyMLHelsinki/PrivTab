import Mathlib

/-!
# Frozen DP-MHCA definitions and theorem statements

The model and privacy interfaces live here independently of their proofs.
The propositions in `Frozen` state the three proof obligations without placeholders
or custom axioms. This entire file is checksum-pinned by `scripts/frozen.sha256`;
the proofs and their checks live in `DPMHCA.lean` and `ProofAudit.lean`.
-/

open scoped BigOperators

namespace PrivTabLean

/-! ## DP-MHCA and its sensitivity -/

abbrev Vec (d : ℕ) := EuclideanSpace ℝ (Fin d)
abbrev Dataset (n d : ℕ) := Fin n → Vec d
abbrev DPMHCAOutput (H m d : ℕ) := Fin H → Fin m → Vec d

/-- Fixed-size substitute adjacency: exactly one row differs. -/
def SubstituteAdjacent (D D' : Dataset n d) : Prop :=
  ∃ changed : Fin n,
    D changed ≠ D' changed ∧ ∀ i, i ≠ changed → D i = D' i

/-- Parameters of a DP-MHCA layer. Query tokens may depend on an earlier private state;
keys and values are computed row-wise from the private context. -/
structure DPMHCASpec (State : Type) (dTok dKey dValue m H : ℕ) where
  queryTokens : State → Fin m → Vec dTok
  queryProjection : Fin H → Vec dTok → Vec dKey
  keyProjection : Fin H → Vec dTok → Vec dKey
  /-- The value transformation includes normalization or clipping to the unit ball. -/
  normalizedValue : Fin H → Vec dTok → Vec dValue
  score : Vec dKey → Vec dKey → ℝ
  /-- These are the assumptions for the DPMHCA to bound the contribution of an individual row. -/
  value_norm_le_one : ∀ h x, ‖normalizedValue h x‖ ≤ 1
  score_abs_le_one : ∀ q k, |score q k| ≤ 1

/-- One private row's contribution to a single head and query output. -/
def rowContribution
    (spec : DPMHCASpec State dTok dKey dValue m H)
    (state : State) (head : Fin H) (queryIndex : Fin m) (row : Vec dTok) : Vec dValue :=
  spec.score
      (spec.queryProjection head (spec.queryTokens state queryIndex))
      (spec.keyProjection head row) •
    spec.normalizedValue head row

/-- The concatenated deterministic output before noise is added. -/
def preNoiseDPMHCA
    (spec : DPMHCASpec State dTok dKey dValue m H)
    (state : State) : Dataset n dTok → DPMHCAOutput H m dValue :=
  fun data head queryIndex => ∑ row, rowContribution spec state head queryIndex (data row)

/-- Squared ℓ₂ distance after concatenating every head and query output. -/
noncomputable def outputDistanceSquared (x y : DPMHCAOutput H m d) : ℝ :=
  ∑ head, ∑ queryIndex, ‖x head queryIndex - y head queryIndex‖ ^ 2

/-- A query has ℓ₂ sensitivity at most `Δ`, expressed in squared form. -/
noncomputable def HasL2Sensitivity
    (query : Dataset n dTok → DPMHCAOutput H m dValue) (Δ : NNReal) : Prop :=
  ∀ ⦃data data'⦄, SubstituteAdjacent data data' →
    outputDistanceSquared (query data) (query data') ≤ (Δ : ℝ) ^ 2

/-- The sensitivity bound for `H` heads and `m` summary queries. -/
noncomputable def dpMhcaSensitivityBound (H m : ℕ) : NNReal :=
  2 * NNReal.sqrt ((H * m : ℕ) : NNReal)

/-! ## Privacy accounting for the DP-MHCA encoder -/

/-- A randomized mechanism represented by the probabilistic effect `Prob`. -/
abbrev Mechanism (Prob : Type → Type) (Input Output : Type) := Input → Prob Output

/-- Run one private mechanism, then choose the next one from its private output. -/
def adaptiveCompose [Monad Prob]
    (first : Mechanism Prob Input First)
    (next : First → Mechanism Prob Input Second) :
    Mechanism Prob Input Second :=
  fun input => do
    let firstOutput ← first input
    next firstOutput input

/-- Deterministic post-processing without further access to private data. -/
def postprocess [Monad Prob]
    (mechanism : Mechanism Prob Input Output) (f : Output → Result) :
    Mechanism Prob Input Result :=
  fun input => do
    let output ← mechanism input
    pure (f output)

/-- A data-independent mechanism. -/
def constMechanism [Monad Prob] (output : Output) : Mechanism Prob Input Output :=
  fun _ => pure output

/-- Tight composition of two μ-GDP parameters. -/
noncomputable def composeMu (μ₁ μ₂ : NNReal) : NNReal :=
  NNReal.sqrt (μ₁ ^ 2 + μ₂ ^ 2)

/-- The μ-GDP rules used by PrivTab. A concrete instance must justify them from a
probabilistic definition of μ-GDP. -/
class MuGDPRules (Prob : Type → Type) [Monad Prob] (Input : Type) where
  /-- `IsPrivate mechanism μ` means that `mechanism` satisfies μ-GDP. -/
  IsPrivate : {Output : Type} → Mechanism Prob Input Output → NNReal → Prop
  /-- Adaptive μ-GDP composition combines privacy parameters in Euclidean norm. -/
  adaptive_compose :
    ∀ {First Second : Type}
      {first : Mechanism Prob Input First}
      {next : First → Mechanism Prob Input Second}
      {μ₁ μ₂ : NNReal},
      IsPrivate first μ₁ →
      (∀ output, IsPrivate (next output) μ₂) →
      IsPrivate (adaptiveCompose first next) (composeMu μ₁ μ₂)
  /-- Deterministic post-processing does not increase privacy loss. -/
  postprocess :
    ∀ {Output Result : Type}
      {mechanism : Mechanism Prob Input Output}
      {f : Output → Result} {μ : NNReal},
      IsPrivate mechanism μ → IsPrivate (PrivTabLean.postprocess mechanism f) μ
  /-- A mechanism that ignores the private input has zero privacy cost. -/
  const :
    ∀ {Output : Type} (output : Output),
      IsPrivate (constMechanism (Prob := Prob) (Input := Input) output) 0

/-- Trusted bridge for the continuous Gaussian mechanism. The intended implementation
adds isotropic Gaussian noise with standard deviation `Δ / μ`. -/
class DPMHCAGaussianMechanism
    {Prob : Type → Type} [Monad Prob]
    {n dTok dValue m H : ℕ}
    (gdp : MuGDPRules Prob (Dataset n dTok)) where
  /-- Apply Gaussian noise calibrated to sensitivity `Δ` and parameter `μ`. -/
  privatize :
    (query : Dataset n dTok → DPMHCAOutput H m dValue) →
    (Δ μ : NNReal) → Mechanism Prob (Dataset n dTok) (DPMHCAOutput H m dValue)
  /-- The trusted Gaussian-mechanism theorem. -/
  isPrivate :
    ∀ {query : Dataset n dTok → DPMHCAOutput H m dValue} {Δ μ : NNReal},
      HasL2Sensitivity query Δ → 0 < μ →
      gdp.IsPrivate (privatize query Δ μ) μ

/-- One encoder layer: a private DP-MHCA release followed by deterministic
post-processing. `postprocess` abstracts the repeated MHSA, normalization, MLP,
and other operations that do not read the private dataset directly. -/
structure DPMHCAEncoderLayer (State : Type) (dTok dKey dValue m H : ℕ) where
  dpMhca : DPMHCASpec State dTok dKey dValue m H
  postprocess : DPMHCAOutput H m dValue → State → State

section EncoderPrivacy

variable
    {Prob : Type → Type} [Monad Prob]
    [gdp : MuGDPRules Prob (Dataset n dTok)]
    [gaussian : DPMHCAGaussianMechanism (H := H) (m := m) (dValue := dValue) gdp]

/- The initial state and model parameters are fixed independently of the private dataset.
Subsequent states are computed from private releases. -/

/-- Run one DP-MHCA release and then its architectural post-processing. -/
noncomputable def runPrivateDPMHCALayer
    (layer : DPMHCAEncoderLayer State dTok dKey dValue m H)
    (state : State) (perLayerMu : NNReal) :
    Mechanism Prob (Dataset n dTok) State :=
  postprocess
    (gaussian.privatize
      (preNoiseDPMHCA layer.dpMhca state)
      (dpMhcaSensitivityBound H m) perLayerMu)
    (fun privateOutput => layer.postprocess privateOutput state)

/-- Run every private layer and return the final post-processed encoder state. -/
noncomputable def privateDPMHCAEncoder
    (layers : List (DPMHCAEncoderLayer State dTok dKey dValue m H))
    (state : State) (perLayerMu : NNReal) :
    Mechanism Prob (Dataset n dTok) State :=
  match layers with
  | [] => constMechanism state
  | layer :: remainingLayers =>
      adaptiveCompose
        (runPrivateDPMHCALayer layer state perLayerMu)
        (fun nextState =>
          privateDPMHCAEncoder remainingLayers nextState perLayerMu)

end EncoderPrivacy

/-! ## Frozen theorem statements -/

namespace Frozen

def preNoiseDPMHCA_hasL2Sensitivity : Prop :=
  ∀ {State : Type} {dTok dKey dValue m H n : ℕ}
    (spec : DPMHCASpec State dTok dKey dValue m H) (state : State),
    HasL2Sensitivity (preNoiseDPMHCA (n := n) spec state)
      (dpMhcaSensitivityBound H m)

def privateDPMHCAEncoder_muGDP : Prop :=
  ∀ {n dTok dValue m H : ℕ}
    {Prob : Type → Type} [Monad Prob]
    [gdp : MuGDPRules Prob (Dataset n dTok)]
    [_gaussian : DPMHCAGaussianMechanism (H := H) (m := m) (dValue := dValue) gdp]
    {State : Type} {dKey : ℕ}
    (layers : List (DPMHCAEncoderLayer State dTok dKey dValue m H))
    (_atLeastOneLayer : 0 < layers.length) (initialState : State)
    (μ : NNReal) (_positiveMu : 0 < μ),
    gdp.IsPrivate
      (privateDPMHCAEncoder layers initialState (μ / NNReal.sqrt layers.length)) μ

def privTabPrediction_muGDP : Prop :=
  ∀ {n dTok dValue m H : ℕ}
    {Prob : Type → Type} [Monad Prob]
    [gdp : MuGDPRules Prob (Dataset n dTok)]
    [_gaussian : DPMHCAGaussianMechanism (H := H) (m := m) (dValue := dValue) gdp]
    {State : Type} {dKey : ℕ} {Prediction : Type}
    (layers : List (DPMHCAEncoderLayer State dTok dKey dValue m H))
    (_atLeastOneLayer : 0 < layers.length) (initialState : State)
    (predict : State → Prediction) (μ : NNReal) (_positiveMu : 0 < μ),
    gdp.IsPrivate
      (postprocess
        (privateDPMHCAEncoder layers initialState (μ / NNReal.sqrt layers.length)) predict) μ

end Frozen

end PrivTabLean
