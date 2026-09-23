import PrivTabLean.FrozenStatements

/-!
# DP-MHCA sensitivity and privacy proofs

Definitions and privacy assumptions are in `FrozenStatements.lean`.
This file proves the sensitivity bound, adaptive encoder privacy, and preservation
of privacy through the final prediction.
-/

open scoped BigOperators

namespace PrivTabLean

/-! ## Sensitivity -/

/- The elementary cancellation fact at the heart of the sensitivity proof. -/
private lemma difference_of_sums_when_one_entry_changes
    (f g : Fin n → Vec d) (changed : Fin n)
    (hSame : ∀ i, i ≠ changed → f i = g i) :
    (∑ i, f i) - ∑ i, g i = f changed - g changed := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_eq_single changed
  · intro i _ hi
    rw [hSame i hi]
    exact sub_self _
  · simp

/- Each private row contributes a vector of norm at most one. -/
private lemma row_contribution_norm_le_one
    (spec : DPMHCASpec State dTok dKey dValue m H)
    (state : State) (head : Fin H) (queryIndex : Fin m) (row : Vec dTok) :
    ‖rowContribution spec state head queryIndex row‖ ≤ 1 := by
  let query := spec.queryProjection head (spec.queryTokens state queryIndex)
  let key := spec.keyProjection head row
  let value := spec.normalizedValue head row
  calc
    ‖rowContribution spec state head queryIndex row‖
        = |spec.score query key| * ‖value‖ := by
          rw [rowContribution, norm_smul, Real.norm_eq_abs]
    _ ≤ 1 * 1 := mul_le_mul
      (spec.score_abs_le_one query key)
      (spec.value_norm_le_one head row)
      (norm_nonneg value)
      (by norm_num)
    _ = 1 := by norm_num

/- Under substitute adjacency, only the changed row remains after cancellation. -/
private lemma head_query_output_difference_norm_le_two
    (spec : DPMHCASpec State dTok dKey dValue m H)
    (state : State) (hAdjacent : SubstituteAdjacent data data')
    (head : Fin H) (queryIndex : Fin m) :
    ‖preNoiseDPMHCA spec state data head queryIndex -
      preNoiseDPMHCA spec state data' head queryIndex‖ ≤ 2 := by
  obtain ⟨changed, _, hSame⟩ := hAdjacent
  have unchangedRowsHaveSameContribution : ∀ row, row ≠ changed →
      rowContribution spec state head queryIndex (data row) =
        rowContribution spec state head queryIndex (data' row) := by
    intro row rowIsUnchanged
    simp [rowContribution, hSame row rowIsUnchanged]
  rw [preNoiseDPMHCA, preNoiseDPMHCA,
    difference_of_sums_when_one_entry_changes _ _ changed
      unchangedRowsHaveSameContribution]
  calc
    ‖rowContribution spec state head queryIndex (data changed) -
        rowContribution spec state head queryIndex (data' changed)‖
      ≤ ‖rowContribution spec state head queryIndex (data changed)‖ +
          ‖rowContribution spec state head queryIndex (data' changed)‖ := norm_sub_le _ _
    _ ≤ 1 + 1 := add_le_add
      (row_contribution_norm_le_one spec state head queryIndex (data changed))
      (row_contribution_norm_le_one spec state head queryIndex (data' changed))
    _ = 2 := by norm_num

/- There are `H * m` head-query outputs, each with squared change at most four. -/
private lemma full_output_difference_squared_le_four_times_heads_times_queries
    (spec : DPMHCASpec State dTok dKey dValue m H)
    (state : State) (hAdjacent : SubstituteAdjacent data data') :
    outputDistanceSquared (preNoiseDPMHCA spec state data)
      (preNoiseDPMHCA spec state data') ≤ 4 * H * m := by
  have eachSquaredDifferenceIsAtMostFour : ∀ head queryIndex,
      ‖preNoiseDPMHCA spec state data head queryIndex -
        preNoiseDPMHCA spec state data' head queryIndex‖ ^ 2 ≤ 4 := by
    intro head queryIndex
    nlinarith [head_query_output_difference_norm_le_two spec state hAdjacent head queryIndex,
      norm_nonneg (preNoiseDPMHCA spec state data head queryIndex -
        preNoiseDPMHCA spec state data' head queryIndex)]
  unfold outputDistanceSquared
  calc
    ∑ head, ∑ queryIndex,
        ‖preNoiseDPMHCA spec state data head queryIndex -
          preNoiseDPMHCA spec state data' head queryIndex‖ ^ 2
      ≤ ∑ _ : Fin H, ∑ _ : Fin m, (4 : ℝ) := by
        apply Finset.sum_le_sum
        intro head _
        apply Finset.sum_le_sum
        intro queryIndex _
        exact eachSquaredDifferenceIsAtMostFour head queryIndex
    _ = 4 * H * m := by simp [mul_comm, mul_left_comm]

/- Arithmetic normalization of the sensitivity bound. -/
private lemma dpMhca_sensitivity_bound_squared (H m : ℕ) :
    (4 * H * m : ℝ) =
      (dpMhcaSensitivityBound H m : ℝ) ^ 2 := by
  unfold dpMhcaSensitivityBound
  norm_cast
  rw [mul_pow, NNReal.sq_sqrt]
  norm_num [Nat.cast_mul]
  ring

/-- Sensitivity of the full pre-noise DP-MHCA output. -/
theorem preNoiseDPMHCA_hasL2Sensitivity
    {n : ℕ}
    (spec : DPMHCASpec State dTok dKey dValue m H) (state : State) :
    HasL2Sensitivity (preNoiseDPMHCA (n := n) spec state)
      (dpMhcaSensitivityBound H m) := by
  intro data data' hAdjacent
  calc
    outputDistanceSquared (preNoiseDPMHCA spec state data)
      (preNoiseDPMHCA spec state data')
      ≤ 4 * H * m :=
        full_output_difference_squared_le_four_times_heads_times_queries
          spec state hAdjacent
    _ = (dpMhcaSensitivityBound H m : ℝ) ^ 2 :=
      dpMhca_sensitivity_bound_squared H m

/-! ## Encoder privacy -/

section EncoderPrivacy

variable
    {Prob : Type → Type} [Monad Prob]
    [gdp : MuGDPRules Prob (Dataset n dTok)]
    [gaussian : DPMHCAGaussianMechanism (H := H) (m := m) (dValue := dValue) gdp]

/- Composing one layer with `k` equal-budget layers gives the budget for `k + 1`. -/
private theorem compose_one_with_k_equal_budgets (k : ℕ) (perLayerMu : NNReal) :
    composeMu perLayerMu (NNReal.sqrt ((k : NNReal) * perLayerMu ^ 2)) =
      NNReal.sqrt (((k + 1 : ℕ) : NNReal) * perLayerMu ^ 2) := by
  unfold composeMu
  rw [NNReal.sq_sqrt]
  congr 1
  push_cast
  ring

/- Inductive accounting theorem before choosing a total privacy budget. -/
private theorem private_encoder_with_equal_per_layer_budget
    (layers : List (DPMHCAEncoderLayer State dTok dKey dValue m H))
    (state : State) (perLayerMu : NNReal) (positivePerLayerMu : 0 < perLayerMu) :
    gdp.IsPrivate (privateDPMHCAEncoder layers state perLayerMu)
      (NNReal.sqrt ((layers.length : NNReal) * perLayerMu ^ 2)) := by
  -- The cool thing here this is exactly what I want, the state is generalized so that the inductive hypothesis is for all possible states which gives the adaptive composition.
  induction layers generalizing state with
  | nil =>
      -- With no layers, the final state is independent of the dataset which is what I want.
      rw [privateDPMHCAEncoder]
      simpa using gdp.const (output := state)
  | cons layer layers ih =>
      -- Privatize DP-MHCA, post-process its release, then run the remaining layers.
      rw [privateDPMHCAEncoder, List.length,
        ← compose_one_with_k_equal_budgets layers.length perLayerMu]
      apply MuGDPRules.adaptive_compose
      · unfold runPrivateDPMHCALayer
        apply MuGDPRules.postprocess
        apply gaussian.isPrivate
        · exact preNoiseDPMHCA_hasL2Sensitivity layer.dpMhca state
        · exact positivePerLayerMu
      · intro nextState
        exact ih nextState

/-- An encoder with `L > 0` private layers is μ-GDP when each layer receives
budget `μ / sqrt L`. Deterministic post-processing is included after every layer. -/
theorem privateDPMHCAEncoder_muGDP
    (layers : List (DPMHCAEncoderLayer State dTok dKey dValue m H))
    (atLeastOneLayer : 0 < layers.length) (initialState : State)
    (μ : NNReal) (positiveMu : 0 < μ) :
    gdp.IsPrivate
      (privateDPMHCAEncoder layers initialState (μ / NNReal.sqrt layers.length)) μ := by
  let perLayerMu := μ / NNReal.sqrt (layers.length : NNReal)
  have positivePerLayerMu : 0 < perLayerMu := by positivity
  have encoderIsPrivate := private_encoder_with_equal_per_layer_budget
    (Prob := Prob) (gdp := gdp) layers initialState perLayerMu positivePerLayerMu
  have composedBudgetEqualsMu :
      (layers.length : NNReal) * perLayerMu ^ 2 = μ ^ 2 := by
    dsimp [perLayerMu]
    rw [← NNReal.sq_sqrt (layers.length : NNReal), NNReal.sqrt_sq]
    field_simp [pow_two, show NNReal.sqrt (layers.length : NNReal) ≠ 0 by positivity]
  rw [composedBudgetEqualsMu, NNReal.sqrt_sq] at encoderIsPrivate
  exact encoderIsPrivate

/-- Target-side MHCA, MHSA, and the prediction MLP are post-processing of the
private encoder state, so the final PrivTab prediction remains μ-GDP. -/
theorem privTabPrediction_muGDP
    (layers : List (DPMHCAEncoderLayer State dTok dKey dValue m H))
    (atLeastOneLayer : 0 < layers.length) (initialState : State)
    (predict : State → Prediction)
    (μ : NNReal) (positiveMu : 0 < μ) :
    gdp.IsPrivate
      (postprocess
        (privateDPMHCAEncoder layers initialState (μ / NNReal.sqrt layers.length)) predict) μ := by
  apply MuGDPRules.postprocess
  exact privateDPMHCAEncoder_muGDP layers atLeastOneLayer initialState μ positiveMu

end EncoderPrivacy

end PrivTabLean
