import PrivTabLean.FrozenTanhAttention
import PrivTabLean.DPMHCA

/-! Bounds and sensitivity proof for the concrete tanh attention sum. -/

namespace PrivTabLean

private theorem defaultNormalizeEps_pos : 0 < defaultNormalizeEps := by
  norm_num [defaultNormalizeEps]

/-- PyTorch-style epsilon normalization keeps every real-valued value vector
inside the closed unit ball, including the zero vector. -/
theorem norm_torchL2Normalize_le_one (v : Vec d) :
    ‖torchL2Normalize v‖ ≤ 1 := by
  let denominator : ℝ := max ‖v‖ defaultNormalizeEps
  have hPositive : 0 < denominator :=
    lt_of_lt_of_le defaultNormalizeEps_pos (le_max_right _ _)
  have hBound : ‖v‖ ≤ denominator := le_max_left _ _
  have hInvNonneg : 0 ≤ denominator⁻¹ := inv_nonneg.mpr hPositive.le
  calc
    ‖torchL2Normalize v‖ = denominator⁻¹ * ‖v‖ := by
      simp [torchL2Normalize, denominator, norm_smul, Real.norm_eq_abs]
    _ ≤ denominator⁻¹ * denominator :=
      mul_le_mul_of_nonneg_left hBound hInvNonneg
    _ = 1 := inv_mul_cancel₀ hPositive.ne'

/-- `tanh` bounds every scaled query-key score independently of query/key norms. -/
theorem abs_scaledTanhScore_le_one (q k : Vec d) :
    |scaledTanhScore q k| ≤ 1 := by
  exact (Real.abs_tanh_lt_one _).le

private noncomputable def tanhDPMHCASpec
    {State : Type}
    (queries : State → Fin m → Vec dTok)
    (queryProjection : Fin H → Vec dTok → Vec dKey)
    (keyProjection : Fin H → Vec dTok → Vec dKey)
    (valueProjection : Fin H → Vec dTok → Vec dValue) :
    DPMHCASpec State dTok dKey dValue m H where
  queryTokens := queries
  queryProjection := queryProjection
  keyProjection := keyProjection
  normalizedValue := fun h row => torchL2Normalize (valueProjection h row)
  score := scaledTanhScore
  value_norm_le_one := by
    intro h row
    exact norm_torchL2Normalize_le_one (valueProjection h row)
  score_abs_le_one := by
    intro q k
    exact abs_scaledTanhScore_le_one q k

/-- The concrete real-valued attention sum used by PrivTab has sensitivity
`2 * sqrt (H * m)` for every fixed latent state and fixed row-wise projection. -/
theorem tanhAttention_hasL2Sensitivity
    {State : Type} {dTok dKey dValue m H n : ℕ}
    (queries : State → Fin m → Vec dTok)
    (queryProjection : Fin H → Vec dTok → Vec dKey)
    (keyProjection : Fin H → Vec dTok → Vec dKey)
    (valueProjection : Fin H → Vec dTok → Vec dValue)
    (state : State) :
    HasL2Sensitivity
      (tanhAttentionPreNoise (n := n)
        queries queryProjection keyProjection valueProjection state)
      (dpMhcaSensitivityBound H m) := by
  let spec := tanhDPMHCASpec queries queryProjection keyProjection valueProjection
  have hEqual :
      tanhAttentionPreNoise (n := n)
          queries queryProjection keyProjection valueProjection state =
        preNoiseDPMHCA spec state := by
    funext data head queryIndex
    rfl
  rw [hEqual]
  exact preNoiseDPMHCA_hasL2Sensitivity spec state

end PrivTabLean
