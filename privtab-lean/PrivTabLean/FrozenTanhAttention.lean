import PrivTabLean.FrozenStatements

/-!
# Concrete real-valued tanh attention

These definitions mirror the private attention sum in `privtab/model.py`: a scaled
query-key inner product, `tanh` scores, and PyTorch-style L2 value normalization
with default epsilon `1e-12`. Projections are arbitrary fixed row-wise maps.
The theorem statement below is pinned separately from its proof.
-/

namespace PrivTabLean

/-- The real-number counterpart of PyTorch's default `F.normalize` epsilon. -/
noncomputable def defaultNormalizeEps : ℝ := 1 / (10 ^ 12 : ℝ)

/-- Value normalization `v / max (‖v‖, eps)` used by private attention. -/
noncomputable def torchL2Normalize (v : Vec d) : Vec d :=
  (max ‖v‖ defaultNormalizeEps)⁻¹ • v

/-- The scaled dot-product score followed by `tanh`. -/
noncomputable def scaledTanhScore (q k : Vec d) : ℝ :=
  Real.tanh (inner ℝ q k / Real.sqrt (d : ℝ))

/-- The exact real-valued pre-noise row sum for tanh attention. The projections
may include fixed row-wise operations such as layer normalization. -/
noncomputable def tanhAttentionPreNoise
    (queries : State → Fin m → Vec dTok)
    (queryProjection : Fin H → Vec dTok → Vec dKey)
    (keyProjection : Fin H → Vec dTok → Vec dKey)
    (valueProjection : Fin H → Vec dTok → Vec dValue)
    (state : State) : Dataset n dTok → DPMHCAOutput H m dValue :=
  fun data head queryIndex =>
    ∑ row, scaledTanhScore
      (queryProjection head (queries state queryIndex))
      (keyProjection head (data row)) •
      torchL2Normalize (valueProjection head (data row))

namespace Frozen

/-- Universal replace-one sensitivity for the concrete tanh/normalized-value sum. -/
def tanhAttention_hasL2Sensitivity : Prop :=
  ∀ {State : Type} {dTok dKey dValue m H n : ℕ}
    (queries : State → Fin m → Vec dTok)
    (queryProjection : Fin H → Vec dTok → Vec dKey)
    (keyProjection : Fin H → Vec dTok → Vec dKey)
    (valueProjection : Fin H → Vec dTok → Vec dValue)
    (state : State),
    HasL2Sensitivity
      (tanhAttentionPreNoise (n := n)
        queries queryProjection keyProjection valueProjection state)
      (dpMhcaSensitivityBound H m)

end Frozen

end PrivTabLean
