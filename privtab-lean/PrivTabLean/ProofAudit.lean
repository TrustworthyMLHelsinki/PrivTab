import PrivTabLean.TanhAttention
import Mathlib.Util.AssertNoSorry

/-!
# DP-MHCA proof audit

This module checks that the proved theorems have the checksum-pinned statements
and depend only on the explicitly approved foundational axioms.
-/

namespace PrivTabLean

example : Frozen.preNoiseDPMHCA_hasL2Sensitivity :=
  @PrivTabLean.preNoiseDPMHCA_hasL2Sensitivity

example : Frozen.tanhAttention_hasL2Sensitivity :=
  @PrivTabLean.tanhAttention_hasL2Sensitivity

example : Frozen.privateDPMHCAEncoder_muGDP :=
  @PrivTabLean.privateDPMHCAEncoder_muGDP

example : Frozen.privTabPrediction_muGDP :=
  @PrivTabLean.privTabPrediction_muGDP

open Lean Meta Elab Command

/-- Fail unless a theorem depends only on the explicitly approved foundational axioms. -/
elab "assert_only_allowed_axioms " theoremName:ident : command => do
  let declarationName ←
    liftCoreM <| Lean.Elab.realizeGlobalConstNoOverloadWithInfo theoremName
  let usedAxioms ← Lean.collectAxioms declarationName
  let allowedAxioms := [``propext, ``Classical.choice, ``Quot.sound]
  for axiomName in usedAxioms do
    unless allowedAxioms.contains axiomName do
      throwError "{theoremName} uses unapproved axiom {axiomName}"

assert_no_sorry PrivTabLean.preNoiseDPMHCA_hasL2Sensitivity
assert_no_sorry PrivTabLean.tanhAttention_hasL2Sensitivity
assert_no_sorry PrivTabLean.privateDPMHCAEncoder_muGDP
assert_no_sorry PrivTabLean.privTabPrediction_muGDP

assert_only_allowed_axioms PrivTabLean.preNoiseDPMHCA_hasL2Sensitivity
assert_only_allowed_axioms PrivTabLean.tanhAttention_hasL2Sensitivity
assert_only_allowed_axioms PrivTabLean.privateDPMHCAEncoder_muGDP
assert_only_allowed_axioms PrivTabLean.privTabPrediction_muGDP

#print axioms PrivTabLean.preNoiseDPMHCA_hasL2Sensitivity
#print axioms PrivTabLean.tanhAttention_hasL2Sensitivity
#print axioms PrivTabLean.privateDPMHCAEncoder_muGDP
#print axioms PrivTabLean.privTabPrediction_muGDP

end PrivTabLean
