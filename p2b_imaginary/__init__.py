"""
phase2i — Phase 2 Imaginary: Rotational Dynamics Analysis

Extends Phase 2's signed-eigenvalue analysis to characterize the ~97%
of V_eff's spectrum in complex conjugate pairs (rotation planes).

Block structure:
  Block 1a: rotational_schur.py    — Schur block extraction, Henrici non-normality
  Block 1b: rotational_rescaled.py — S/A decomposition, causal isolation
  Block 2:  fiedler_tracking.py    — Hemispheric structure tracking
            rotation_hemisphere.py — Rotation plane vs hemisphere alignment
  New:      ffn_rotation.py        — FFN interaction with rotational structure

Decision logic:
  Block 1a always runs (pure weight analysis).
  Block 1b always runs (the decision-point test).
  Block 2 runs if Block 1b shows rotation contributes.
  FFN rotation runs if Block 1b shows rotation contributes AND the model
  is in Regime B (FFN-mediated, GPT-2 large/xl).
"""
