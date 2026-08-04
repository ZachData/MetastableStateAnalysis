"""
p2_eigenspectra/visualization — Phase 2 checkpoint figures.

Three classes, one entry point (`python -m p2_eigenspectra.visualization`):

  spectra  — OV eigenspectrum vs. depth vs. training step. Weights only,
             so 27 checkpoints rather than 27 × 9 runs and no prompt
             dependence.
  scalars  — one number per checkpoint vs. log(step), plus the verdict-side
             scalars as a median-with-IQR over prompts. Writes
             transitions_p2_{base}.json.
  clouds   — the raw complex-plane spectrum as a filmstrip across the
             checkpoints the scalars flagged.

Step-axis, colormap, baseline-resolution, and family-grouping conventions
are imported from p1_mstate_tracking/visualization/checkpoints.py rather
than restated, so the two phases' checkpoint figures cannot drift apart.
Analysis logic is likewise imported from p2_eigenspectra, never copied:
this package contains figure code only, and anything that decides what a
number MEANS (zone thresholds, violation criteria) stays in the phase
package where the verdict reads it.
"""
