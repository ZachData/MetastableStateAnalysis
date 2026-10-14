# Track A — literature scan at opening (2026-10-14)

Trigger 1 (`CLAUDE.md` "Literature scans"): Track A opens, so this precedes `design-A.md`'s
freeze. Scope: what A1 (T_eff against t*, β per head over training) must not get wrong. Five web
searches (2026-10-14), read at abstract / summary level unless a line says otherwise; the
project's own scans (`p1c_frames/lit-1c.md`, `p1e_energy_field/lit-1e.md`,
`docs/readings/meanfield_reading_list_2026-10-01.md`) are not repeated.

## 1. The collapse time is not one number: metastability at large β

- Geshkovski, Koubbi, Polyanskiy, Rigollet, *Dynamic metastability in the self-attention model*
  (arXiv 2410.06833, 2024): collapse to one cluster happens only as t → ∞; the particles sit near
  multi-cluster configurations for times exponentially long in β, and after a time rescaling the
  energy climbs a staircase (saddle to saddle). **Bearing:** "t*" from Theorem 6.8 is the time
  of the first, orthogonal-start sigmoid; at large β it is not the time to collapse. The design's
  t* table shows the same thing in (6.9) itself (from g0 = 0, 8.7 at β 7, 86 at β 10, n = 467),
  and the `sympy` check's item 3 says why. Exact theorem conditions not read.
- Bruno, Pasqualotto, Agazzi, *Emergence of meta-stable clustering in mean-field transformer
  models* (arXiv 2410.23228) and the survey *The Mean-Field Dynamics of Transformers*
  (arXiv 2512.01868): from near-uniform starts the metastable cluster count is set by β through
  the index maximising a rescaled Gegenbauer polynomial; the argument covers small β in general
  d, large β only on the circle. **Bearing:** 1e's question (one well at β 3.5) has a candidate
  closed form here; A1b does not derive it (`design-A.md` A1b, beside), and a count at d = 1024
  is not stated in the summaries read.
- *Multiscale analysis of mean-field transformers* (arXiv 2509.25040): β growing with the number
  of tokens. **Bearing:** the long passages (n ≈ 2,000) at fixed β are the other regime; not used.

## 2. Normalisation sets the speed

- Karagodin, Ge, Polyanskiy, Rigollet, *Normalization in Attention Dynamics* (arXiv 2510.22026,
  NeurIPS 2025): read as particles on the sphere, normalisation is speed control. Under Pre-LN
  (Pythia's), token norms grow, the angular update scales with 1/norm, and clustering slows from
  exponential to polynomial in depth. **Bearing, the main one for A1:** T_eff must be measured
  as angle moved per unit of field, which already divides out the norm (1c's `h_cal`, the
  design's `h_proj`); a step counted on the raw residual without the norm would mix depth with
  norm growth. It also predicts the steps shrink with depth in a Pre-LN stack, a shape A1b can
  check.

## 3. Layers as Euler steps

- *Flowing Through Layers* (arXiv 2502.05656, 2025): the transformer update is a forward-Euler
  step of a continuous system; under Lipschitz assumptions the representation converges to an
  ODE solution as depth grows. **Bearing:** the premise of T_eff is published; nothing found
  measures T_eff on a trained model. Geshkovski et al. (arXiv 2312.10794) take the step as 1 per
  layer, which is the convention 1c's finding 1 says understates T_eff by ~5.7× on its trajectory.
- Prior on discretisation (`p1c_frames/status-1c.md` card: "the Euler-discretisation reading is
  published", §3.37) stands.

## 4. β over training

- No study found tracking per-head attention inverse temperature across Pythia's checkpoints.
  Nearest: Zhai et al., *Stabilizing Transformer Training by Preventing Attention Entropy
  Collapse* (arXiv 2303.06296; per-head entropy over training, not Pythia); *Attention rows as
  compositional data* (arXiv 2608.14712; Pythia 70m–1.4b, final checkpoints only); a unified
  critical-scaling framework for β (arXiv 2605.12697, not read past the title). **Bearing:** F0b
  (β's slope over 18 steps) is not pre-empted.

## What changed in the design because of this scan

- t* is computed from the measured start and over the head-β bracket, not quoted as 4.2 (§1).
- The step is angle per unit field, never raw displacement (§2); its depth profile is read.
- The 1e question stays *beside*: a closed form exists in §1's papers but is not yet derived here.
