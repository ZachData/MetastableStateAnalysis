<!-- trackB_circuit/lit-B.md -->
# Track B — literature scan at opening (2026-10-13)

Trigger 1 (`CLAUDE.md` "Literature scans"): Track B opens, so this precedes `design-B.md`'s freeze.
Scope: what B0 (the nudge harness) and B1 (`L5H2` as positive control) must not get wrong. Three
web searches (2026-10-13), read at abstract / summary level only unless a section says otherwise;
the project's own Phase 7 / 7d scans (`p7_motifs/lit-7.md`, `p7d_redundancy/`) are not repeated.

## 1. Previous-token heads form first, in Pythia's 512–1000 window

- *Fingerprint, Not Blueprint* (arXiv 2607.06621, 2026, preprint): on 22 public checkpoints of
  `pythia-410m` and `pythia-160m`, previous-token behaviour jumps first (0.37 at step 512 → 0.95
  at 1000), ahead of induction; the previous-token head's rotary-phase signature moves to the top
  decile by 1000, with neither leading at checkpoint resolution. **Bearing:** an independent
  measurement that B1's target switches on inside (512, 1000], the interval 7d found for `L5H2`'s
  causal effect (`p7d_redundancy/status-7d.md` "Q2"). B1's prediction is therefore a known answer
  from two sources, which is what a positive control should be.
- Singh et al., *What needs to go right for an induction head?* (arXiv 2404.07129, 2024): the
  induction circuit is a previous-token head feeding an induction head; it forms in a sharp phase
  change driven by the two heads' interaction (clamping experiments). **Bearing:** the
  signature should switch on, not ramp; a ramp would itself be a reading.
- *Predicting the Emergence of Induction Heads* (arXiv 2511.16893, 2025): Pythia's early public
  checkpoints are the only family dense enough at the transition. **Bearing:** Pythia publishes
  nothing between 512 and 1000, so B1 cannot place the onset finer than that interval.
- Stagewise data attribution (arXiv 2510.12071, 2025) retrains Pythia-14M with checkpoints every
  100 steps to resolve the transition. **Bearing:** finer placement needs own training (triage
  M1), out of B's scope.

## 2. Perturbing the residual stream: plateaus, and the random-direction baseline

- Heimersheim & Mendel, *Activation plateaus and sensitive directions in GPT-2* (2024; summarised
  in arXiv 2409.15019 and 2410.12555): model-generated activations resist small perturbations
  (a "plateau": the output barely moves until the perturbation crosses a threshold), and
  directions toward other real activations move the output sooner than random directions of equal
  size. Later work finds the plateaus around real activations but not around random points of the
  same distribution. **Bearing, the main one for B0:** the ± expansion that B0's antisymmetric
  part relies on (`tools/math_checks/b0_antisymmetric_nudge.py`) can be swamped on a plateau, where
  the response below threshold is not its first-order term. So B0 sweeps ε and reads the curve,
  and reports where the response turns on, rather than fixing one small ε.
- arXiv 2410.12555 (*Investigating sensitive directions in GPT-2: an improved baseline*): the
  standard protocol is `x ← x_base + α·d` at one site (layer 6 `resid_pre`), the effect read as
  next-token KL, against random unit directions; the baseline's choice changes conclusions about
  SAE directions. **Bearing:** B0's null (random directions of the same norm at the same site) is
  the field's convention, and its known weakness is that random directions are a weak baseline
  (they leave the activation manifold). B0 reports the rank against random directions *and* the
  ε at which each turns on, so a direction that is merely "on-manifold" is not read as specific.

## 3. Attention as an interacting particle system: no perturbation-response theory found

- Karagodin, Polyanskiy & Rigollet (arXiv 2411.04990, read for Phase 10): causal masking gives a
  particle system that is not a mean-field gradient flow, converging to one cluster for any
  `Q, K` with `V = I`. SympFormer (arXiv 2603.16535, 2026) and the mean-field well-posedness work
  treat layers as discretised particle dynamics.
- **Gap:** no paper found on the *linear response* of such a system to a single particle's
  nudge (which is what B1 measures: token i's response to token i−1's nudge through one head).
  B's readings are therefore descriptive, not tests of a theorem. **Bearing:** B1 is a positive
  control for the instrument, not a test of the particle picture.

## What changes in the design

| from | change |
|---|---|
| §1 (2607.06621, 7d) | B1's expected onset is (512, 1000], not ROADMAP's wider (512, 4000]; the wider window stays the pass condition, the narrower one is recorded beside it before any reading |
| §1 (2404.07129) | B1 labels a step on or off; a gradual rise across 512–4000 is reported as such |
| §2 (plateaus) | ε is swept (4 values per site), the turn-on ε is a readout, and the calibration gate is run at every ε |
| §2 (baseline) | the random-direction null is kept, with the rank and the turn-on ε both reported |
| §3 | no theory to test; B1 is an instrument check |

Not read: the primary Heimersheim & Mendel post (only its two follow-ups' summaries); 2607.06621
beyond its search summary. Both are cited at that level and flagged here.
