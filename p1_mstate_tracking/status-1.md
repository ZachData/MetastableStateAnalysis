# Phase 1 — STATUS

**Last verified:** 2026-04-23 (run dir `results/2026-04-23_18-30-06`)
**Overall:** Complete. Falsification criterion passed — metastability survives trained dynamics.

## Verdict table

| Prediction | Result |
|---|---|
| Tokens cluster over layers | Confirmed, universal |
| Two-timescale dynamics | Confirmed above a depth threshold (BERT-base, GPT-2-large/xl, random-weight ALBERT-base); absent/weak in GPT-2-medium, trained ALBERT-base |
| Metastable plateaus exist | Confirmed, universal |
| ALBERT cleaner than BERT/GPT-2 | Partial — ALBERT-base yes, ALBERT-xlarge resists collapse (theory-violating) |
| Higher β → stronger metastability | Confirmed |
| Higher d → faster convergence (Thm 6.1) | **Falsified** — governed by spectral radius of V, not d |
| Monotone energy $E_\beta$ (Thm 3.4) | **Falsified universally**, including under random weights |
| Metastability is architecture- not weight-determined | Confirmed via 10-seed random sweep — all architecture-level quantities seed-invariant |

## Known blockers / open items

1. ALBERT-xlarge collapse status undetermined past 48 iterations — needs 96–128-iteration runs.
2. Depth-conditioning regime shift (GPT-2-medium → GPT-2-large spectral radius drop) is unexplained.
3. Trained-vs-random contrast on `ext_sem_frac` and on two-timescale ratio: both flagged, neither quantified at matched iteration depths.
4. Final-layer LM-head contamination (gpt2-small/medium) flagged but not stripped from plots.
5. `bert-large-uncased` in registry but not in the standard run.

## Not yet done (per transition plan, v2 numbering)

Pythia rerun (item 9) is blocked on, in order: core foundations (item 2), core analysis
primitives (item 3), Pythia model support (item 5), the replication gate (item 6 — Phase 1
run at step 0 / step 143,000 / `pythia-1.4b-random` against Blog 1's pass criteria; a failed
gate stops the sweep), visualization convention (item 7), and the Pythia-410M pilot sweep
(item 8, which also fixes the final 1.4B checkpoint schedule below).

**v2 changes directly affecting this phase:**
- The checkpoint schedule below is now provisional, not final. A dense pilot on Pythia-410M
  (item 8) locates where energy-monotonicity break, Fiedler drop, and effective-rank
  collapse actually transition, and the anchors are adjusted to match — plus 2–3 reserved
  adaptive slots placed wherever the pilot shows the sharpest inter-checkpoint change.
- Two random baselines now, not one: `pythia-1.4b-random` (final-checkpoint weights,
  norm-matched randomization — the actual continuity control for Blog 1's trained-vs-random
  contrast, since GPT-NeoX's init variance-scaling differs from GPT-2's) and true step-0 init
  (a separate developmental-origin object, not a stand-in for the random control).
- `energy_decomposition.py` / `energy_attribution_aggregate.py` (this phase's visualization
  extension) get a real Pythia path now, not a skip-fallback — see Phase 2's design doc for
  why the parallel-residual architecture is an upgrade, not a loss, for this decomposition.
