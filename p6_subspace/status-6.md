# Phase 6 — STATUS

**Last verified:** run date recorded only as "2026-04-xx" in source (incomplete timestamp —
backfill exact date from `phase6_report.txt` file metadata if needed).
**Overall: PARTIAL RUN, MISLABELED AS "NOT STARTED."** `README_phase6.md`'s header says "Not
started," but the same file's own "First Run Results" section and `phase6_report.txt`
confirm one model (albert-xlarge-v2 only) has been run. Treat header as stale.

## Verdict table (albert-xlarge-v2 only — no other model tested)

**0/6 tested predictions passed. 6/12 predictions did not run at all.**

| Track | Status |
|---|---|
| Track A (imaginary subspace / relational computation: `head_classify`, `qk_decompose`, P6-C1, P6-DD1/DD2) | **No data.** Skipped due to missing prerequisites (`qk_logit_matrices`, `qk_matrices` never populated) and un-threaded `model`/`tokenizer`/`text`/`hook_targets` for the dissociation test. |
| P6-R2 / P6-R4 (LDA alignment, linear probes) | **Ran — result inverted from prediction.** Mean LDA alignment with $U_A$ (imaginary) = 0.887 vs. $U_\text{neg}$ (real repulsive) = 0.067. 0/49 layers show the predicted direction. Probe accuracy: real-only 0.152 (chance level), imaginary-only 0.564 (near full-activation 0.590). |
| P6-R5 (local contraction) | **Partial pass.** 29/44 plateau steps contract in real subspace; 28/44 show neutral rotation in imaginary — as predicted. Merge destabilization fails: only 121/341 merge steps show the predicted real-subspace expansion. |

## Known blockers

1. `_compute_qk_logit_matrices` in `run_6.py` not confirmed to populate `ctx["qk_matrices"]`
   / `ctx["qk_logit_matrices"]` correctly — blocks all of Track A.
2. `model`, `tokenizer`, `text`, `hook_targets` not threaded into `ctx` before the
   dissociation subexperiment registers — blocks the double-dissociation causal test (the
   single most falsifiable prediction in the phase).
3. **`eigenspace_degeneracy.py` — NameError, `d` undefined** (project-wide known bug list;
   fixed in v2 execution-order item 4, "Freeze-for-deletion, bug fixes, oracle + smoke
   tiers" — alongside phase 3's freeze, not as a phase-6-specific task. Needs a named
   regression test first, per policy, and is a candidate for the oracle-tier suite since a
   degeneracy-ratio computation has a known-correct answer on planted synthetic clusters).
4. **`write_subspace.py` — `channel_orthogonality` called with unsupported `top_r` kwarg**
   (project-wide known bug list; same item-4 fix location as above).
5. Two competing explanations for the R2/R4 inversion, neither ruled out yet: (a) a
   projector-construction error in `subspace_build.py` (Schur block mislabeling, swapping
   $U_\text{neg}$ and $U_A$) that would invert all four geometry tests together; (b) the
   real/imaginary functional-separation hypothesis genuinely doesn't hold under ALBERT's
   weight-tying, where one OV matrix implements both channels.
6. ALBERT-specific caveat on P6-R5: the 0/49-layer inversion is not 49 independent
   measurements — same projector, 49 activation snapshots from the same shared OV weights.
   Result is weaker evidence than it would be for a non-weight-tied model.

## Not yet done

- Re-run with Track A prerequisites fixed, before drawing further conclusions from R2/R4.
- Run on any non-ALBERT model — current result is ALBERT-only, and the ALBERT-specific
  weight-tying caveat (blocker 6) means the inversion needs a per-layer-weight model to be
  read as a real functional-separation failure rather than an ALBERT artifact.
- Falsification-table retrofit not needed here — this phase already has one (12-row table
  in the README), unlike Phases 1/2/5.
